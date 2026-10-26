"""
逻辑验证智能体在带噪声 FB15K-237 上的实验：抽样 → 注入负例 → 逻辑智能体打分 → 评估指标。

流程：
  1. 加载 FB15K-237 指定划分（train/valid/test/**all**），再按 --max_triples 从该划分中随机抽样正例（默认 all+5000）；**all**=三路合并为全库正例池。
  2. 用 noise_injector 按比例注入负例（随机替换头/尾），得到带标签 (triple, label)，label=1 为正例，0 为负例。
  3. 对每条 triple 按 --logic_mode 打分（见下）。
  4. 阈值分类：Accuracy / Precision / Recall / F1 / AUC。
  5. 错误检测 P@K、R@K（K=1%..5% 可调）：分数升序，前排为可疑；针对 label=0（假三元组）。

逻辑模式 --logic_mode：
  - fusion：路径 + LLM，若加载到 RGCN 则传入 s_logic_rgcn，由 LogicVerificationAgent 内融合（默认）。
  - llm_only：仅路径 + LLM，不使用 RGCN（等价于旧版 --no_gcn）。
  - rgcn_only：仅 RGCN 子图分数作为 S_logic，不调用 LLM（需 GCN checkpoint）。

依赖：
  - CC_gra/data/FB15k 下 fb15k237-*.txt 及 entity2text.txt、relation2text.txt
  - outputs/embeddings/fb15k237_transE.pt（必需）
  - outputs/gcn/fb15k237_gcn.pt（可选，无则仅用 LLM 分支）
  - 本地 Qwen（transformers）；若加载失败则用 Mock，指标无参考价值

默认（不加参数）：--split all --max_triples 5000 --noise_ratio 0.05
  即从 train+valid+test 全库中随机抽 5000 条正例，再注入约 5%% 噪声（负例条数=int(5000×0.05)=250），
  合计约 5250 条待打分；--max_triples 0 表示不抽样（用划分内全部正例，慎用）。

快速试跑：
  python examples/run_logic_agent_noisy_fb15k237.py --max_triples 50
  python examples/run_logic_agent_noisy_fb15k237.py --split test --noise_ratio 0.05 --max_triples 200
  python examples/run_logic_agent_noisy_fb15k237.py --threshold 0.5 --out outputs/logic_noisy_fb15k237_test.jsonl

三种逻辑打分对比（均需 CC_gra 根目录执行）：
  # 仅 LLM（路径+LLM）
  python examples/run_logic_agent_noisy_fb15k237.py --logic_mode llm_only --split test --max_triples 500
  # 仅 RGCN
  python examples/run_logic_agent_noisy_fb15k237.py --logic_mode rgcn_only --gcn_ckpt outputs/gcn/fb15k237_gcn_new_100k.pt --max_triples 500
  # LLM + RGCN 融合（默认 fusion，需能加载到 GCN）
  python examples/run_logic_agent_noisy_fb15k237.py --logic_mode fusion --gcn_ckpt outputs/gcn/fb15k237_gcn_new_100k.pt --max_triples 500

自动保存（默认开启）：结果写入 CC_gra/outputs/logic/<子目录>/，子目录名含 logic_mode、split、
  N/pos/neg、噪声百分比与 seed；内含 run_info.txt、metrics.json、per_triple.jsonl。加 --no_auto_save 可关闭。

在项目根目录 CC_gra 下执行：
  python examples/run_logic_agent_noisy_fb15k237.py
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from ccg_data.kg_dataset import KGDataset
from ccg_data.path_mining import build_graph
from ccg_data.text_mapping import load_fb15k237_text_maps
from ccg_data.noise_injector import inject_noise
from ccg_models.gcn_encoder import (
    build_adj_from_triples,
    extract_k_hop_subgraph_from_adj,
    AdjacencyBuilder,
    RelationalSubgraphEncoder,
    encode_triple_subgraph,
    SubgraphScorer,
)
from ccg_agents.llm_client import LocalQwenClient, MockLLMClient
from ccg_agents.logic_agent import LogicVerificationAgent, LogicAgentConfig
from ccg_pipeline.train_embeddings import load_config, load_model_config, load_transe_from_checkpoint

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    def roc_auc_score(y_true, y_score):
        if len(y_true) == 0 or sum(y_true) == 0 or sum(y_true) == len(y_true):
            return 0.0
        order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        S = sum((rank + 1) for rank, i in enumerate(order) if y_true[i] == 1)
        return (S - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

TripleId = Tuple[int, int, int]


def precision_recall_errors_at_percents(
    labels: Sequence[int],
    scores: Sequence[float],
    percents: Sequence[int] = (1, 2, 3, 4, 5),
) -> Dict[str, Any]:
    """
    错误检测口径：按分数升序（分低在前 = 更可疑），在前 K% 条中统计假三元组 (label=0)。
    P@K% = 前段中假的比例；R@K% = 命中假数 / 全体假数。
    """
    n = len(labels)
    n_err = sum(1 for y in labels if y == 0)
    out: Dict[str, Any] = {"n": n, "n_errors": n_err, "percents": list(percents)}
    if n == 0:
        for p in percents:
            out[f"precision_at_{p}"] = 0.0
            out[f"recall_at_{p}"] = 0.0
            out[f"k_eff_{p}"] = 0
        return out
    order = sorted(range(n), key=lambda i: float(scores[i]), reverse=False)
    for p in percents:
        p = int(p)
        k = min(n, max(1, math.ceil(n * p / 100.0)))
        top_idx = order[:k]
        hits = sum(1 for i in top_idx if labels[i] == 0)
        out[f"precision_at_{p}"] = hits / float(k) if k else 0.0
        out[f"recall_at_{p}"] = (hits / float(n_err)) if n_err > 0 else 0.0
        out[f"k_eff_{p}"] = k
        out[f"I_found_{p}"] = hits
    return out


def _create_llm_client():
    """本地 Qwen 优先，失败则 Mock。"""
    model_name = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    try:
        client = LocalQwenClient(
            model_name=model_name,
            device_map="auto",
        )
        return client, f"本地 {model_name}"
    except Exception:
        return MockLLMClient(), None


def compute_metrics(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
) -> dict:
    """
    根据真实标签与 S_logic 分数计算二分类指标。
    label=1 表示正确三元组，0 表示噪声；S_logic 越高越判为正（正确）。
    """
    preds = [1 if s >= threshold else 0 for s in scores]
    n = len(labels)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

    tp = sum(1 for i in range(n) if preds[i] == 1 and labels[i] == 1)
    tn = sum(1 for i in range(n) if preds[i] == 0 and labels[i] == 0)
    fp = sum(1 for i in range(n) if preds[i] == 1 and labels[i] == 0)
    fn = sum(1 for i in range(n) if preds[i] == 0 and labels[i] == 1)

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _load_gcn_from_path(
    gcn_ckpt: Path,
    device: torch.device,
    model_cfg: dict,
) -> Optional[Tuple[Any, Any, Any, int, AdjacencyBuilder]]:
    """从指定路径加载 GCN；失败或不存在则返回 None。"""
    if not gcn_ckpt.exists():
        return None
    try:
        state = torch.load(str(gcn_ckpt), map_location=device)
        ent_dim = int(state["ent_dim"])
        hidden_dim = int(state["hidden_dim"])
        num_relations = int(state["num_relations"])
        k_hop = int(state.get("k_hop", model_cfg.get("gcn", {}).get("num_layers", 2)))
        encoder = RelationalSubgraphEncoder(
            ent_dim=ent_dim,
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            dropout=float(model_cfg.get("gcn", {}).get("dropout", 0.3)),
        ).to(device)
        scorer = SubgraphScorer(triple_vec_dim=hidden_dim * 2 + ent_dim).to(device)
        encoder.load_state_dict(state["encoder_state"])
        scorer.load_state_dict(state["scorer_state"])
        encoder.eval()
        scorer.eval()
        builder = AdjacencyBuilder()
        return encoder, scorer, state, k_hop, builder
    except Exception:
        return None


def _load_gcn_if_available(
    base: Path,
    dataset_name: str,
    device: torch.device,
    model_cfg: dict,
) -> Optional[Tuple[Any, Any, Any, int, AdjacencyBuilder]]:
    gcn_ckpt = base / "gcn" / f"{dataset_name}_gcn.pt"
    return _load_gcn_from_path(gcn_ckpt, device, model_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="带噪声 FB15K-237：注入负例 → 逻辑智能体打分 → HIT/Precision/Recall/F1/AUC"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "test", "all"],
        help="从该划分中抽样正例再注噪；all=全库 train+valid+test 中抽取（默认配合 max_triples）",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.05,
        help="inject_noise 比例 [0,1]：负例条数=int(|正例|×noise_ratio)；默认 0.05（5%% 噪声）",
    )
    parser.add_argument(
        "--max_triples",
        type=int,
        default=5000,
        help="从划分中随机抽样的正例条数，再按 noise_ratio 注噪；0 表示不限制（使用划分内全部正例）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="S_logic >= threshold 判为正例（正确三元组），默认 0.5",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="保存 (triple, label, S_logic, ...) 的 JSONL 路径",
    )
    parser.add_argument(
        "--no_gcn",
        action="store_true",
        help="已弃用：请用 --logic_mode llm_only。若仍传入，则强制为仅 LLM。",
    )
    parser.add_argument(
        "--logic_mode",
        type=str,
        default="fusion",
        choices=["fusion", "llm_only", "rgcn_only"],
        help="fusion=LLM(+可选RGCN)；llm_only=仅路径+LLM；rgcn_only=仅RGCN分数，不调LLM",
    )
    parser.add_argument(
        "--gcn_ckpt",
        type=str,
        default=None,
        help="RGCN 权重路径（相对 CC_gra 根）；fusion/rgcn_only 时使用，默认 outputs/gcn/fb15k237_gcn.pt",
    )
    parser.add_argument(
        "--percents",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="错误检测 P@K/R@K 的 K（占样本数百分比）",
    )
    parser.add_argument(
        "--no_auto_save",
        action="store_true",
        help="禁用自动保存到 outputs/logic/<子目录>/（默认会保存 metrics.json、per_triple.jsonl、run_info.txt）",
    )
    args = parser.parse_args()
    max_triples_limit: Optional[int] = None if int(args.max_triples) == 0 else int(args.max_triples)

    logic_mode = str(args.logic_mode)
    if args.no_gcn:
        logic_mode = "llm_only"
        print("提示: --no_gcn 已指定，等价于 --logic_mode llm_only")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = "fb15k237"
    cfg = load_config()
    model_cfg = load_model_config()
    ds_cfg = cfg.get(dataset_name) or cfg.get("fb15k237")
    if not ds_cfg:
        print("未找到 fb15k237 配置。")
        return

    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")

    if not train_path.exists():
        print(f"训练集不存在: {train_path}")
        return

    ds = KGDataset.load_from_files(str(train_path), str(valid_path), str(test_path))
    train_triples = ds.train_triples
    graph = build_graph(train_triples)

    triples = ds.get_all_triples(args.split)
    if args.split == "all":
        print(f"划分 all: 池中共有 {len(triples)} 条正例 (train+valid+test)")
        if max_triples_limit is None and logic_mode in ("fusion", "llm_only"):
            print(
                "警告: 未限制抽样条数且将调用 LLM，耗时会极长；建议 --max_triples 5000 或 --logic_mode rgcn_only"
            )
    if max_triples_limit is not None and len(triples) > max_triples_limit:
        triples = random.sample(triples, max_triples_limit)
        print(f"抽样正例: {len(triples)} 条 (--max_triples={max_triples_limit})")
    elif max_triples_limit is not None:
        print(f"正例条数: {len(triples)}（未超过 --max_triples={max_triples_limit}，使用全部）")

    num_entities = ds.num_entities
    clean_triples, noisy_triples, labeled = inject_noise(
        triples, num_entities, args.noise_ratio, mode="head_tail",
    )
    n_pos = len(clean_triples)
    n_neg = len(noisy_triples)
    print(f"数据集: {dataset_name}, 划分: {args.split}")
    print(f"正例: {n_pos}, 负例: {n_neg}, 合计: {len(labeled)}, noise_ratio={args.noise_ratio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dir = base / "embeddings"
    transe_ckpt = embed_dir / f"{dataset_name}_transE.pt"
    if not transe_ckpt.exists():
        print(f"未找到 TransE 权重: {transe_ckpt}，请先训练 TransE。")
        return
    transe_model = load_transe_from_checkpoint(str(transe_ckpt), device=device)
    transe_model.eval()
    entity_emb = transe_model.ent_emb.weight.detach()
    rel_emb = transe_model.rel_emb

    default_gcn = base / "gcn" / f"{dataset_name}_gcn.pt"
    if args.gcn_ckpt:
        gp = Path(args.gcn_ckpt)
        gcn_ckpt_path = gp if gp.is_absolute() else (_ROOT / str(gp).lstrip("./"))
    else:
        gcn_ckpt_path = default_gcn

    gcn_ctx = None
    adj = None
    builder = None
    encoder = None
    scorer = None
    k_hop = None
    max_nodes = int(model_cfg.get("gcn", {}).get("max_subgraph_nodes", 2000))

    if logic_mode == "llm_only":
        print("逻辑模式 llm_only：不使用 RGCN，仅路径 + LLM。")
    elif logic_mode in ("fusion", "rgcn_only"):
        gcn_ctx = _load_gcn_from_path(gcn_ckpt_path, device, model_cfg)
        if gcn_ctx is None:
            if logic_mode == "rgcn_only":
                raise FileNotFoundError(
                    f"rgcn_only 需要 RGCN 权重，未找到或加载失败: {gcn_ckpt_path.resolve()}"
                )
            print(f"未找到 RGCN checkpoint ({gcn_ckpt_path})，fusion 退化为仅 LLM。")
        else:
            encoder, scorer, _state, k_hop, builder = gcn_ctx
            adj = build_adj_from_triples(train_triples)
            print(f"已加载 RGCN: k_hop={k_hop}, ckpt={gcn_ckpt_path.resolve()}")
            if logic_mode == "fusion":
                print("逻辑模式 fusion：LLM 与 S_logic_rgcn 在 LogicVerificationAgent 内融合。")
            else:
                print("逻辑模式 rgcn_only：仅使用 RGCN 分数，不调用 LLM。")

    data_root = _ROOT / "data" / "FB15k"
    try:
        ent2text, rel2text = load_fb15k237_text_maps(root=str(data_root))
        print(f"已加载实体映射 {len(ent2text)} 条、关系映射 {len(rel2text)} 条。")
    except FileNotFoundError as e:
        print(f"映射文件缺失: {e}，逻辑智能体将使用 ID 或路径形式。")
        ent2text, rel2text = {}, {}

    client, client_src = _create_llm_client()
    if logic_mode != "rgcn_only":
        if client_src:
            print(f"使用 {client_src} 进行逻辑推理。")
        else:
            print("本地 Qwen 加载失败，使用 MockLLMClient（LLM 分支分数恒为默认）。")

    agent: Optional[LogicVerificationAgent] = None
    if logic_mode != "rgcn_only":
        agent = LogicVerificationAgent(
            llm_client=client,
            graph=graph,
            id2entity=ds.id2entity,
            id2relation=ds.id2relation,
            ent2text=ent2text,
            rel2text=rel2text,
            config=LogicAgentConfig(max_path_len=3, max_paths=10, max_tokens=350),
            save_result_path=None,
        )

    scores: List[float] = []
    labels: List[int] = []
    records: List[dict] = []

    verbose = len(labeled) <= 15
    desc = f"逻辑打分 ({logic_mode})"
    it = tqdm(labeled, desc=desc, unit="triple")
    for (h, r, t, label) in it:
        s_logic_rgcn: Optional[float] = None
        if encoder is not None and adj is not None and builder is not None and scorer is not None:
            sub_nodes, sub_edges = extract_k_hop_subgraph_from_adj(
                adj, (h, r, t), k=k_hop, max_nodes=max_nodes
            )
            if len(sub_nodes) >= 2 and len(sub_edges) > 0:
                X, edge_index, edge_type, node2id = builder.build(sub_nodes, sub_edges, entity_emb.to(device))
                with torch.no_grad():
                    triple_vec = encode_triple_subgraph(
                        encoder, X, edge_index, edge_type, node2id, (h, r, t), rel_emb,
                    )
                    logit = scorer(triple_vec)
                    s_logic_rgcn = torch.sigmoid(logit).item()

        if logic_mode == "rgcn_only":
            s = float(s_logic_rgcn) if s_logic_rgcn is not None else 0.5
            result = {
                "S_logic": s,
                "S_logic_llm": None,
                "S_logic_rgcn": s_logic_rgcn,
                "path_count": None,
            }
        else:
            assert agent is not None
            # llm_only：不传 RGCN；fusion：传入 s_logic_rgcn（可能为 None）
            rgcn_in = None if logic_mode == "llm_only" else s_logic_rgcn
            result = agent.score_triple((h, r, t), s_logic_rgcn=rgcn_in)
            s = result["S_logic"]
        scores.append(s)
        labels.append(label)
        rec = {
            "triple": [h, r, t],
            "label": label,
            "logic_mode": logic_mode,
            "S_logic": s,
            "S_logic_llm": result.get("S_logic_llm"),
            "S_logic_rgcn": result.get("S_logic_rgcn"),
            "path_count": result.get("path_count"),
        }
        records.append(rec)
        if verbose:
            print(f"    triple ({h}, {r}, {t}) label={label} -> S_logic={s:.3f}  paths={result.get('path_count', 0)}")

    metrics = compute_metrics(labels, scores, threshold=args.threshold)
    prk = precision_recall_errors_at_percents(labels, scores, percents=args.percents)

    print("\n" + "=" * 60)
    print("逻辑验证智能体 — 模式={} — 阈值分类（阈值 = {:.2f}）".format(logic_mode, args.threshold))
    print("=" * 60)
    print("  HIT (Accuracy) = {:.4f}".format(metrics["accuracy"]))
    print("  Precision      = {:.4f}".format(metrics["precision"]))
    print("  Recall         = {:.4f}".format(metrics["recall"]))
    print("  F1             = {:.4f}".format(metrics["f1"]))
    print("  AUC            = {:.4f}".format(metrics["auc"]))
    print("  (TP={}, TN={}, FP={}, FN={})".format(
        metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]))

    print("\n" + "-" * 60)
    print("错误检测排名指标（分数升序，前排为可疑；P/R 针对 label=0 假三元组）")
    print(f"  N={prk['n']}  假条数={prk['n_errors']}")
    hdr = ["K%"] + [f"{p}%" for p in args.percents]
    pw = [f"{prk.get(f'precision_at_{p}', 0):.4f}" for p in args.percents]
    rw = [f"{prk.get(f'recall_at_{p}', 0):.4f}" for p in args.percents]
    kw = [str(prk.get(f"k_eff_{p}", 0)) for p in args.percents]
    iw = [str(prk.get(f"I_found_{p}", 0)) for p in args.percents]
    print("  " + " | ".join(f"{h:>8}" for h in hdr))
    print("  P@K " + " | ".join(f"{x:>8}" for x in pw))
    print("  R@K " + " | ".join(f"{x:>8}" for x in rw))
    print("  k(条数) " + " | ".join(f"{x:>8}" for x in kw))
    print("  I(错数) " + " | ".join(f"{x:>8}" for x in iw))
    print("-" * 60)

    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]
    if pos_scores:
        print("\n  S_logic 分布 — 正例: mean={:.4f}, median={:.4f}".format(
            sum(pos_scores) / len(pos_scores), sorted(pos_scores)[len(pos_scores) // 2]))
    if neg_scores:
        print("  S_logic 分布 — 负例: mean={:.4f}, median={:.4f}".format(
            sum(neg_scores) / len(neg_scores), sorted(neg_scores)[len(neg_scores) // 2]))

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _ROOT / str(out_path).lstrip("./")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n结果已写入: {out_path.resolve()}")

    if not args.no_auto_save:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nr_pct = int(round(float(args.noise_ratio) * 100))
        n_total = len(labeled)
        folder_name = (
            f"{logic_mode}_split-{args.split}_N{n_total}_"
            f"pos{n_pos}_neg{n_neg}_nr{nr_pct}pct_seed{args.seed}_{ts}"
        )
        save_dir = base / "logic" / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        run_info_lines = [
            "逻辑智能体单测 — 运行摘要",
            "============================",
            f"逻辑模式 (logic_mode): {logic_mode}",
            f"数据划分 (split):       {args.split}  （正例池来源）",
            f"抽样上限 (max_triples): {max_triples_limit if max_triples_limit is not None else '不限制(0)'}",
            f"噪声比例 (noise_ratio): {args.noise_ratio}  （约 {nr_pct}% 相对正例条数生成负例）",
            f"随机种子 (seed):        {args.seed}",
            "",
            f"正例条数: {n_pos}",
            f"负例条数: {n_neg}",
            f"合计评估条数 N: {n_total}",
            f"分类阈值 (threshold):   {args.threshold}",
            "",
        ]
        if logic_mode != "llm_only":
            run_info_lines.append(f"RGCN 权重路径: {gcn_ckpt_path.resolve()}")
        else:
            run_info_lines.append("RGCN: 未使用 (llm_only)")
        run_info_lines.extend(
            [
                "",
                f"输出文件: metrics.json, per_triple.jsonl, run_info.txt",
                f"生成时间: {ts}",
            ]
        )
        with open(save_dir / "run_info.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(run_info_lines) + "\n")

        payload: Dict[str, Any] = {
            "dataset": dataset_name,
            "split": args.split,
            "logic_mode": logic_mode,
            "noise_ratio": float(args.noise_ratio),
            "seed": int(args.seed),
            "max_triples": max_triples_limit,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_total": n_total,
            "threshold": float(args.threshold),
            "percents": [int(x) for x in args.percents],
            "metrics_threshold": metrics,
            "metrics_error_rank_prk": prk,
        }
        if logic_mode != "llm_only":
            payload["gcn_ckpt"] = str(gcn_ckpt_path.resolve())
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        per_triple_path = save_dir / "per_triple.jsonl"
        with open(per_triple_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\n已自动保存至: {save_dir.resolve()}")
        print(f"  - run_info.txt（模式与数据量说明）")
        print(f"  - metrics.json（指标汇总）")
        print(f"  - per_triple.jsonl（逐条分数）")


if __name__ == "__main__":
    main()
