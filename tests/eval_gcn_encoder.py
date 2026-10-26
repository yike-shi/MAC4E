"""
GCN/RGCN 子图编码器评估：在指定划分上做「真 vs 噪」二分类，汇报 P@K、R@K（K 为排名比例 1%–5%）。

- 加载训练好的 encoder + scorer 与 TransE（提供实体/关系嵌入）；
- 正例来源：`--split valid` / `test` / `all`（all = train+valid+test 全部三元组）；
- inject_noise 按 noise_ratio 从正例中抽样生成负例（默认 5%，即负例条数≈ceil(|正例|×0.05)）；子图仍建在 **训练集** 邻接上；
- 对每条样本提取 k-hop 子图并打分；label=1 真、label=0 注入假；
- --eval_target real：分数降序，P@K/R@K 针对「真三元组」（合理性 / 链式补全口径）；
- --eval_target error：分数升序（分低在前 = 更可疑），P@K/R@K 针对「假三元组」（错误检测口径，与端到端找错一致）；
- 另可选输出 AUC（--show_auc，以 label=1 为正类）。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.noise_injector import inject_noise
from ccg_models.gcn_encoder import (
    build_adj_from_triples,
    extract_k_hop_subgraph_from_adj,
    AdjacencyBuilder,
    RelationalSubgraphEncoder,
    encode_triple_subgraph,
    SubgraphScorer,
)
from ccg_pipeline.train_embeddings import load_config, load_model_config, load_transe_from_checkpoint

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

TripleId = Tuple[int, int, int]

# 默认报告的排名比例（占有效样本数的百分比）
DEFAULT_PK_PERCENTS: Tuple[int, ...] = (1, 2, 3, 4, 5)


def precision_recall_at_rank_percent(
    preds: Sequence[float],
    labels: Sequence[int],
    k_percents: Sequence[int] = DEFAULT_PK_PERCENTS,
    *,
    eval_target: Literal["real", "error"] = "real",
) -> List[Tuple[int, int, float, float]]:
    """
    对每个 K% 计算 P@K% 与 R@K%。

    eval_target == "real"（分数降序，高分在前）：
        - 考察「真三元组」是否排在前面；hits = label==1。
        - R@K% 分母为全体真条数。

    eval_target == "error"（分数升序，低分在前，错误检测）：
        - 考察「假/噪声三元组」是否排在前面（模型给低分）；hits = label==0。
        - R@K% 分母为全体假条数。
    """
    n = len(preds)
    if n == 0:
        return []
    total_pos = int(sum(labels))
    total_neg = n - total_pos
    if eval_target == "real":
        order = sorted(range(n), key=lambda i: preds[i], reverse=True)
        want_one = True
        denom = total_pos
    elif eval_target == "error":
        order = sorted(range(n), key=lambda i: preds[i], reverse=False)
        want_one = False
        denom = total_neg
    else:
        raise ValueError(f"eval_target 应为 real 或 error，当前: {eval_target}")

    out: List[Tuple[int, int, float, float]] = []
    for k_pct in k_percents:
        n_top = min(n, max(1, math.ceil(n * k_pct / 100.0)))
        top_idx = order[:n_top]
        if want_one:
            hits = sum(1 for i in top_idx if labels[i] == 1)
        else:
            hits = sum(1 for i in top_idx if labels[i] == 0)
        p_k = hits / n_top if n_top else 0.0
        r_k = hits / denom if denom > 0 else float("nan")
        out.append((k_pct, n_top, p_k, r_k))
    return out


def load_gcn_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[RelationalSubgraphEncoder, SubgraphScorer, int, int]:
    """加载 GCN checkpoint，返回 encoder、scorer 以及 k_hop、max_subgraph_nodes（从 config 读）。"""
    state = torch.load(ckpt_path, map_location=device)
    ent_dim = state["ent_dim"]
    hidden_dim = state["hidden_dim"]
    num_relations = state["num_relations"]
    k_hop = state["k_hop"]

    encoder = RelationalSubgraphEncoder(
        ent_dim=ent_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
    ).to(device)
    triple_vec_dim = hidden_dim * 2 + ent_dim
    scorer = SubgraphScorer(triple_vec_dim=triple_vec_dim).to(device)

    encoder.load_state_dict(state["encoder_state"])
    scorer.load_state_dict(state["scorer_state"])
    encoder.eval()
    scorer.eval()
    return encoder, scorer, k_hop, state.get("max_subgraph_nodes", 2000)


def evaluate_gcn(
    dataset_name: str,
    split: str = "all",
    max_samples: int | None = 5000,
    gcn_ckpt: str | None = None,
    pk_percents: Sequence[int] | None = None,
    show_auc: bool = False,
    noise_ratio: float = 0.05,
    eval_target: Literal["real", "error"] = "error",
) -> None:
    cfg = load_config()
    model_cfg = load_model_config()
    gcn_cfg = model_cfg.get("gcn", {})
    max_subgraph_nodes = int(gcn_cfg.get("max_subgraph_nodes", 2000))

    ds_cfg = cfg.get(dataset_name.lower())
    if not ds_cfg:
        raise KeyError(f"未知数据集: {dataset_name}")
    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")

    ds = KGDataset.load_from_files(str(train_path), str(valid_path), str(test_path))
    train_triples = ds.train_triples
    num_entities = ds.num_entities

    if split == "valid":
        pos_triples = list(ds.valid_triples)
    elif split == "test":
        pos_triples = list(ds.test_triples)
    elif split == "all":
        pos_triples = list(ds.train_triples) + list(ds.valid_triples) + list(ds.test_triples)
    else:
        raise ValueError(f"未知 split: {split}，应为 valid / test / all")

    if not pos_triples:
        print(f"  {split} 集为空，无法评估")
        return
    print(f"  正例条数 ({split}): {len(pos_triples)}")
    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError(f"noise_ratio 须在 [0,1] 内，当前为 {noise_ratio}")
    print(f"  噪声注入比例 noise_ratio: {noise_ratio}（负例条数 ≈ int(|正例|×noise_ratio)）")

    # 负例：inject_noise 中 noise_num = int(len(正例) * noise_ratio)
    _, neg_triples, _ = inject_noise(
        pos_triples, num_entities=num_entities, noise_ratio=noise_ratio, mode="head_tail"
    )
    print(f"  负例条数: {len(neg_triples)}")
    eval_triples: List[TripleId] = []
    eval_labels: List[int] = []
    eval_triples.extend(pos_triples)
    eval_labels.extend([1] * len(pos_triples))
    eval_triples.extend(neg_triples)
    eval_labels.extend([0] * len(neg_triples))

    if max_samples is not None and len(eval_triples) > max_samples:
        import random
        print(
            f"  子采样: 从 {len(eval_triples)} 条候选中随机取 {max_samples} 条 "
            f"（全量请设 --max_samples 0）"
        )
        idx = random.sample(range(len(eval_triples)), max_samples)
        eval_triples = [eval_triples[i] for i in idx]
        eval_labels = [eval_labels[i] for i in idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transe_path = base / "embeddings" / f"{dataset_name.lower()}_transE.pt"
    if gcn_ckpt:
        gcn_path = Path(gcn_ckpt).expanduser()
        if not gcn_path.is_absolute():
            gcn_path = (_ROOT / gcn_path).resolve()
    else:
        gcn_path = base / "gcn" / f"{dataset_name.lower()}_gcn.pt"
    if not gcn_path.exists():
        raise FileNotFoundError(f"GCN checkpoint 不存在: {gcn_path}")
    if not transe_path.exists():
        raise FileNotFoundError(f"TransE checkpoint 不存在: {transe_path}")

    transe = load_transe_from_checkpoint(str(transe_path), device=device)
    transe.eval()
    entity_emb = transe.ent_emb.weight.detach()
    rel_emb = transe.rel_emb

    encoder, scorer, k_hop, _ = load_gcn_from_checkpoint(str(gcn_path), device)
    print(f"  使用权重: {gcn_path}")
    max_nodes = max_subgraph_nodes

    adj = build_adj_from_triples(train_triples)
    builder = AdjacencyBuilder()

    preds: List[float] = []
    labels_out: List[int] = []
    n_skip = 0

    for (triple, y) in tqdm(zip(eval_triples, eval_labels), total=len(eval_triples), desc="Eval GCN", unit="triple"):
        sub_nodes, sub_edges = extract_k_hop_subgraph_from_adj(adj, triple, k=k_hop, max_nodes=max_nodes)
        if len(sub_nodes) < 2 or len(sub_edges) == 0:
            n_skip += 1
            continue
        X, edge_index, edge_type, node2id = builder.build(sub_nodes, sub_edges, entity_emb.to(device))
        with torch.no_grad():
            triple_vec = encode_triple_subgraph(encoder, X, edge_index, edge_type, node2id, triple, rel_emb)
            logit = scorer(triple_vec)
            score = torch.sigmoid(logit).item()
        preds.append(score)
        labels_out.append(y)

    if n_skip:
        print(f"  跳过无效子图: {n_skip} 条")
    n = len(preds)
    if n == 0:
        print("  无有效预测，无法计算指标")
        return

    k_list = tuple(pk_percents) if pk_percents is not None else DEFAULT_PK_PERCENTS
    pr_rows = precision_recall_at_rank_percent(
        preds, labels_out, k_percents=k_list, eval_target=eval_target
    )
    n_pos = sum(labels_out)
    n_neg = n - n_pos

    print("=" * 60)
    print(
        f"GCN/RGCN 子图编码器评估: dataset={dataset_name}, split={split}, "
        f"noise_ratio={noise_ratio}, eval_target={eval_target}, device={device}"
    )
    print(f"  有效样本数 N={n} (真={n_pos}, 假/噪={n_neg}, 跳过无效子图={n_skip})")
    if eval_target == "real":
        print("  排名规则: 分数降序 → 前排考察「真三元组」占比（合理性）")
        print("  P@K%/R@K%: 针对真三元组；R 的分母=全体真条数")
    else:
        print("  排名规则: 分数升序 → 前排考察「假三元组」占比（错误检测，与端到端找错一致）")
        print("  P@K%/R@K%: 针对假/噪声三元组；R 的分母=全体假条数")
    print("-" * 60)
    print(f"  {'K(占N%)':>8}  {'top条数':>8}  {'P@K%':>10}  {'R@K%':>10}")
    for k_pct, n_top, p_k, r_k in pr_rows:
        r_str = f"{r_k:.4f}" if r_k == r_k else "   n/a"
        print(f"  {k_pct:>7}%  {n_top:>8}  {p_k:>10.4f}  {r_str:>10}")
    print("-" * 60)
    if show_auc:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_out, preds)
            print(f"  AUC (附加): {auc:.4f}" if auc == auc else "  AUC: (sklearn 不可用)")
        except Exception:
            print("  AUC (附加): (计算失败)")
    print("=" * 60)


def _parse_int_list(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description="GCN 子图编码器二分类评估 (P@K% / R@K%，K 为排名比例)")
    parser.add_argument("--dataset", type=str, default="fb15k237", help="数据集名称")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["valid", "test", "all"],
        help="正例来源：valid / test；all=train+valid+test 全部分（数据量大时请用 --max_samples 控制或 0 全量）",
    )
    parser.add_argument("--max_samples", type=int, default=5000, help="最多评估条数，避免过慢；0 表示不限制")
    parser.add_argument(
        "--gcn_ckpt",
        type=str,
        default=None,
        help="GCN/RGCN 权重路径；默认 outputs/gcn/{dataset}_gcn.pt。可传如 outputs/gcn/fb15k237_gcn_new_100k.pt",
    )
    parser.add_argument(
        "--pk_percents",
        type=str,
        default="1,2,3,4,5",
        help="P@K/R@K 的 K（占有效样本数的百分比），逗号分隔；默认 1,2,3,4,5",
    )
    parser.add_argument("--show_auc", action="store_true", help="额外打印 AUC")
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.05,
        help="inject_noise 注入比例 [0,1]：负例条数=int(|正例|×noise_ratio)；默认 0.05（5%%）",
    )
    parser.add_argument(
        "--eval_target",
        type=str,
        default="error",
        choices=["error", "real"],
        help=(
            "error=分数升序，P@K/R@K 针对假三元组（错误检测，默认）；"
            "real=分数降序，P@K/R@K 针对真三元组（链式合理性）。"
        ),
    )
    args = parser.parse_args()
    max_samples = args.max_samples if args.max_samples > 0 else None
    pk_percents = _parse_int_list(args.pk_percents)
    evaluate_gcn(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=max_samples,
        gcn_ckpt=args.gcn_ckpt,
        pk_percents=pk_percents,
        show_auc=args.show_auc,
        noise_ratio=args.noise_ratio,
        eval_target=args.eval_target,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
