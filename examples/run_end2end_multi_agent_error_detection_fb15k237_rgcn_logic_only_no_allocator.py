"""
端到端多智能体错误检测（FB15K-237，逻辑仅 RGCN）— **无 Allocator 消融**：

与 ``run_end2end_multi_agent_error_detection_fb15k237_rgcn_logic_only.py`` 对齐，但 **不加载、不调用 Allocator**，
任务分配权重 **固定为 1:1:1（已归一化）** 传入融合智能体（含 ``--fusion_llm_reliability`` 时，prompt 中任务分配参考亦为 1:1:1；
若使用 ``--fusion_llm_prompt_allocator_mlp``，prompt 中「MLP 参考」亦用同一均匀权重，因无 MLP）。

其余：结构/事实/逻辑三路、zscore、GCK、partial jsonl 等与原版一致。

三路分数名义上多在 [0,1]，但各分支尺度仍可能不一致。
默认 --agent_score_norm zscore：对本 run 全部样本的 S_struct、S_fact、S_logic **分别**做列内 z-score 后经 Φ(z) 映到 (0,1)（常数列退化为 0.5），再与 Allocator 权重加权融合；也可用 minmax=[0,1] 或 none。

逻辑 RGCN 权重默认 outputs/gcn/fb15k237_gcn.pt；可用 --gcn_ckpt 指定任意训练得到的 .pt（如 fb15k237_gcn_new_100k.pt）。

适用场景：
- 希望明显降低时间复杂度（相比“事实+逻辑都调 LLM”）
- 仍保留事实分支的 LLM 能力，同时让逻辑分支走图模型快速打分
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.noise_injector import inject_noise
from ccg_data.path_mining import build_graph
from ccg_data.text_mapping import load_fb15k237_text_maps
from ccg_utils.agent_score_norm import normalize_score_channel
from ccg_utils.kg_utils import build_entity_frequency
from ccg_pipeline.train_embeddings import (
    load_complex_from_checkpoint,
    load_config,
    load_model_config,
    load_transe_from_checkpoint,
)
from ccg_agents.fact_agent import FactAgentConfig, FactVerificationAgent
from ccg_agents.fusion_agent import FusionAgentConfig, FusionDecisionAgent
from ccg_agents.fusion_llm_agent import FusionLLMWeightAgent, FusionLLMWeightAgentConfig
from ccg_agents.logic_agent import LogicAgentConfig, LogicVerificationAgent
from ccg_agents.llm_client import LocalQwenClient, MockLLMClient
from ccg_agents.structure_agent import StructureAgentConfig, StructureConstraintAgent
from ccg_models.gcn_encoder import (
    AdjacencyBuilder,
    RelationalSubgraphEncoder,
    SubgraphScorer,
    build_adj_from_triples,
    encode_triple_subgraph,
    extract_k_hop_subgraph_from_adj,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it


TripleId = Tuple[int, int, int]


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("end2end_rgcn_logic_only")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _create_llm_client(prefer_local_qwen: bool = True):
    if not prefer_local_qwen:
        return MockLLMClient(), None
    model_name = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_map = "cuda:0" if cuda_visible is not None else "auto"
    try:
        client = LocalQwenClient(model_name=model_name, device_map=device_map)
        return client, f"本地 {model_name}"
    except Exception:
        return MockLLMClient(), None


def _load_gcn_if_available(
    gcn_ckpt_path: Path,
    device: torch.device,
    model_cfg: dict,
) -> Optional[Tuple[Any, Any, Any, int, AdjacencyBuilder]]:
    """从给定路径加载 RGCN encoder+scorer；文件不存在或加载失败则返回 None。"""
    if not gcn_ckpt_path.exists():
        return None
    try:
        state = torch.load(gcn_ckpt_path, map_location=device)
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


def _compute_s_logic_rgcn(
    triple: TripleId,
    *,
    gcn_ctx: Optional[Tuple[Any, Any, Any, int, AdjacencyBuilder]],
    adj: Any,
    entity_emb: torch.Tensor,
    rel_emb: torch.nn.Parameter,
    device: torch.device,
    model_cfg: dict,
) -> Optional[float]:
    if gcn_ctx is None or adj is None:
        return None
    encoder, scorer, _state, k_hop, builder = gcn_ctx
    h, r, t = triple
    max_nodes = int(model_cfg.get("gcn", {}).get("max_subgraph_nodes", 2000))
    sub_nodes, sub_edges = extract_k_hop_subgraph_from_adj(adj, (h, r, t), k=k_hop, max_nodes=max_nodes)
    if len(sub_nodes) < 2 or len(sub_edges) == 0:
        return None
    X, edge_index, edge_type, node2id = builder.build(sub_nodes, sub_edges, entity_emb.to(device))
    with torch.no_grad():
        triple_vec = encode_triple_subgraph(
            encoder, X, edge_index, edge_type, node2id, (h, r, t), rel_emb
        )
        logit = scorer(triple_vec)
        return float(torch.sigmoid(logit).item())


def _uniform_task_weights() -> List[float]:
    """消融：无 Allocator，三路任务分配固定 1:1:1。"""
    return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


def precision_recall_at_percents_for_errors(
    labels: Sequence[int],
    scores: Sequence[float],
    percents: Sequence[int] = (1, 2, 3, 4, 5),
) -> Dict[str, Any]:
    n = len(labels)
    f_err = sum(1 for y in labels if y == 0)
    out: Dict[str, Any] = {"n_total": n, "n_errors": f_err, "percents": list(percents)}
    if n == 0:
        for p in percents:
            out[f"precision_at_{p}"] = 0.0
            out[f"recall_at_{p}"] = 0.0
            out[f"k_eff_{p}"] = 0
            out[f"I_found_{p}"] = 0
        return out

    order = sorted(range(n), key=lambda i: float(scores[i]))
    for p in percents:
        p = int(p)
        k = int((n * p + 99) // 100)
        k = max(1, min(k, n))
        idx = order[:k]
        i_found = sum(1 for i in idx if labels[i] == 0)
        out[f"precision_at_{p}"] = i_found / float(k)
        out[f"recall_at_{p}"] = (i_found / float(f_err)) if f_err > 0 else 0.0
        out[f"k_eff_{p}"] = k
        out[f"I_found_{p}"] = i_found
    return out


def _format_prk_table(metrics: Dict[str, Any], percents: Sequence[int]) -> str:
    header = ["Metric"] + [f"{p}%" for p in percents]
    rows = [
        ["P@K"] + [f"{metrics.get(f'precision_at_{p}', 0.0):.4f}" for p in percents],
        ["R@K"] + [f"{metrics.get(f'recall_at_{p}', 0.0):.4f}" for p in percents],
        ["k(条数)"] + [str(metrics.get(f'k_eff_{p}', 0)) for p in percents],
        ["I(错数)"] + [str(metrics.get(f'I_found_{p}', 0)) for p in percents],
    ]
    col_w = [max(len(str(x)) for x in col) for col in zip(header, *rows)]

    def fmt_row(items):
        return "  " + " | ".join(str(it).ljust(w) for it, w in zip(items, col_w))

    lines = [fmt_row(header), "  " + "-+-".join("-" * w for w in col_w)]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FB15K-237 端到端错误检测（逻辑仅RGCN；无 Allocator，任务权重固定 1:1:1）"
    )
    parser.add_argument("--dataset", type=str, default="fb15k237", choices=["fb15k237"])
    parser.add_argument("--noise_ratio", type=float, default=0.05, choices=[0.05, 0.10, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_triples", type=int, default=None)
    parser.add_argument("--out_metrics", type=str, default=None)
    parser.add_argument("--out_jsonl", type=str, default=None, help="可选：保存逐条分数（不含逻辑LLM解释）")
    parser.add_argument(
        "--out_jsonl_partial",
        type=str,
        default=None,
        help=(
            "可选：保存逐条分数的增量 jsonl（每条追加一行并定期 flush）。"
            "即使中途进程被 Terminated，也能保留已完成部分的 final_score / S_* / reliability_weights 等明细。"
            "不会做 bottom% 过滤（仅用于兜底）。"
        ),
    )
    parser.add_argument(
        "--out_jsonl_partial_flush_every",
        type=int,
        default=200,
        help=(
            "--out_jsonl_partial 下每处理 N 条 flush 一次；0 表示每条都 flush（更稳但更慢）。"
        ),
    )
    parser.add_argument(
        "--explain_bottom_percent",
        type=float,
        default=100.0,
        help="仅在 --out_jsonl 时生效：按 final_score 升序仅导出最低分底部 x%%（默认 100%% 即全导出）",
    )
    parser.add_argument("--percents", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--no_gcn", action="store_true", help="禁用逻辑 RGCN（此时逻辑分数固定 0.5）")
    parser.add_argument(
        "--gcn_ckpt",
        type=str,
        default=None,
        help=(
            "逻辑 RGCN 权重路径；默认 outputs/gcn/fb15k237_gcn.pt。"
            "可传相对 CC_gra 根目录的路径，如 outputs/gcn/fb15k237_gcn_new_100k.pt。"
            "若显式指定但文件不存在则报错退出。"
        ),
    )
    parser.add_argument("--no_llm", action="store_true", help="同时禁用事实 LLM（事实分固定 0.5）")
    parser.add_argument(
        "--logic_llm",
        action="store_true",
        help=(
            "逻辑分支启用 LogicVerificationAgent：多跳路径+LLM 得 S_logic_llm，并与 RGCN 分数在 Agent 内 beta 融合"
            "（与 run_end2end_multi_agent_error_detection_fb15k237.py 一致）。默认关闭=仅 RGCN、更快。"
            "与 --no_llm 互斥。"
        ),
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument(
        "--agent_score_norm",
        type=str,
        default="zscore",
        choices=["none", "minmax", "zscore"],
        help=(
            "融合前对本 run 全部样本的 S_struct、S_fact、S_logic 分别做列归一化，再与 Allocator 权重加权。"
            "默认 zscore（Φ(z)）；none=不归一化；minmax=[0,1]。"
        ),
    )
    parser.add_argument(
        "--fusion_llm_reliability",
        action="store_true",
        help=(
            "使用 fusion_llm_agent.FusionLLMWeightAgent：由 LLM 根据三路分数与说明输出可靠性权重 w_struct/w_fact/w_logic，"
            "再算 final_score（prompt 中可含任务分配智能体 Allocator 的参考权重）。"
            "不加本参数时仍为 FusionDecisionAgent 纯加权（use_llm=False）。"
        ),
    )
    parser.add_argument(
        "--fusion_llm_max_tokens",
        type=int,
        default=512,
        help="--fusion_llm_reliability 时融合 LLM 的 max_tokens。",
    )
    parser.add_argument(
        "--fusion_llm_skip_llm",
        action="store_true",
        help="--fusion_llm_reliability 下仍不调融合 LLM，仅用 Allocator 基线加权（用于消融）。",
    )
    parser.add_argument(
        "--fusion_llm_no_allocator_hint",
        action="store_true",
        help="--fusion_llm_reliability 下不在 prompt 中写入 Allocator 参考权重节。",
    )
    parser.add_argument(
        "--fusion_llm_prompt_allocator_mlp",
        action="store_true",
        help=(
            "仅当 --fusion_llm_reliability：prompt 中「第二套任务分配参考」仍写 1:1:1（消融无 MLP，"
            "reference_allocator_weights 与主参考相同，便于与原版 --fusion_llm_prompt_allocator_mlp 脚本对照）。"
            "默认不加：prompt 仅展示与 decide 一致的 1:1:1。"
        ),
    )
    args = parser.parse_args()
    if args.logic_llm and args.no_llm:
        parser.error("--logic_llm 需要调用逻辑 LLM，不能与 --no_llm 同时使用")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    model_cfg = load_model_config()
    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        Path(args.log_file)
        if args.log_file
        else base / "logs" / f"end2end_rgcn_logic_only_no_allocator_{ts}.log"
    )
    log = _setup_logger(log_path)

    log.info("=" * 72)
    log.info("端到端运行（逻辑仅RGCN；消融：无 Allocator，任务权重 1:1:1）启动")
    log.info("log_file=%s", log_path.resolve())
    log.info("CLI=%s", json.dumps(vars(args), ensure_ascii=False, default=str))
    if args.agent_score_norm != "none":
        log.info("agent_score_norm=%s（三路同一规则，融合用）", args.agent_score_norm)
    if args.fusion_llm_reliability:
        log.info(
            "融合: FusionLLMWeightAgent（LLM 输出 w_*×S_*；任务分配参考 m_* 固定 1:1:1）"
            " max_tokens=%d skip_llm=%s allocator_hint=%s prompt_use_allocator_mlp=%s",
            args.fusion_llm_max_tokens,
            args.fusion_llm_skip_llm,
            not args.fusion_llm_no_allocator_hint,
            args.fusion_llm_prompt_allocator_mlp,
        )
    log.info("任务分配（消融）: 固定 1:1:1，不使用 Allocator")

    ds_cfg = cfg.get("fb15k237")
    if not isinstance(ds_cfg, dict):
        raise KeyError("未找到 fb15k237 配置")
    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")
    ds = KGDataset.load_from_files(str(train_path), str(valid_path), str(test_path))

    clean_all: List[TripleId] = list(ds.train_triples) + list(ds.valid_triples) + list(ds.test_triples)
    if args.max_triples is not None and args.max_triples > 0 and len(clean_all) > args.max_triples:
        clean_all = random.sample(clean_all, args.max_triples)
    clean_triples, noisy_triples, labeled = inject_noise(
        clean_all,
        num_entities=ds.num_entities,
        noise_ratio=float(args.noise_ratio),
        mode="head_tail",
    )
    noisy_all: List[TripleId] = list(clean_triples) + list(noisy_triples)
    graph = build_graph(noisy_all)
    log.info("N=%d clean=%d noisy=%d", len(labeled), len(clean_triples), len(noisy_triples))

    transe_ckpt = base / "embeddings" / "fb15k237_transE.pt"
    if not transe_ckpt.exists():
        raise FileNotFoundError(f"未找到 TransE checkpoint: {transe_ckpt}")
    transe = load_transe_from_checkpoint(str(transe_ckpt), device=device)
    transe.eval()
    entity_emb = transe.ent_emb.weight.detach()
    rel_emb = transe.rel_emb

    complex_model = None
    complex_ckpt = base / "embeddings" / "fb15k237_complEx.pt"
    if complex_ckpt.exists():
        complex_model = load_complex_from_checkpoint(str(complex_ckpt), device=device)
        complex_model.eval()

    default_gcn = base / "gcn" / "fb15k237_gcn.pt"
    if args.gcn_ckpt:
        gp = Path(args.gcn_ckpt)
        gcn_ckpt_path = gp if gp.is_absolute() else (_ROOT / str(gp).lstrip("./"))
        if not gcn_ckpt_path.exists():
            raise FileNotFoundError(f"未找到 RGCN checkpoint（--gcn_ckpt）: {gcn_ckpt_path.resolve()}")
    else:
        gcn_ckpt_path = default_gcn

    gcn_ctx = None
    adj_for_gcn = None
    if not args.no_gcn:
        gcn_ctx = _load_gcn_if_available(gcn_ckpt_path, device, model_cfg)
    if gcn_ctx is not None:
        adj_for_gcn = build_adj_from_triples(ds.train_triples)
        _enc, _sc, _st, k_hop, _bd = gcn_ctx
        log.info("逻辑RGCN已启用: k_hop=%d, ckpt=%s", k_hop, gcn_ckpt_path.resolve())
    else:
        log.info(
            "逻辑RGCN未启用（--no_gcn 或 checkpoint 缺失/加载失败），逻辑分将使用0.5默认值。尝试路径: %s",
            gcn_ckpt_path.resolve(),
        )

    from ccg_data.type_constraints import TypeConstraints
    type_constraints = TypeConstraints()
    type_constraints.build_from_triples(clean_triples)
    entity_freq = build_entity_frequency(clean_triples)

    emb_mode = "transe"
    fusion_alpha = 0.5

    llm_client, llm_src = _create_llm_client(prefer_local_qwen=not args.no_llm)
    # --no_llm 时事实分支用 Mock；若仍启用 --fusion_llm_reliability，融合 LLM 需单独加载真实客户端，
    # 否则 FusionLLMWeightAgent 也会走 Mock，消融「关事实 LLM」失效。
    if args.no_llm and args.fusion_llm_reliability:
        fusion_llm_client, fusion_llm_src = _create_llm_client(prefer_local_qwen=True)
    else:
        fusion_llm_client, fusion_llm_src = llm_client, llm_src
    if args.no_llm:
        log.info("事实LLM禁用：--no_llm")
    else:
        log.info("事实LLM: %s", llm_src or "Mock (fallback)")
    if args.fusion_llm_reliability:
        log.info("融合LLM: %s", fusion_llm_src or "Mock (fallback)")

    struct_agent = StructureConstraintAgent(
        transe_model=transe,
        graph=graph,
        complex_model=complex_model,
        config=StructureAgentConfig(emb_mode="transe"),
        llm_explainer=None,
        id2entity=ds.id2entity,
        id2relation=ds.id2relation,
    )
    data_root = _ROOT / "data" / "FB15k"
    try:
        ent2text, rel2text = load_fb15k237_text_maps(root=str(data_root))
    except FileNotFoundError:
        ent2text, rel2text = {}, {}
    fact_agent = FactVerificationAgent(
        llm_client=llm_client,
        id2entity=ds.id2entity,
        id2relation=ds.id2relation,
        ent2text=ent2text,
        rel2text=rel2text,
        config=FactAgentConfig(default_score_no_readable=0.5, skip_fact_when_no_readable=True, max_tokens=300),
        save_result_path=None,
    )
    if args.fusion_llm_reliability:
        fusion_agent = FusionLLMWeightAgent(
            fusion_llm_client,
            config=FusionLLMWeightAgentConfig(
                use_llm=not args.fusion_llm_skip_llm,
                max_tokens=int(args.fusion_llm_max_tokens),
                include_allocator_hint=not args.fusion_llm_no_allocator_hint,
            ),
            save_result_path=None,
        )
    else:
        fusion_agent = FusionDecisionAgent(
            llm_client=llm_client,
            config=FusionAgentConfig(use_llm=False, max_tokens=350),
            save_result_path=None,
        )

    triples_all: List[TripleId] = [(h, r, t) for (h, r, t, _y) in labeled]
    labels_all: List[int] = [int(_y) for (_h, _r, _t, _y) in labeled]
    final_scores: List[float] = []
    records: List[Dict[str, Any]] = []
    # (task_weights_m_s_m_f_m_l, s_res, f_res, l_res)，融合前可选列归一化；m 固定 1:1:1
    packed: List[Tuple[List[float], Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []

    partial_fh = None
    if args.out_jsonl_partial:
        partial_path = Path(args.out_jsonl_partial)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        # 使用覆盖写入，避免不同 run 混在同一个 partial 文件里。
        partial_fh = open(partial_path, "w", encoding="utf-8")
        log.info("逐条打分增量已打开: %s", partial_path.resolve())

    t0 = time.perf_counter()
    for idx, triple in enumerate(tqdm(triples_all, desc="Scoring (fact LLM + logic RGCN only)", unit="triple"), start=1):
        w = _uniform_task_weights()
        s_res = struct_agent.score_triple(triple)

        if args.no_llm:
            f_res = {"S_fact": 0.5, "explanation": None, "skipped": True, "readable_triple": None}
        else:
            f_res = fact_agent.score_triple(triple)

        s_rgcn = _compute_s_logic_rgcn(
            triple,
            gcn_ctx=gcn_ctx,
            adj=adj_for_gcn,
            entity_emb=entity_emb,
            rel_emb=rel_emb,
            device=device,
            model_cfg=model_cfg,
        )
        s_logic = float(s_rgcn) if s_rgcn is not None else 0.5
        l_res = {
            "S_logic": s_logic,
            "S_logic_llm": None,
            "S_logic_rgcn": s_rgcn,
            "beta": 0.0 if s_rgcn is not None else None,
            "explanation": "[logic-rgcn-only] 使用RGCN分数；未调用逻辑LLM。",
            "path_count": None,
        }

        packed.append((w, s_res, f_res, l_res))

        if args.log_every > 0 and idx % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            log.info(
                "[进度] %d/%d (%.1f%%) elapsed=%.1fs speed=%.2f triple/s",
                idx,
                len(triples_all),
                100.0 * idx / len(triples_all),
                elapsed,
                idx / elapsed if elapsed > 0 else 0.0,
            )

    norm_mode = str(args.agent_score_norm)
    struct_raw = [float(p[1].get("S_struct", 0.0)) for p in packed]
    fact_raw = [float(p[2].get("S_fact", 0.5)) for p in packed]
    logic_raw = [float(p[3].get("S_logic", 0.5)) for p in packed]
    if norm_mode != "none":
        struct_n = normalize_score_channel(struct_raw, norm_mode)
        fact_n = normalize_score_channel(fact_raw, norm_mode)
        logic_n = normalize_score_channel(logic_raw, norm_mode)
    else:
        struct_n = fact_n = logic_n = []

    for idx, (w, s_res, f_res, l_res) in enumerate(packed, start=1):
        if norm_mode in ("minmax", "zscore"):
            s_use = {**s_res, "S_struct": struct_n[idx - 1]}
            f_use = {**f_res, "S_fact": fact_n[idx - 1]}
            l_use = {**l_res, "S_logic": logic_n[idx - 1]}
        else:
            s_use, f_use, l_use = s_res, f_res, l_res

        if args.fusion_llm_reliability:
            rt = f_use.get("readable_triple")
            readable_triple = rt if isinstance(rt, tuple) and len(rt) == 3 else None
            # w 为固定 1:1:1。可选 w_ref 同上（消融无 MLP）。
            w_ref = None
            if args.fusion_llm_prompt_allocator_mlp:
                # 消融：无 Allocator MLP，prompt 中「MLP 参考」与任务分配一致，均为 1:1:1
                w_ref = _uniform_task_weights()
            fused = fusion_agent.decide(
                w,
                s_use,
                f_use,
                l_use,
                readable_triple=readable_triple,
                triple_id=triples_all[idx - 1],
                subgraph_info=None,
                reference_allocator_weights=w_ref,
            )
        else:
            fused = fusion_agent.decide(w, s_use, f_use, l_use, readable_triple=None)
        final_scores.append(float(fused["final_score"]))

        if args.log_every > 0 and idx % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            log.info(
                "[融合进度] %d/%d last_final=%.4f elapsed=%.1fs",
                idx,
                len(packed),
                final_scores[-1],
                elapsed,
            )

        if args.out_jsonl or args.out_jsonl_partial:
            rec: Dict[str, Any] = {
                "triple": list(triples_all[idx - 1]),
                "label": int(labels_all[idx - 1]),
                "final_score": float(fused["final_score"]),
                "weighted_score": float(fused["weighted_score"]),
                "allocator_weights": fused.get("allocator_weights"),
                "S_struct": s_use.get("S_struct"),
                "S_fact": f_use.get("S_fact"),
                "S_logic": l_use.get("S_logic"),
                "S_logic_rgcn": l_res.get("S_logic_rgcn"),
                "agent_score_norm": norm_mode,
            }
            if norm_mode != "none":
                rec["S_struct_raw"] = struct_raw[idx - 1]
                rec["S_fact_raw"] = fact_raw[idx - 1]
                rec["S_logic_raw"] = logic_raw[idx - 1]
            if args.fusion_llm_reliability:
                for k in ("reliability_weights", "allocator_baseline_score", "llm_reason"):
                    if k in fused and fused[k] is not None:
                        rec[k] = fused[k]
                rec["fusion_llm_reliability"] = True
            if args.out_jsonl:
                records.append(rec)
            if partial_fh is not None:
                partial_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if args.out_jsonl_partial_flush_every == 0 or idx % args.out_jsonl_partial_flush_every == 0:
                    partial_fh.flush()

    percents = [int(x) for x in args.percents]
    metrics = precision_recall_at_percents_for_errors(labels_all, final_scores, percents=percents)
    table = _format_prk_table(metrics, percents)
    print("\n" + "=" * 70)
    print("End-to-end Multi-Agent Error Detection (RGCN-logic-only, NO ALLOCATOR ablation)")
    print("=" * 70)
    print(f"dataset=fb15k237  noise_ratio={args.noise_ratio}  N={metrics['n_total']}  F(errors)={metrics['n_errors']}")
    print(table)
    print("=" * 70)
    log.info("评估结果:\n%s", table)

    if args.out_jsonl:
        bottom_p = max(0.0, min(100.0, float(args.explain_bottom_percent)))
        n_total = len(records)
        n_bottom = int((n_total * bottom_p + 99.0) // 100.0) if bottom_p > 0 else 0
        n_bottom = max(0, min(n_bottom, n_total))
        if n_bottom < n_total:
            order = sorted(range(n_total), key=lambda i: float(records[i]["final_score"]))
            keep = set(order[:n_bottom])
            records = [rec for i, rec in enumerate(records) if i in keep]

        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info(
            "逐条打分已写入: %s（bottom %.2f%%, n=%d）",
            out_path.resolve(),
            bottom_p,
            len(records),
        )

    if args.out_metrics:
        mpath = Path(args.out_metrics)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "fb15k237",
            "noise_ratio": float(args.noise_ratio),
            "seed": int(args.seed),
            "n_clean": len(clean_triples),
            "n_noisy": len(noisy_triples),
            "n_total": metrics["n_total"],
            "n_errors": metrics["n_errors"],
            "percents": percents,
            "metrics": {k: v for k, v in metrics.items() if k not in ("percents",)},
            "logic_mode": "rgcn_only",
            "fusion_mode": (
                "llm_reliability_weights"
                if args.fusion_llm_reliability
                else "weighted_only"
            ),
            "fusion_llm_reliability": bool(args.fusion_llm_reliability),
            "allocator": {
                "ablation": "no_allocator",
                "fixed_task_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                "emb_mode_for_structure_features": emb_mode,
                "fusion_alpha_note": fusion_alpha,
            },
            "structure_agent_config": asdict(struct_agent.config),
            "llm": {"fact_disabled": bool(args.no_llm), "source": llm_src},
            "logic_gcn": {
                "enabled": gcn_ctx is not None and not args.no_gcn,
                "checkpoint": str(gcn_ckpt_path.resolve()),
            },
            "agent_score_norm": str(args.agent_score_norm),
            "fusion_weights": {
                "mode": "fixed_uniform_no_allocator",
                "m_struct": 1.0 / 3.0,
                "m_fact": 1.0 / 3.0,
                "m_logic": 1.0 / 3.0,
            },
            "run_log": str(log_path.resolve()),
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("指标已写入: %s", mpath.resolve())

    if partial_fh is not None:
        partial_fh.close()

    log.info("全部完成。")


if __name__ == "__main__":
    main()

