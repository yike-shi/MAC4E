"""
FB15K-237 端到端错误检测（仅结构 + 事实，无 Allocator、无逻辑智能体）：

- 结构智能体：StructureConstraintAgent（与主端到端一致）
- 事实智能体：FactVerificationAgent，默认调用 LLM（可用 --no_llm 做消融，事实分固定 0.5）
- 融合：固定两路加权 final_score = w_struct * S_struct + w_fact * S_fact，默认 w_struct=w_fact=0.5（1:1）
- 无任务分配、无逻辑/RGCN

评测与主端到端一致：按 final_score 升序，错误检测 P@K / R@K（K 默认 1%–5%）；可选列归一化后再融合。

在项目根目录 CC_gra 下执行（示例）::

    cd CC_gra
    python examples/run_end2end_multi_agent_error_detection_fb15k237_struct_fact_only.py \\
      --max_triples 5000 --noise_ratio 0.05 \\
      --out_metrics outputs/logic/metrics_struct_fact_only.json \\
      --out_jsonl outputs/logic/scores_struct_fact_only.jsonl \\
      --log_file outputs/logic/struct_fact_only.log

常用可调参数：--noise_ratio、--max_triples、--seed、--struct_weight/--fact_weight（默认 0.5/0.5 即 1:1）、
--no_llm、--agent_score_norm、--percents、--explain_bottom_percent、--log_every、环境变量 LOCAL_LLM_MODEL / CUDA_VISIBLE_DEVICES。

完整列表见：python examples/run_end2end_multi_agent_error_detection_fb15k237_struct_fact_only.py --help
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
from typing import Any, Dict, List, Sequence, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.noise_injector import inject_noise
from ccg_data.path_mining import build_graph
from ccg_data.text_mapping import load_fb15k237_text_maps
from ccg_utils.agent_score_norm import normalize_score_channel
from ccg_pipeline.train_embeddings import (
    load_complex_from_checkpoint,
    load_config,
    load_model_config,
    load_transe_from_checkpoint,
)
from ccg_agents.fact_agent import FactAgentConfig, FactVerificationAgent
from ccg_agents.llm_client import LocalQwenClient, MockLLMClient
from ccg_agents.structure_agent import StructureAgentConfig, StructureConstraintAgent

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it


TripleId = Tuple[int, int, int]


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("end2end_struct_fact_only")
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
        ["k(条数)"] + [str(metrics.get(f"k_eff_{p}", 0)) for p in percents],
        ["I(错数)"] + [str(metrics.get(f"I_found_{p}", 0)) for p in percents],
    ]
    col_w = [max(len(str(x)) for x in col) for col in zip(header, *rows)]

    def fmt_row(items):
        return "  " + " | ".join(str(it).ljust(w) for it, w in zip(items, col_w))

    lines = [fmt_row(header), "  " + "-+-".join("-" * w for w in col_w)]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def _normalize_two_weights(a: float, b: float) -> Tuple[float, float]:
    if a < 0 or b < 0:
        raise ValueError("struct_weight / fact_weight 须为非负数")
    s = a + b
    if s <= 1e-12:
        raise ValueError("两路权重之和须 > 0")
    return a / s, b / s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FB15K-237 端到端：仅结构+事实，固定两路融合（默认 1:1），无 Allocator/逻辑"
    )
    parser.add_argument("--dataset", type=str, default="fb15k237", choices=["fb15k237"])
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.05,
        help="注入噪声比例 [0,1]，默认 0.05（5%%）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_triples", type=int, default=None, help="限制抽样正例条数；不设则全库注噪")
    parser.add_argument("--out_metrics", type=str, default=None)
    parser.add_argument("--out_jsonl", type=str, default=None)
    parser.add_argument(
        "--explain_bottom_percent",
        type=float,
        default=100.0,
        help="--out_jsonl 时按 final_score 升序只导出最低分底部 x%%",
    )
    parser.add_argument("--percents", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--no_llm", action="store_true", help="禁用事实 LLM（S_fact 固定 0.5）")
    parser.add_argument(
        "--struct_weight",
        type=float,
        default=0.5,
        help="结构分支权重（与 --fact_weight 归一化后相乘 S_struct）",
    )
    parser.add_argument(
        "--fact_weight",
        type=float,
        default=0.5,
        help="事实分支权重（与 --struct_weight 归一化；默认 1:1）",
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument(
        "--agent_score_norm",
        type=str,
        default="zscore",
        choices=["none", "minmax", "zscore"],
        help="融合前对本 run 全部样本的 S_struct、S_fact 分别做列归一化（两路固定加权；默认 zscore）",
    )
    args = parser.parse_args()

    w_s, w_f = _normalize_two_weights(float(args.struct_weight), float(args.fact_weight))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    model_cfg = load_model_config()
    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else base / "logs" / f"end2end_struct_fact_only_{ts}.log"
    log = _setup_logger(log_path)

    log.info("=" * 72)
    log.info("端到端（仅结构+事实，两路融合）启动")
    log.info("log_file=%s", log_path.resolve())
    log.info("CLI=%s", json.dumps(vars(args), ensure_ascii=False, default=str))
    log.info("融合权重（归一化后）: w_struct=%.6f w_fact=%.6f", w_s, w_f)
    if args.agent_score_norm != "none":
        log.info("agent_score_norm=%s", args.agent_score_norm)

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

    complex_model = None
    complex_ckpt = base / "embeddings" / "fb15k237_complEx.pt"
    if complex_ckpt.exists():
        complex_model = load_complex_from_checkpoint(str(complex_ckpt), device=device)
        complex_model.eval()

    llm_client, llm_src = _create_llm_client(prefer_local_qwen=not args.no_llm)
    if args.no_llm:
        log.info("事实 LLM 已禁用 (--no_llm)")
    else:
        log.info("事实 LLM: %s", llm_src or "Mock (fallback)")

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

    triples_all: List[TripleId] = [(h, r, t) for (h, r, t, _y) in labeled]
    labels_all: List[int] = [int(_y) for (_h, _r, _t, _y) in labeled]
    packed: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    t0 = time.perf_counter()
    for idx, triple in enumerate(
        tqdm(triples_all, desc="Scoring (struct + fact only)", unit="triple"), start=1
    ):
        s_res = struct_agent.score_triple(triple)
        if args.no_llm:
            f_res = {"S_fact": 0.5, "explanation": None, "skipped": True, "readable_triple": None}
        else:
            f_res = fact_agent.score_triple(triple)
        packed.append((s_res, f_res))

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
    struct_raw = [float(p[0].get("S_struct", 0.0)) for p in packed]
    fact_raw = [float(p[1].get("S_fact", 0.5)) for p in packed]
    struct_n = normalize_score_channel(struct_raw, norm_mode)
    fact_n = normalize_score_channel(fact_raw, norm_mode)

    final_scores: List[float] = []
    records: List[Dict[str, Any]] = []

    for i in range(len(packed)):
        s_res, f_res = packed[i]
        if norm_mode != "none":
            ss = struct_n[i]
            ff = fact_n[i]
        else:
            ss = float(s_res.get("S_struct", 0.0))
            ff = float(f_res.get("S_fact", 0.5))
        final = w_s * ss + w_f * ff
        final_scores.append(final)

        if args.out_jsonl:
            rec: Dict[str, Any] = {
                "triple": list(triples_all[i]),
                "label": int(labels_all[i]),
                "final_score": float(final),
                "weighted_score": float(final),
                "fusion": {"w_struct": w_s, "w_fact": w_f, "mode": "struct_fact_only"},
                "S_struct": ss,
                "S_fact": ff,
                "agent_score_norm": norm_mode,
            }
            if norm_mode != "none":
                rec["S_struct_raw"] = struct_raw[i]
                rec["S_fact_raw"] = fact_raw[i]
            records.append(rec)

        if args.log_every > 0 and (i + 1) % args.log_every == 0:
            log.info(
                "[融合进度] %d/%d last_final=%.4f",
                i + 1,
                len(packed),
                final_scores[-1],
            )

    percents = [int(x) for x in args.percents]
    metrics = precision_recall_at_percents_for_errors(labels_all, final_scores, percents=percents)
    table = _format_prk_table(metrics, percents)
    print("\n" + "=" * 70)
    print("End-to-end Error Detection (struct + fact only, 1:1 fusion)")
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
            records = [rec for j, rec in enumerate(records) if j in keep]

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
            "metrics": {k: v for k, v in metrics.items() if k != "percents"},
            "pipeline": "struct_fact_only",
            "fusion": {
                "w_struct": w_s,
                "w_fact": w_f,
                "raw_struct_weight": float(args.struct_weight),
                "raw_fact_weight": float(args.fact_weight),
            },
            "structure_agent_config": asdict(struct_agent.config),
            "llm": {"fact_disabled": bool(args.no_llm), "source": llm_src},
            "agent_score_norm": str(args.agent_score_norm),
            "run_log": str(log_path.resolve()),
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("指标已写入: %s", mpath.resolve())

    log.info("全部完成。")


if __name__ == "__main__":
    main()
