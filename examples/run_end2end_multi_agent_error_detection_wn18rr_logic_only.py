"""
WN18RR 仅逻辑智能体（RGCN）的错误检测。

口径与 run_end2end_multi_agent_error_detection_wn18rr_rgcn_logic_fusion_llm.py 对齐：
- 数据：train+valid+test 全量三元组，按 noise_ratio 注入噪声（label: 1=clean, 0=noisy）
- 排序：final_score 按升序（低分更可疑）进行错误检测
- 指标：precision_at_p / recall_at_p / k_eff_p / I_found_p
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.noise_injector import inject_noise
from ccg_models.gcn_encoder import build_adj_from_triples
from ccg_pipeline.train_embeddings import (
    load_config,
    load_model_config,
    load_transe_from_checkpoint,
)
from run_end2end_multi_agent_error_detection_fb15k237_rgcn_logic_only import (
    _compute_s_logic_rgcn,
    _format_prk_table,
    _load_gcn_if_available,
    _setup_logger,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it


TripleId = Tuple[int, int, int]


def _resolve_ckpt(args_ckpt: str | None, default_ckpt: Path) -> Path:
    if args_ckpt:
        p = Path(args_ckpt)
        return p if p.is_absolute() else (_ROOT / str(p).lstrip("./"))
    return default_ckpt


def _build_payload(
    *,
    ds_key: str,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    percents: Sequence[int],
    clean_n: int,
    noisy_n: int,
    transe_ckpt: Path,
    gcn_ckpt: Path,
    gcn_enabled: bool,
    log_path: Path,
) -> Dict[str, Any]:
    return {
        "dataset": ds_key,
        "noise_ratio": float(args.noise_ratio),
        "seed": int(args.seed),
        "n_clean": int(clean_n),
        "n_noisy": int(noisy_n),
        "n_total": int(metrics["n_total"]),
        "n_errors": int(metrics["n_errors"]),
        "percents": [int(p) for p in percents],
        "metrics": {k: v for k, v in metrics.items() if k not in ("percents",)},
        "logic_mode": "rgcn_only",
        "fusion_mode": "none",
        "allocator": {"enabled": False},
        "structure_agent_config": None,
        "llm": {
            "fact_disabled": True,
            "fusion_disabled": True,
            "source": "disabled",
        },
        "logic_gcn": {
            "enabled": bool(gcn_enabled),
            "checkpoint": str(gcn_ckpt.resolve()) if gcn_ckpt.exists() else str(gcn_ckpt),
        },
        "agent_score_norm": "none",
        "run_log": str(log_path.resolve()),
        "transe_checkpoint": str(transe_ckpt.resolve()),
    }


def _precision_recall_at_percents_for_errors_with_tie_break(
    labels: Sequence[int],
    scores: Sequence[float],
    triples: Sequence[TripleId],
    percents: Sequence[int],
    tie_break_seed: int,
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

    def tie_key(tri: TripleId) -> int:
        h, r, t = tri
        raw = f"{h}_{r}_{t}_{tie_break_seed}".encode("utf-8")
        # 可复现的并列分打散键；仅在 score 相等时生效
        return int(hashlib.md5(raw).hexdigest()[:8], 16)

    order = sorted(range(n), key=lambda i: (float(scores[i]), tie_key(triples[i])))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WN18RR 错误检测（仅 Logic RGCN，端到端同口径 P@K/R@K）"
    )
    parser.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr"])
    parser.add_argument("--noise_ratio", type=float, default=0.05, choices=[0.05, 0.10, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_triples", type=int, default=None)
    parser.add_argument("--percents", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--transe_ckpt", type=str, default=None, help="默认 outputs/embeddings/{dataset}_transE.pt")
    parser.add_argument("--gcn_ckpt", type=str, default=None, help="默认 outputs/gcn/{dataset}_gcn.pt")
    parser.add_argument(
        "--logic_temperature",
        type=float,
        default=5.0,
        help="逻辑分温度 T：s=sigmoid(logit/T)，T>1 可减少分数挤压到 0/1",
    )
    parser.add_argument(
        "--score_eps",
        type=float,
        default=1e-6,
        help="分数截断下界/上界：s=clip(s, eps, 1-eps)，避免精确 0/1",
    )
    parser.add_argument(
        "--tie_break_seed",
        type=int,
        default=42,
        help="并列分打散种子（可复现）；用于避免大量 ties 被输入顺序污染",
    )
    parser.add_argument("--out_metrics", type=str, default=None)
    parser.add_argument("--out_jsonl", type=str, default=None, help="可选：保存逐条 final_score")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    model_cfg = load_model_config()
    ds_key = args.dataset.lower()
    ds_cfg = cfg.get(ds_key)
    if not isinstance(ds_cfg, dict):
        raise KeyError(f"未找到 {ds_key} 配置（ccg_config/datasets.yaml）")

    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        Path(args.log_file)
        if args.log_file
        else base / "logs" / f"end2end_{ds_key}_logic_only_{ts}.log"
    )
    log = _setup_logger(log_path)
    log.info("=" * 72)
    log.info("WN18RR 仅 Logic RGCN 错误检测启动")
    log.info("CLI=%s", json.dumps(vars(args), ensure_ascii=False, default=str))
    log.info("log_file=%s", log_path.resolve())
    log.info("logic_temperature=%.4f score_eps=%.8f", float(args.logic_temperature), float(args.score_eps))
    log.info("tie_break_seed=%d", int(args.tie_break_seed))

    ds = KGDataset.load_from_files(
        str(_ROOT / ds_cfg["train_path"].lstrip("./")),
        str(_ROOT / ds_cfg["valid_path"].lstrip("./")),
        str(_ROOT / ds_cfg["test_path"].lstrip("./")),
    )
    clean_all: List[TripleId] = list(ds.train_triples) + list(ds.valid_triples) + list(ds.test_triples)
    if args.max_triples is not None and args.max_triples > 0 and len(clean_all) > args.max_triples:
        clean_all = random.sample(clean_all, args.max_triples)

    clean_triples, noisy_triples, labeled = inject_noise(
        clean_all,
        num_entities=ds.num_entities,
        noise_ratio=float(args.noise_ratio),
        mode="head_tail",
    )
    log.info("N=%d clean=%d noisy=%d", len(labeled), len(clean_triples), len(noisy_triples))

    transe_ckpt = _resolve_ckpt(args.transe_ckpt, base / "embeddings" / f"{ds_key}_transE.pt")
    if not transe_ckpt.exists():
        raise FileNotFoundError(f"未找到 TransE checkpoint: {transe_ckpt}")
    transe = load_transe_from_checkpoint(str(transe_ckpt), device=device)
    transe.eval()
    entity_emb = transe.ent_emb.weight.detach()
    rel_emb = transe.rel_emb

    gcn_ckpt = _resolve_ckpt(args.gcn_ckpt, base / "gcn" / f"{ds_key}_gcn.pt")
    if not gcn_ckpt.exists():
        raise FileNotFoundError(f"未找到 GCN checkpoint: {gcn_ckpt}")
    gcn_ctx = _load_gcn_if_available(gcn_ckpt, device, model_cfg)
    if gcn_ctx is None:
        raise RuntimeError(f"GCN checkpoint 加载失败: {gcn_ckpt}")
    _enc, _sc, _st, k_hop, _bd = gcn_ctx
    log.info("逻辑RGCN已启用: k_hop=%d ckpt=%s", k_hop, gcn_ckpt.resolve())

    adj_for_gcn = build_adj_from_triples(ds.train_triples)
    triples_all: List[TripleId] = [(h, r, t) for (h, r, t, _lb) in labeled]
    labels_all: List[int] = [int(lb) for (_h, _r, _t, lb) in labeled]

    final_scores: List[float] = []
    records: List[Dict[str, Any]] = []
    for i, triple in enumerate(
        tqdm(triples_all, total=len(triples_all), desc="Logic-only scoring", unit="triple"),
        start=1,
    ):
        s_rgcn = _compute_s_logic_rgcn(
            triple,
            gcn_ctx=gcn_ctx,
            adj=adj_for_gcn,
            entity_emb=entity_emb,
            rel_emb=rel_emb,
            device=device,
            model_cfg=model_cfg,
            temperature=float(args.logic_temperature),
            score_eps=float(args.score_eps),
        )
        s_logic = float(s_rgcn) if s_rgcn is not None else 0.5
        final_scores.append(s_logic)
        if args.out_jsonl:
            records.append(
                {
                    "triple": list(triple),
                    "label": int(labels_all[i - 1]),
                    "final_score": s_logic,
                    "S_logic": s_logic,
                    "S_logic_rgcn": s_rgcn,
                    "logic_temperature": float(args.logic_temperature),
                    "score_eps": float(args.score_eps),
                }
            )
        if i % 50000 == 0:
            log.info("[打分进度] %d/%d", i, len(triples_all))

    percents = [int(x) for x in args.percents]
    metrics = _precision_recall_at_percents_for_errors_with_tie_break(
        labels_all,
        final_scores,
        triples_all,
        percents=percents,
        tie_break_seed=int(args.tie_break_seed),
    )
    table = _format_prk_table(metrics, percents)
    print("\n" + "=" * 70)
    print("WN18RR Error Detection (Logic RGCN only)")
    print("=" * 70)
    print(
        f"dataset={ds_key}  noise_ratio={args.noise_ratio}  N={metrics['n_total']}  F(errors)={metrics['n_errors']}"
    )
    print(table)
    print("=" * 70)
    log.info("评估结果:\n%s", table)

    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("逐条打分已写入: %s", out_path.resolve())

    if args.out_metrics:
        mpath = Path(args.out_metrics)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_payload(
            ds_key=ds_key,
            args=args,
            metrics=metrics,
            percents=percents,
            clean_n=len(clean_triples),
            noisy_n=len(noisy_triples),
            transe_ckpt=transe_ckpt,
            gcn_ckpt=gcn_ckpt,
            gcn_enabled=True,
            log_path=log_path,
        )
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("指标已写入: %s", mpath.resolve())

    log.info("全部完成。")


if __name__ == "__main__":
    main()
