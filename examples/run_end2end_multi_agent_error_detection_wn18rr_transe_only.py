"""
WN18RR 仅 TransE 的错误检测（与端到端脚本相同的 P@K/R@K 指标口径）。

要点：
- 数据构造与 run_end2end_multi_agent_error_detection_wn18rr_rgcn_logic_fusion_llm.py 保持一致：
  使用 train+valid+test 全量干净三元组后注噪（label: 1=clean, 0=noisy）。
- 仅使用你训练好的 TransE checkpoint 打分：
  S_transe = sigmoid(-d(h+r,t))，分数越高越像正确三元组。
- 错误检测排序按 final_score 升序（低分优先）计算 P@K/R@K：
  precision_at_p / recall_at_p / k_eff_p / I_found_p，与端到端脚本一致。
"""

from __future__ import annotations

import argparse
import datetime
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
from ccg_pipeline.train_embeddings import (
    compute_struct_confidence,
    load_config,
    load_transe_from_checkpoint,
)
from run_end2end_multi_agent_error_detection_fb15k237_rgcn_logic_only import (
    _format_prk_table,
    _setup_logger,
    precision_recall_at_percents_for_errors,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it


TripleId = Tuple[int, int, int]


def _resolve_ckpt(args_ckpt: str | None, base: Path, ds_key: str) -> Path:
    if args_ckpt:
        p = Path(args_ckpt)
        return p if p.is_absolute() else (_ROOT / str(p).lstrip("./"))
    return base / "embeddings" / f"{ds_key}_transE.pt"


def _build_payload(
    *,
    ds_key: str,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    percents: Sequence[int],
    clean_n: int,
    noisy_n: int,
    transe_ckpt: Path,
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
        "logic_mode": "transe_only",
        "fusion_mode": "none",
        "allocator": {"enabled": False},
        "llm": {
            "fact_disabled": True,
            "fusion_disabled": True,
            "source": "disabled",
        },
        "logic_gcn": {
            "enabled": False,
            "checkpoint": "",
        },
        "agent_score_norm": "none",
        "run_log": str(log_path.resolve()),
        "transe_checkpoint": str(transe_ckpt.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WN18RR 错误检测（仅 TransE，端到端同口径 P@K/R@K）"
    )
    parser.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr"])
    parser.add_argument("--noise_ratio", type=float, default=0.05, choices=[0.05, 0.10, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_triples", type=int, default=None)
    parser.add_argument("--percents", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--transe_ckpt",
        type=str,
        default=None,
        help="默认 outputs/embeddings/{dataset}_transE.pt",
    )
    parser.add_argument("--out_metrics", type=str, default=None)
    parser.add_argument("--out_jsonl", type=str, default=None, help="可选：保存逐条 final_score")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    ds_key = args.dataset.lower()
    ds_cfg = cfg.get(ds_key)
    if not isinstance(ds_cfg, dict):
        raise KeyError(f"未找到 {ds_key} 配置（ccg_config/datasets.yaml）")

    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        Path(args.log_file)
        if args.log_file
        else base / "logs" / f"end2end_{ds_key}_transe_only_{ts}.log"
    )
    log = _setup_logger(log_path)
    log.info("=" * 72)
    log.info("WN18RR 仅 TransE 错误检测启动")
    log.info("CLI=%s", json.dumps(vars(args), ensure_ascii=False, default=str))
    log.info("log_file=%s", log_path.resolve())

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

    transe_ckpt = _resolve_ckpt(args.transe_ckpt, base, ds_key)
    if not transe_ckpt.exists():
        raise FileNotFoundError(f"未找到 TransE checkpoint: {transe_ckpt}")
    transe = load_transe_from_checkpoint(str(transe_ckpt), device=device)
    transe.eval()

    triples_all: List[TripleId] = [(h, r, t) for (h, r, t, _lb) in labeled]
    labels_all: List[int] = [int(lb) for (_h, _r, _t, lb) in labeled]

    final_scores: List[float] = []
    records: List[Dict[str, Any]] = []
    for i, ((h, r, t), lb) in enumerate(
        tqdm(zip(triples_all, labels_all), total=len(triples_all), desc="TransE scoring", unit="triple"),
        start=1,
    ):
        s_transe = float(compute_struct_confidence((h, r, t), transe, device))
        final_scores.append(s_transe)
        if args.out_jsonl:
            records.append(
                {
                    "triple": [int(h), int(r), int(t)],
                    "label": int(lb),
                    "final_score": s_transe,
                    "S_transe": s_transe,
                }
            )
        if i % 50000 == 0:
            log.info("[打分进度] %d/%d", i, len(triples_all))

    percents = [int(x) for x in args.percents]
    metrics = precision_recall_at_percents_for_errors(labels_all, final_scores, percents=percents)
    table = _format_prk_table(metrics, percents)

    print("\n" + "=" * 70)
    print("WN18RR Error Detection (TransE only)")
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
            log_path=log_path,
        )
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("指标已写入: %s", mpath.resolve())

    log.info("全部完成。")


if __name__ == "__main__":
    main()
