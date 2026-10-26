#!/usr/bin/env python3
"""
将端到端日志中的 P@K / R@K（多档 top-%）画成 Precision–Recall 平面上的折线。

说明：
- 经典 sklearn 的 PR 曲线需要「每个样本的分数+标签」沿阈值扫描；日志里通常只有少量 K 档的
  Precision@K 与 Recall@K。
- 本脚本把各档的 (R@K, P@K) 视为 PR 空间中的离散点并顺次连线，用于展示「扩大筛查比例时
  精度–召回的折衷」，适用于不平衡排序式错误检测的汇报。

用法：
  python examples/plot_prk_points_curve.py --out outputs/try/prk_mac4e_50k.png

也可改下面 DEFAULT_* 或传参填入你自己的表。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot P@K vs R@K as PR-style curve")
    parser.add_argument(
        "--precision",
        type=float,
        nargs="+",
        default=[0.9719, 0.9243, 0.8698, 0.7294, 0.6488],
        help="各档 P@K（与 recall 列表等长）",
    )
    parser.add_argument(
        "--recall",
        type=float,
        nargs="+",
        default=[0.2220, 0.3840, 0.5528, 0.6068, 0.6592],
        help="各档 R@K",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=["1%", "2%", "3%", "4%", "5%"],
        help="各点标注（如 top 比例），长度需与 precision 一致",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MAC4E (fb15k237, N=325621, noise=5%) — P@K vs R@K",
    )
    parser.add_argument("--out", type=str, default="outputs/try/prk_curve.png")
    args = parser.parse_args()

    prec = args.precision
    rec = args.recall
    if len(prec) != len(rec):
        raise SystemExit("precision 与 recall 长度必须相同")
    labels = args.labels
    if len(labels) != len(prec):
        raise SystemExit("labels 数量需与 precision 一致")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("需要 matplotlib：pip install matplotlib") from e

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(rec, prec, "o-", color="#1f77b4", linewidth=2, markersize=8, label="Operating points")

    for i, (r, p, lb) in enumerate(zip(rec, prec, labels)):
        ax.annotate(lb, (r, p), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("Recall@K", fontsize=12)
    ax.set_ylabel("Precision@K", fontsize=12)
    ax.set_title(args.title, fontsize=12)
    ax.set_xlim(left=0.0, right=max(1.0, max(rec) * 1.08))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
