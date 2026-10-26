#!/usr/bin/env python3
"""
从 Allocator 训练保存的 history JSON 绘制损失曲线。

训练时由 ccg_pipeline/train_allocator.py 写入：
  outputs/logs/allocator_{dataset}_history.json
  字段：{"train_loss": [...]}

用法：
  # 自动选 outputs/logs 里最新修改的 allocator_*_history.json
  python scripts/plot_allocator_loss.py

  # 指定数据集名（对应 allocator_fb15k237_history.json）
  python scripts/plot_allocator_loss.py --dataset fb15k237

  # 指定 JSON 与输出图路径
  python scripts/plot_allocator_loss.py --json outputs/logs/allocator_fb15k237_history.json --out outputs/plots/my_loss.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _find_latest_history(log_dir: Path) -> Path | None:
    files = sorted(
        log_dir.glob("allocator_*_history.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 Allocator train_loss 曲线")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="history JSON 路径；不传则自动选 outputs/logs 下最新的 allocator_*_history.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名，如 fb15k237，使用 outputs/logs/allocator_{dataset}_history.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 PNG 路径；默认 outputs/plots/allocator_loss_from_history_{stem}.png",
    )
    args = parser.parse_args()

    log_dir = _ROOT / "outputs" / "logs"

    if args.json:
        hist_path = Path(args.json)
        if not hist_path.is_absolute():
            hist_path = _ROOT / hist_path
    elif args.dataset:
        hist_path = log_dir / f"allocator_{args.dataset.lower()}_history.json"
    else:
        hist_path = _find_latest_history(log_dir)

    if hist_path is None or not hist_path.exists():
        print(f"未找到 history 文件。请指定 --json 或 --dataset，或先完成 Allocator 训练。")
        sys.exit(1)

    with open(hist_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    losses = data.get("train_loss")
    if not losses or not isinstance(losses, list):
        print(f"{hist_path} 中缺少有效的 train_loss 列表。")
        sys.exit(1)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib：pip install matplotlib")
        sys.exit(1)

    epochs = list(range(1, len(losses) + 1))
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        plot_dir = _ROOT / "outputs" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        out_path = plot_dir / f"allocator_loss_from_history_{hist_path.stem}.png"
    elif not out_path.is_absolute():
        out_path = _ROOT / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, losses, color="C0", linewidth=1.5, marker="o", markersize=3, label="Train Loss (KLDiv)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Allocator Training Loss\n{hist_path.name}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(left=1)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"已读取: {hist_path}")
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
