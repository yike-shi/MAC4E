# 测试脚本：test_allocator_effect.py
# 特征与训练一致：先对 S1 做 log1p(100*S1)、对 support_paths 做 log1p，再 (x-mean)/std，再进 AllocatorMLP。
# 用法：python -m tests.test_allocator_effect --dataset fb15k237

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch

from ccg_agents.allocator import AllocatorMLP


def raw_to_training_feature(raw: torch.Tensor) -> torch.Tensor:
    """与 kg_utils.build_feature_vector_for_triple 一致：S1 → log1p(100*S1)，paths → log1p(paths)。"""
    out = raw.clone()
    out[:, 1] = torch.log1p(raw[:, 1] * 100.0)
    out[:, 2] = torch.log1p(raw[:, 2].clamp(min=0.0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Allocator 推理测试（特征变换+标准化）")
    parser.add_argument("--dataset", type=str, default="fb15k237", help="如 fb15k237，对应 outputs/allocator/{dataset}_allocator.pt")
    args = parser.parse_args()

    dataset_key = args.dataset.lower().replace("-", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _ROOT / "outputs" / "allocator" / f"{dataset_key}_allocator.pt"
    if not ckpt_path.exists():
        print(f"未找到 checkpoint: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "feat_mean" not in ckpt or "feat_std" not in ckpt:
        print("checkpoint 缺少 feat_mean/feat_std，请用新版 train_allocator 训练。")
        sys.exit(1)

    in_dim = ckpt["in_dim"]
    hidden_dims = tuple(ckpt["hidden_dims"])
    model = AllocatorMLP(in_dim=in_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feat_mean = torch.tensor(ckpt["feat_mean"], dtype=torch.float32, device=device)
    feat_std = torch.tensor(ckpt["feat_std"], dtype=torch.float32, device=device).clamp(min=1e-6)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - feat_mean) / feat_std

    # 每行 [raw_score, S1_raw, paths_raw, is_func, freq_h, freq_t]，与 kg_utils 顺序一致
    test_cases = [
        ("案例 1 (事实型：高 S1、路径多、非功能性)", [2.0, 0.8, 15.0, 0.0, 0.5, 0.5]),
        ("案例 2 (结构型：S1 极低、路径 0)", [12.0, 0.01, 0.0, 0.0, 0.1, 0.1]),
        ("案例 3 (逻辑型：功能性、路径适中)", [5.0, 0.3, 2.0, 1.0, 0.4, 0.4]),
        # 组 A 偏事实
        ("A1 事实型", [2.0, 0.82, 15.0, 0.0, 0.5, 0.5]),
        ("A2 事实型", [1.5, 0.90, 20.0, 0.0, 0.6, 0.6]),
        ("A3 事实型", [3.0, 0.70, 10.0, 0.0, 0.4, 0.4]),
        # 组 B 偏结构
        ("B1 结构型", [12.0, 0.02, 0.0, 0.0, 0.1, 0.1]),
        ("B2 结构型", [10.0, 0.05, 1.0, 0.0, 0.2, 0.2]),
        ("B3 结构型", [15.0, 0.01, 0.0, 0.0, 0.05, 0.05]),
        # 组 C 偏逻辑
        ("C1 逻辑型", [4.0, 0.5, 8.0, 1.0, 0.4, 0.4]),
        ("C2 逻辑型", [3.0, 0.6, 12.0, 1.0, 0.5, 0.5]),
        ("C3 逻辑型", [5.0, 0.45, 6.0, 1.0, 0.3, 0.3]),
        # 组 D 边界/混合
        ("D1 边界(路径少、S1中)", [6.0, 0.35, 2.0, 0.0, 0.3, 0.3]),
        ("D2 功能性但路径少", [7.0, 0.30, 1.0, 1.0, 0.2, 0.2]),
        ("D3 高S1+功能性+路径多", [2.5, 0.80, 18.0, 1.0, 0.5, 0.5]),
        # 组 E 接近训练分布
        ("E1 中等偏事实", [5.0, 0.40, 50.0, 0.0, 0.5, 0.5]),
        ("E2 低S1路径少", [8.0, 0.15, 2.0, 0.0, 0.2, 0.2]),
        ("E3 功能性路径多", [4.0, 0.50, 60.0, 1.0, 0.4, 0.4]),
    ]

    print(f"checkpoint: {ckpt_path}\n")
    with torch.no_grad():
        for name, raw_list in test_cases:
            raw = torch.tensor([raw_list], device=device, dtype=torch.float32)
            x = raw_to_training_feature(raw)
            x_norm = normalize(x)
            probs = model.probs(x_norm).cpu().numpy().ravel()
            print(f"{name}")
            print(f"  [结构, 事实, 逻辑]: [{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]")
            print()


if __name__ == "__main__":
    main()
