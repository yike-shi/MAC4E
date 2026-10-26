"""
仅在 WN18RR 上训练 Allocator：伪标签规则与 ccg_pipeline/train_allocator.py 默认规则不同，
全部写在本脚本内，不修改 train_allocator.py。

默认（针对 WordNet 式图）：
- 结构(S)：S1≤P30(s1) 且 support_paths≤P50(paths)（log1p 空间；用≤避免分位处并列时「严格<」为空）
- 逻辑(L)：默认 **原始支持路径条数** raw_paths = expm1(support_paths)（与 mining 的整数条数一致），
  **raw_paths ≥ logic_if_raw_paths_ge**（默认 **3**）。此前 P58 常等于 log1p(2)，与「≥2 条」等价，分布不会变。
- 其余：事实(F)

用法（在 CC_gra 根目录）：
  python examples/train_allocator_wn18rr.py
  python examples/train_allocator_wn18rr.py --logic_if_raw_paths_ge 4
  # 高级（会覆盖 raw）：--logic_paths_min_percentile 62 或 --logic_paths_min_factor 1.65
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.path_mining import build_graph
from ccg_data.type_constraints import TypeConstraints
from ccg_utils.kg_utils import build_entity_frequency, build_feature_vector_for_triple
from ccg_pipeline.train_allocator import (
    AllocatorDataset,
    _plot_allocator_loss,
    _setup_logger,
)
from ccg_pipeline.train_embeddings import (
    load_complex_from_checkpoint,
    load_config,
    load_model_config,
    load_transe_from_checkpoint,
)
from ccg_agents.allocator import AllocatorMLP

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


TripleId = Tuple[int, int, int]


def train_allocator_for_wn18rr(
    max_triples: Optional[int] = None,
    epochs: int | None = None,
    seed: int = 42,
    emb_model: str = "transe",
    fusion_alpha: float = 0.5,
    ckpt_suffix: Optional[str] = None,
    *,
    struct_s1_percentile: float = 30.0,
    struct_paths_max_percentile: float = 50.0,
    logic_if_raw_paths_ge: int = 3,
    logic_paths_min_percentile: Optional[float] = None,
    logic_paths_min_factor: Optional[float] = None,
) -> None:
    """WN18RR 专用 Allocator 训练；打标规则见模块 docstring。"""
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    model_cfg = load_model_config()
    alloc_cfg = model_cfg.get("allocator", {})
    hidden_dims = alloc_cfg.get("hidden_dims", [64, 32])
    lr = float(alloc_cfg.get("learning_rate", 0.001))
    if epochs is None:
        epochs = int(alloc_cfg.get("epochs", 50))

    output_dir = cfg.get("output_dir", "./outputs")
    base = _ROOT / output_dir.lstrip("./")
    base.mkdir(parents=True, exist_ok=True)
    alloc_dir = base / "allocator"
    alloc_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = "wn18rr"
    dataset_name_lower = dataset_name.lower()

    log_file = log_dir / f"train_allocator_{dataset_name_lower}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = _setup_logger(log_file)
    log.info("=" * 60)
    log.info("Allocator 训练（WN18RR 专用脚本，伪标规则见 examples/train_allocator_wn18rr.py）")
    log.info(f"  dataset={dataset_name}, device={device}")
    suf = (ckpt_suffix or "").strip()
    suf_safe = "".join(c for c in suf if c.isalnum() or c in "._-")[:80] if suf else ""
    ckpt_stem = f"{dataset_name_lower}_allocator_{suf_safe}" if suf_safe else f"{dataset_name_lower}_allocator"
    log.info(
        f"  emb_model={emb_model}, fusion_alpha={fusion_alpha}, hidden_dims={hidden_dims}, lr={lr}, epochs={epochs}, max_triples={max_triples}"
    )
    log.info(
        f"  WN18RR 伪标: struct_s1_percentile={struct_s1_percentile}, "
        f"struct_paths_max_percentile={struct_paths_max_percentile}, "
        f"logic_if_raw_paths_ge={logic_if_raw_paths_ge}, "
        f"logic_paths_min_percentile={logic_paths_min_percentile}, logic_paths_min_factor={logic_paths_min_factor}"
    )
    log.info(f"  保存权重: {ckpt_stem}.pt")
    log.info(f"  日志文件: {log_file}")
    log.info("=" * 60)

    if dataset_name_lower not in cfg or not isinstance(cfg[dataset_name_lower], dict):
        raise KeyError(f"配置中未找到 {dataset_name}")
    ds_cfg = cfg[dataset_name_lower]
    ds = KGDataset.load_from_files(
        ds_cfg["train_path"],
        ds_cfg["valid_path"],
        ds_cfg["test_path"],
    )
    train_triples: List[TripleId] = ds.train_triples

    if max_triples is not None and max_triples < len(train_triples):
        train_triples = random.sample(train_triples, max_triples)
        log.info(f"  采样训练三元组子集: {len(train_triples)} 条（共 {len(ds.train_triples)} 条）")
    else:
        log.info(f"  使用全部训练三元组: {len(train_triples)} 条")

    ckpt_path = base / "embeddings" / f"{dataset_name_lower}_transE.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TransE checkpoint 不存在: {ckpt_path}，请先训练 TransE。")

    model = load_transe_from_checkpoint(str(ckpt_path), device=device)
    model.eval()

    complex_model = None
    emb_mode = (emb_model or "transe").lower()
    if emb_mode == "complex" or emb_mode == "fusion":
        ckpt_complex = base / "embeddings" / f"{dataset_name_lower}_complEx.pt"
        if not ckpt_complex.exists():
            raise FileNotFoundError(f"ComplEx checkpoint 不存在: {ckpt_complex}，请先训练 ComplEx 或使用 --emb_model transe。")
        complex_model = load_complex_from_checkpoint(str(ckpt_complex), device=device)
        complex_model.eval()

    type_constraints = TypeConstraints()
    type_constraints.build_from_triples(train_triples)

    graph = build_graph(train_triples)
    entity_freq = build_entity_frequency(train_triples)

    raw_features: List[List[float]] = []
    for h, r, t in tqdm(train_triples, desc="  Collect features", unit="triple"):
        fea = build_feature_vector_for_triple(
            (h, r, t),
            model=model,
            device=device,
            type_constraints=type_constraints,
            graph=graph,
            entity_freq=entity_freq,
            use_sigmoid_score=True,
            max_path_len=3,
            complex_model=complex_model,
            emb_mode=emb_mode,
            fusion_alpha=fusion_alpha,
        )
        raw_features.append(fea)

    import numpy as np

    features_np = np.array(raw_features, dtype=np.float64)
    s1_scores = features_np[:, 1]
    paths_counts = features_np[:, 2]
    s1_median = np.median(s1_scores)
    s1_low_threshold = float(np.percentile(s1_scores, float(struct_s1_percentile)))
    paths_median = np.median(paths_counts)

    p_path = float(struct_paths_max_percentile)
    if not (0.0 < p_path < 100.0):
        raise ValueError(f"struct_paths_max_percentile 须在 (0,100) 内，当前: {struct_paths_max_percentile}")
    paths_struct_threshold = float(np.percentile(paths_counts, p_path))

    paths_logic_min: Optional[float] = None
    logic_mode: str = "raw_ge"

    if logic_paths_min_factor is not None:
        lf = float(logic_paths_min_factor)
        if lf <= 0:
            raise ValueError(f"logic_paths_min_factor 须为正，当前: {logic_paths_min_factor}")
        paths_logic_min = float(paths_median * lf)
        logic_mode = "median_factor"
        logic_desc = f"median×{lf:g}={paths_logic_min:.4f}"
        log.info(
            f"  统计结果：S1 中位数: {s1_median:.4f}, S1 结构低阈值=P{struct_s1_percentile:g}={s1_low_threshold:.4f}, "
            f"路径中位数: {paths_median:.4f}, 结构路径上界(P{p_path:g})={paths_struct_threshold:.4f}, "
            f"逻辑: {logic_desc}（support_paths≥此值）"
        )
        if paths_median > 1e-9 and 1.05 < lf < 1.58:
            log.warning(
                "  提示：median×factor 常落在 log1p(1) 与 log1p(2) 之间，微调可能不改变伪标。"
            )
    elif logic_paths_min_percentile is not None:
        lp = float(logic_paths_min_percentile)
        if not (0.0 < lp < 100.0):
            raise ValueError(f"logic_paths_min_percentile 须在 (0,100) 内，当前: {logic_paths_min_percentile}")
        paths_logic_min = float(np.percentile(paths_counts, lp))
        logic_mode = "logp_percentile"
        logic_desc = f"P{lp:g}={paths_logic_min:.4f}"
        log.info(
            f"  统计结果：S1 中位数: {s1_median:.4f}, S1 结构低阈值=P{struct_s1_percentile:g}={s1_low_threshold:.4f}, "
            f"路径中位数: {paths_median:.4f}, 结构路径上界(P{p_path:g})={paths_struct_threshold:.4f}, "
            f"逻辑: support_paths≥{logic_desc}"
        )
        if abs(paths_logic_min - np.log1p(2.0)) < 1e-3:
            log.warning(
                "  提示：逻辑阈值≈log1p(2)，与「原始路径条数≥2」等价；若伪标分布与旧实验相同，"
                "请改用默认 --logic_if_raw_paths_ge 3（按整数条数区分）。"
            )
    else:
        thr = int(logic_if_raw_paths_ge)
        if thr < 0:
            raise ValueError(f"logic_if_raw_paths_ge 须为非负整数，当前: {logic_if_raw_paths_ge}")
        raw_all = np.expm1(paths_counts)
        log.info(
            f"  统计结果：S1 中位数: {s1_median:.4f}, S1 结构低阈值=P{struct_s1_percentile:g}={s1_low_threshold:.4f}, "
            f"路径中位数: {paths_median:.4f}, 结构路径上界(P{p_path:g})={paths_struct_threshold:.4f}, "
            f"逻辑: 原始支持路径数 raw_paths=expm1(support_paths) ≥ {thr} "
            f"（raw 的 min/median/max={raw_all.min():.0f}/{np.median(raw_all):.1f}/{raw_all.max():.0f}）"
        )

    log.info(
        "  结构判定: support_paths<=结构路径上界 且 s1<=S1阈值（含等号，避免 log1p 并列导致空集）"
    )

    features: List[List[float]] = []
    labels: List[List[float]] = []
    for fea in raw_features:
        s1, support_paths = fea[1], fea[2]
        struct_ok = (support_paths <= paths_struct_threshold) and (s1 <= s1_low_threshold)
        if struct_ok:
            label = [0.4, 0.3, 0.3]
        elif logic_mode == "raw_ge":
            raw_n = float(np.expm1(support_paths))
            if raw_n >= float(logic_if_raw_paths_ge):
                label = [0.3, 0.3, 0.4]
            else:
                label = [0.3, 0.4, 0.3]
        elif paths_logic_min is not None and support_paths >= paths_logic_min:
            label = [0.3, 0.3, 0.4]
        else:
            label = [0.3, 0.4, 0.3]
        features.append(fea)
        labels.append(label)

    in_dim = len(features[0]) if features else 0
    if in_dim == 0:
        log.error("特征为空，无法训练 Allocator。")
        return

    agent_counts: dict = {"结构(S)": 0, "事实(F)": 0, "逻辑(L)": 0}
    for lab in labels:
        i = int(np.argmax(lab))
        if i == 0:
            agent_counts["结构(S)"] += 1
        elif i == 2:
            agent_counts["逻辑(L)"] += 1
        else:
            agent_counts["事实(F)"] += 1
    log.info("=" * 30)
    log.info("动态调整后的伪标签分布 (Label Distribution):")
    for agent_type in ("结构(S)", "事实(F)", "逻辑(L)"):
        count = agent_counts[agent_type]
        percent = (count / len(labels)) * 100
        log.info(f"  {agent_type} : {count} 条 ({percent:.2f}%)")
    log.info("=" * 30)

    F = np.array(features, dtype=np.float64)
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    std[std < 1e-8] = 1.0
    F = (F - mean) / std
    features = F.tolist()
    feat_mean_std = {"mean": mean.tolist(), "std": std.tolist()}

    alloc_ds = AllocatorDataset(features, labels)
    loader = DataLoader(alloc_ds, batch_size=256, shuffle=True, num_workers=0)

    if isinstance(hidden_dims, (list, tuple)):
        h_dims = [int(x) for x in hidden_dims]
    else:
        h_dims = [int(hidden_dims)]
    alloc_model = AllocatorMLP(in_dim=in_dim, hidden_dims=h_dims).to(device)
    optimizer = torch.optim.Adam(alloc_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    history: Dict[str, List[float]] = {"train_loss": []}

    log.info("  开始训练 Allocator MLP...")
    for epoch in range(epochs):
        alloc_model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred_logp = alloc_model(x)
            y = y.clamp_min(0.0)
            y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-12)
            loss = criterion(pred_logp, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)
        scheduler.step(avg_loss)
        log.info(f"  epoch {epoch+1}/{epochs}  avg_loss={avg_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

    plot_dir = base / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / f"allocator_{dataset_name_lower}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    log.info(f"  训练曲线数据已保存: {history_path}")
    plot_path = _plot_allocator_loss(history, plot_dir, dataset_name)
    if plot_path is not None:
        log.info(f"  训练损失曲线图已保存: {plot_path}")
    else:
        log.info("  未安装 matplotlib，跳过绘图。")

    alloc_ckpt = alloc_dir / f"{ckpt_stem}.pt"
    torch.save(
        {
            "state_dict": alloc_model.state_dict(),
            "in_dim": in_dim,
            "hidden_dims": h_dims,
            "feat_mean": feat_mean_std["mean"],
            "feat_std": feat_mean_std["std"],
            "emb_model": emb_mode,
            "fusion_alpha": fusion_alpha,
        },
        alloc_ckpt,
    )
    log.info(f"Allocator 训练完成，模型已保存到: {alloc_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WN18RR Allocator（独立脚本，不修改 train_allocator.py）"
    )
    parser.add_argument(
        "--struct_s1_percentile",
        type=float,
        default=55.0,
        help="结构类：S1≤P该分位（默认 40）",
    )
    parser.add_argument(
        "--struct_paths_max_percentile",
        type=float,
        default=65.0,
        help="结构类：support_paths≤P该分位（默认 60）",
    )
    parser.add_argument(
        "--logic_if_raw_paths_ge",
        type=int,
        default=4,
        help="逻辑类：原始路径条数 expm1(support_paths)≥该整数（默认 3，比「≥2」更严，事实更多）",
    )
    parser.add_argument(
        "--logic_paths_min_percentile",
        type=float,
        default=None,
        help="若指定则逻辑改为 support_paths≥P该分位（覆盖 raw 阈值；易与 log1p(2) 重合需谨慎）",
    )
    parser.add_argument(
        "--logic_paths_min_factor",
        type=float,
        default=None,
        help="若指定则逻辑改为 support_paths≥median×系数（覆盖 raw；优先于分位数）",
    )
    parser.add_argument(
        "--emb_model",
        type=str,
        default="transe",
        choices=["transe", "complex", "fusion"],
    )
    parser.add_argument("--fusion_alpha", type=float, default=0.5)
    parser.add_argument("--max_triples", type=int, default=None, metavar="N")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_suffix", type=str, default=None, metavar="SUFFIX")
    args = parser.parse_args()

    p_s1 = float(args.struct_s1_percentile)
    p_path = float(args.struct_paths_max_percentile)
    if not (0.0 < p_s1 < 100.0):
        parser.error("--struct_s1_percentile 须在 (0, 100) 内")
    if not (0.0 < p_path < 100.0):
        parser.error("--struct_paths_max_percentile 须在 (0, 100) 内")
    if args.logic_paths_min_percentile is not None:
        lp_log = float(args.logic_paths_min_percentile)
        if not (0.0 < lp_log < 100.0):
            parser.error("--logic_paths_min_percentile 须在 (0, 100) 内")
    if args.logic_paths_min_factor is not None and float(args.logic_paths_min_factor) <= 0:
        parser.error("--logic_paths_min_factor 须为正")
    if int(args.logic_if_raw_paths_ge) < 0:
        parser.error("--logic_if_raw_paths_ge 须为非负整数")

    train_allocator_for_wn18rr(
        max_triples=args.max_triples,
        epochs=args.epochs,
        seed=args.seed,
        emb_model=args.emb_model,
        fusion_alpha=args.fusion_alpha,
        ckpt_suffix=args.ckpt_suffix,
        struct_s1_percentile=p_s1,
        struct_paths_max_percentile=p_path,
        logic_if_raw_paths_ge=int(args.logic_if_raw_paths_ge),
        logic_paths_min_percentile=(
            float(args.logic_paths_min_percentile) if args.logic_paths_min_percentile is not None else None
        ),
        logic_paths_min_factor=(
            float(args.logic_paths_min_factor) if args.logic_paths_min_factor is not None else None
        ),
    )


if __name__ == "__main__":
    main()
