"""
Allocator 训练脚本：

- 利用已训练好的 TransE、TypeConstraints、图结构等，为训练集中的三元组构造特征向量 x；
- 两阶段动态打标：先收集特征并计算数据分布（S1 分位阈值、路径中位数等），再按规则生成伪标签 [m_s, m_f, m_l]。

三个 Agent 的职责简述：
  - 结构(S)：嵌入/图结构类可疑（低 S1、路径支撑少等；类型冲突已移除）。
  - 事实(F)：语义/嵌入合理性（TransE 分数、路径支持等）。
  - 逻辑(L)：功能性关系，唯一性等逻辑约束，多跳链式推理。

打标规则（路径均在 log1p 空间；类型冲突已移除）：
  - 结构：路径偏少（< paths_struct_threshold）且 S1 低于 80% 分位阈值 -> [0.8, 0.1, 0.1]
  - 逻辑：（暂不使用 is_func）路径≥paths_logic_min -> [0.1, 0.1, 0.8]
  - 否则：事实 -> [0.1, 0.8, 0.1]

- 以 KLDivLoss 训练 AllocatorMLP（与 log_softmax 输出一致），默认保存 outputs/allocator/{dataset}_allocator.pt；
  可用 --ckpt_suffix 保存为 {dataset}_allocator_{suffix}.pt（含 feat_mean/feat_std 供推理标准化）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.type_constraints import TypeConstraints
from ccg_data.path_mining import build_graph
from ccg_utils.kg_utils import build_entity_frequency, build_feature_vector_for_triple
from ccg_pipeline.train_embeddings import (
    load_transe_from_checkpoint,
    load_complex_from_checkpoint,
    load_config,
    load_model_config,
)
from ccg_agents.allocator import AllocatorMLP

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


TripleId = Tuple[int, int, int]


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train_allocator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _plot_allocator_loss(
    history: Dict[str, List[float]],
    save_dir: Path,
    dataset_name: str,
) -> Optional[Path]:
    """绘制 Allocator 训练损失曲线并保存为 PNG（英文标签避免乱码）。"""
    if not history.get("train_loss"):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], color="C0", linewidth=1.5, label="Train Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Allocator Training Loss - {dataset_name}", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    save_path = save_dir / f"allocator_{dataset_name.lower()}_train_loss.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


class AllocatorDataset(Dataset):
    """
    (特征向量 x, 伪标签 M_label) 的数据集。
    """

    def __init__(self, features: List[List[float]], labels: List[List[float]]) -> None:
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def train_allocator_for_dataset(
    dataset_name: str,
    max_triples: Optional[int] = None,
    epochs: int | None = None,
    seed: int = 42,
    emb_model: str = "transe",
    fusion_alpha: float = 0.5,
    ckpt_suffix: Optional[str] = None,
) -> None:
    """
    针对某个数据集训练 Allocator：
    1. 加载 TransE（及可选 ComplEx）、TypeConstraints、图和训练三元组；
    2. 按 emb_model（transe/complex/fusion）为每条三元组构造特征向量 x；
    3. 生成伪标签 M_label；
    4. 用 KLDivLoss 训练 AllocatorMLP；默认保存 {dataset}_allocator.pt，若 ckpt_suffix 非空则为 {dataset}_allocator_{suffix}.pt。
    """
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
    base = Path(__file__).resolve().parent.parent / output_dir.lstrip("./")
    base.mkdir(parents=True, exist_ok=True)
    alloc_dir = base / "allocator"
    alloc_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"train_allocator_{dataset_name.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = _setup_logger(log_file)
    log.info("=" * 60)
    log.info(f"Allocator 训练开始: dataset={dataset_name}, device={device}")
    dataset_name_lower = dataset_name.lower()
    suf = (ckpt_suffix or "").strip()
    suf_safe = "".join(c for c in suf if c.isalnum() or c in "._-")[:80] if suf else ""
    ckpt_stem = f"{dataset_name_lower}_allocator_{suf_safe}" if suf_safe else f"{dataset_name_lower}_allocator"
    log.info(
        f"  emb_model={emb_model}, fusion_alpha={fusion_alpha}, hidden_dims={hidden_dims}, lr={lr}, epochs={epochs}, max_triples={max_triples}"
    )
    log.info(f"  保存权重: {ckpt_stem}.pt" + ("" if suf_safe else "（默认名；可加 --ckpt_suffix 区分多组实验）"))
    log.info(f"  日志文件: {log_file}")
    log.info("=" * 60)

    # 1) 加载数据集
    if dataset_name_lower not in cfg or not isinstance(cfg[dataset_name_lower], dict):
        raise KeyError(f"配置中未找到数据集 {dataset_name}，可选项: fb15k, fb15k237, wn18rr, nell995")
    ds_cfg = cfg[dataset_name_lower]
    ds = KGDataset.load_from_files(
        ds_cfg["train_path"],
        ds_cfg["valid_path"],
        ds_cfg["test_path"],
    )
    train_triples: List[TripleId] = ds.train_triples

    if max_triples is not None and max_triples < len(train_triples):
        train_triples = random.sample(train_triples, max_triples)
        log.info(f"  采样训练三元组子集: {len(train_triples)} 条（共 {len(ds.train_triples)} 条，max_triples={max_triples}）")
    else:
        log.info(f"  使用全部训练三元组: {len(train_triples)} 条")

    # 2) 加载 TransE（及可选 ComplEx）与构建图、类型约束
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

    # 3) 两阶段动态打标：先收集特征算分布，再按相对阈值生成平衡伪标签
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

    # fea 索引: [0: raw_score, 1: S1, 2: support_paths, 3: is_func, 4: freq_h, 5: freq_t]（已移除 type_conflict）
    # 说明：本实验阶段暂不使用 is_func 参与打标判定
    import numpy as np
    features_np = np.array(raw_features, dtype=np.float64)
    s1_scores = features_np[:, 1]
    paths_counts = features_np[:, 2]
    # 用于打标的统计量（特征已是 log 空间：S1=log1p(100*s1), paths=log1p(raw_paths)）
    s1_median = np.median(s1_scores)
    # 结构：S1 低于 80% 分位（与模块 docstring 一致）
    s1_low_threshold = np.percentile(s1_scores, 50)
    paths_median = np.median(paths_counts)
    # 结构：路径偏少判定（相对 paths_median）
    paths_struct_threshold = paths_median * 0.5
    # 逻辑：仅用路径阈值（暂时注释掉 is_func 条件）
    paths_logic_min = paths_median * 1.09
    # is_func_logic_threshold = 0.50
    log.info(
        f"  统计结果：S1 中位数: {s1_median:.4f}, S1 低阈值(80%分位): {s1_low_threshold:.4f}, "
        f"路径中位数: {paths_median:.4f}, 结构路径阈值: {paths_struct_threshold:.4f}, 逻辑路径下限: {paths_logic_min:.4f}"
    )

    features: List[List[float]] = []
    labels: List[List[float]] = []
    for fea in raw_features:
        s1, support_paths = fea[1], fea[2]
        # is_func = fea[3]  # 暂时不参与逻辑类判定

        # 1) 明显结构型：路径偏少 & 结构置信度低于阈值
        if support_paths < paths_struct_threshold and s1 < s1_low_threshold:
            label = [0.4, 0.3, 0.3]   # 结构(S)

        # 2) 逻辑型：仅用路径阈值（is_func 条件已注释）
        elif support_paths >= paths_logic_min:
            label = [0.3, 0.3, 0.4]   # 逻辑(L)

        # 3) 其余：事实型
        else:
            label = [0.3, 0.4, 0.3]   # 事实(F)

        features.append(fea)
        labels.append(label)

    in_dim = len(features[0]) if features else 0
    if in_dim == 0:
        log.error("特征为空，无法训练 Allocator。")
        return

    # --- 统计伪标签分布（动态调整后）---
    # 软标签 [0.4,0.3,0.3] 等任一分量均 ≤0.5，不能用 >0.5 判定；按 argmax 对应主类别
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

    # 3.5) 特征标准化：按列做 zero-mean unit-variance，避免 raw_score/support_paths 等量纲压制
    F = np.array(features, dtype=np.float64)
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    std[std < 1e-8] = 1.0
    F = (F - mean) / std
    features = F.tolist()
    # 保存 mean/std 到 checkpoint，推理时复用
    feat_mean_std = {"mean": mean.tolist(), "std": std.tolist()}

    # 4) 构造 Dataset / DataLoader 并训练 MLP
    alloc_ds = AllocatorDataset(features, labels)
    loader = DataLoader(alloc_ds, batch_size=256, shuffle=True, num_workers=0)

    # 使用 model.yaml 中 allocator.hidden_dims 定义的多层结构，如 [128, 64, 32]
    if isinstance(hidden_dims, (list, tuple)):
        h_dims = [int(x) for x in hidden_dims]
    else:
        h_dims = [int(hidden_dims)]
    alloc_model = AllocatorMLP(in_dim=in_dim, hidden_dims=h_dims).to(device)
    optimizer = torch.optim.Adam(alloc_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    # AllocatorMLP 输出为 log_softmax（见 ccg_agents/allocator.py），标签为互斥软分布 [m_s, m_f, m_l]。
    # 因此应使用 KLDivLoss（输入 log-prob，target 为概率分布），而不是 BCELoss（要求输入在 [0,1]）。
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
            pred_logp = alloc_model(x)  # shape (N,3), log-probabilities
            # 保险：保证 target 是合法分布（防止数值误差/脏数据导致 sum!=1）
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

    # 4.5) 保存训练曲线并绘图
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
        log.info("  未安装 matplotlib，跳过绘图；可读取 history JSON 自行绘图。")

    # 5) 保存 Allocator 模型（含 emb_model / fusion_alpha 供推理一致）
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
    parser = argparse.ArgumentParser(description="训练任务分配智能体 Allocator")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fb15k", "fb15k237", "wn18rr", "nell995"],
        help="数据集名称",
    )
    parser.add_argument(
        "--emb_model",
        type=str,
        default="transe",
        choices=["transe", "complex", "fusion"],
        help="构造特征时使用的嵌入：transe / complex / fusion（TransE 与 ComplEx 加权），默认 transe",
    )
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.5,
        help="仅当 emb_model=fusion 时有效：S1 = alpha*S_transe + (1-alpha)*S_complex，默认 0.5",
    )
    parser.add_argument(
        "--max_triples",
        type=int,
        default=None,
        metavar="N",
        help="用于训练的最大三元组数，不设则使用全部训练集；设则采样 N 条（主要耗时在特征构造，MLP 训练较快）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数，不填则用 model.yaml 里 allocator.epochs（默认 50）",
    )
    parser.add_argument(
        "--ckpt_suffix",
        type=str,
        default=None,
        metavar="SUFFIX",
        help=(
            "权重文件名后缀：保存为 outputs/allocator/{数据集}_allocator_{SUFFIX}.pt；"
            "不设则仍为 {数据集}_allocator.pt（多组伪标规则时不要同名覆盖）"
        ),
    )
    args = parser.parse_args()

    train_allocator_for_dataset(
        args.dataset,
        max_triples=args.max_triples,
        epochs=args.epochs,
        emb_model=args.emb_model,
        fusion_alpha=args.fusion_alpha,
        ckpt_suffix=args.ckpt_suffix,
    )


if __name__ == "__main__":
    main()

