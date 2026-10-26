"""
从 ccg_config 读取数据与模型配置，加载 KGDataset，训练 TransE / ComplEx，并提供结构置信度接口。

- 读取 datasets.yaml / model.yaml
- KGDataset 加载 fb15k / wn18rr / nell995
- TransE 训练：DataLoader + 负采样 + margin ranking loss，定期保存 checkpoint
- ComplEx 训练：同流程，loss = relu(margin - pos_score + neg_score)，保存到 {dataset}_complEx.pt
- compute_struct_confidence(triple, model, device) -> TransE 的 S1 = sigmoid(-score)
- ComplEx 结构置信度：sigmoid(score)，见 ccg_models.complex.complex_struct_confidence
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import os
import random
import logging
import datetime

import yaml
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# 保证从 CC_gra 根目录可导入 ccg_models
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from ccg_models.transE import TransEModel
from ccg_models.complex import ComplExModel, complex_struct_confidence


Triple = Tuple[str, str, str]


class KGDataset:
    """
    把 train/valid/test 三个「三元组文本文件」读进来，统一给实体和关系编 id，
    并得到 ID 形式的三元组列表，供 TransE 等模型使用。

    - 每行格式：head<TAB>relation<TAB>tail（也兼容空格分隔）。
    - 属性：entity2id / relation2id（字符串 -> id）、id2entity / id2relation（反查）、
      train_triples / valid_triples / test_triples（List[(h_id, r_id, t_id)]）。
    - 统计：num_entities、num_relations。
    """

    def __init__(self, train_path: str, valid_path: str = None, test_path: str = None) -> None:
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}

        self.train_triples: List[Tuple[int, int, int]] = []
        self.valid_triples: List[Tuple[int, int, int]] = []
        self.test_triples: List[Tuple[int, int, int]] = []

        self._load_all()

    def _read_triple_file(self, path: str) -> List[Triple]:
        triples: List[Triple] = []
        if path is None:
            return triples
        if not os.path.exists(path):
            raise FileNotFoundError(f"Triple file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    # 兼容空格分隔
                    parts = line.split()
                    if len(parts) != 3:
                        continue
                h, r, t = parts
                triples.append((h, r, t))
        return triples

    def _build_vocab(self, triples: List[Triple]) -> None:
        for h, r, t in triples:
            if h not in self.entity2id:
                idx = len(self.entity2id)
                self.entity2id[h] = idx
                self.id2entity[idx] = h
            if t not in self.entity2id:
                idx = len(self.entity2id)
                self.entity2id[t] = idx
                self.id2entity[idx] = t
            if r not in self.relation2id:
                idx = len(self.relation2id)
                self.relation2id[r] = idx
                self.id2relation[idx] = r

    def _map_triples_to_ids(self, triples: List[Triple]) -> List[Tuple[int, int, int]]:
        mapped: List[Tuple[int, int, int]] = []
        for h, r, t in triples:
            h_id = self.entity2id[h]
            t_id = self.entity2id[t]
            r_id = self.relation2id[r]
            mapped.append((h_id, r_id, t_id))
        return mapped

    def _load_all(self) -> None:
        all_triples: List[Triple] = []
        train_triples = self._read_triple_file(self.train_path)
        all_triples.extend(train_triples)

        valid_triples: List[Triple] = []
        test_triples: List[Triple] = []

        if self.valid_path:
            valid_triples = self._read_triple_file(self.valid_path)
            all_triples.extend(valid_triples)
        if self.test_path:
            test_triples = self._read_triple_file(self.test_path)
            all_triples.extend(test_triples)

        self._build_vocab(all_triples)

        self.train_triples = self._map_triples_to_ids(train_triples)
        if valid_triples:
            self.valid_triples = self._map_triples_to_ids(valid_triples)
        if test_triples:
            self.test_triples = self._map_triples_to_ids(test_triples)

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)


def load_config() -> Dict[str, Any]:
    """
    读取 ccg_config/datasets.yaml 配置。
    """
    config_path = _ROOT / "ccg_config" / "datasets.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model_config() -> Dict[str, Any]:
    """
    读取 ccg_config/model.yaml 中的模型超参（如 embedding、gcn、allocator 等）。
    """
    config_path = _ROOT / "ccg_config" / "model.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_dataset_by_name(cfg: Dict[str, Any], name: str, base_dir: Optional[Path] = None) -> KGDataset:
    """
    根据数据集名称（fb15k / wn18rr / nell995）从配置中读取路径并构造 KGDataset。
    base_dir: 配置中路径的相对根目录，默认 _ROOT（CC_gra）。
    """
    name = name.lower()
    if name not in cfg:
        raise KeyError(f"未知数据集名称: {name}，可选项: {list(k for k in cfg.keys() if k not in ['noise_levels', 'output_dir'])}")
    base = base_dir if base_dir is not None else _ROOT
    ds_cfg = cfg[name]
    train_path = str((base / ds_cfg["train_path"].lstrip("./")).resolve())
    valid_path = str((base / ds_cfg["valid_path"].lstrip("./")).resolve())
    test_path = str((base / ds_cfg["test_path"].lstrip("./")).resolve())
    return KGDataset(train_path, valid_path, test_path)

# ---------------------------------------------------------------------------
# TransE 训练：负采样、DataLoader、margin loss、checkpoint、结构置信度
# ---------------------------------------------------------------------------


class TriplesDataset(Dataset):
    """
    将 (h_id, r_id, t_id) 列表封装为 PyTorch Dataset，每次返回一个三元组的三个 id 张量。
    """

    def __init__(self, triples: List[Tuple[int, int, int]]) -> None:
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, r, t = self.triples[idx]
        return torch.tensor(h, dtype=torch.long), torch.tensor(r, dtype=torch.long), torch.tensor(t, dtype=torch.long)


def negative_sample(
    pos_h: torch.Tensor,
    pos_r: torch.Tensor,
    pos_t: torch.Tensor,
    num_entities: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对每个正样本三元组生成一个负样本：以 50% 概率随机替换头实体，否则替换尾实体。
    """
    device = pos_h.device
    N = pos_h.size(0)
    neg_h = pos_h.clone()
    neg_t = pos_t.clone()
    replace_head = torch.rand(N, device=device) < 0.5
    random_ent = torch.randint(0, num_entities, (N,), device=device, dtype=pos_h.dtype)
    neg_h[replace_head] = random_ent[replace_head]
    neg_t[~replace_head] = random_ent[~replace_head]
    return neg_h, pos_r, neg_t


def _setup_train_logger(log_file: Path) -> logging.Logger:
    """
    创建训练用 logger：同时输出到控制台和日志文件。
    日志文件路径例如：outputs/logs/train_fb15k_20250104_123456.log
    """
    logger = logging.getLogger("train_embeddings")
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


def _plot_embedding_train_history(
    history: Dict[str, List[float]],
    save_dir: Path,
    dataset_name: str,
    model_name: str = "transe",
) -> Optional[Path]:
    """
    通用：绘制嵌入模型（TransE / ComplEx 等）训练曲线并保存为 PNG。
    history 须包含 "train_loss"。
    model_name: 如 "transe" / "complex"，用于文件名与标题。
    返回保存路径；若 matplotlib 不可用则返回 None。
    """
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
    title_name = model_name.upper() if model_name.lower() in ("transe", "complex") else model_name
    ax.set_title(f"{title_name} training loss — {dataset_name}", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    save_path = save_dir / f"{model_name.lower()}_{dataset_name.lower()}_train_loss.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def train_transe_for_dataset(
    dataset_name: str,
    seed: Optional[int] = 42,
    device: Optional[torch.device] = None,
    epochs_override: Optional[int] = None,
) -> Tuple[TransEModel, KGDataset]:
    """
    对指定数据集训练 TransE，并按配置保存 checkpoint。

    每 10 个 epoch（以及第 1 个 epoch）保存一次 checkpoint 到
    {output_dir}/embeddings/{dataset_name}_transE.pt；
    训练过程同时写入 {output_dir}/logs/train_{dataset_name}_{时间戳}.log。
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = load_config()
    model_cfg_all = load_model_config()
    emb_cfg = model_cfg_all.get("embedding", {})
    dim = int(emb_cfg.get("dim", 100))
    margin = float(emb_cfg.get("margin", 1.0))
    lr = float(emb_cfg.get("learning_rate", 0.001))
    batch_size = int(emb_cfg.get("batch_size", 1024))
    epochs = int(emb_cfg.get("epochs", 200))
    if epochs_override is not None:
        epochs = epochs_override
    neg_size = int(emb_cfg.get("negative_sample_size", 1))

    output_dir = ds_cfg.get("output_dir", "./outputs")
    base = _ROOT / output_dir.lstrip("./")
    base.mkdir(parents=True, exist_ok=True)
    embed_dir = base / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = embed_dir / f"{dataset_name.lower()}_transE.pt"
    log_file = log_dir / f"train_{dataset_name.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = _setup_train_logger(log_file)

    log.info("=" * 60)
    log.info(f"TransE 训练开始: dataset={dataset_name}, device={device}")
    log.info(f"  dim={dim}, margin={margin}, lr={lr}, batch_size={batch_size}, epochs={epochs}, neg_size={neg_size}")
    log.info(f"  日志文件: {log_file}")
    log.info("=" * 60)

    dataset = load_dataset_by_name(ds_cfg, dataset_name)
    num_entities = dataset.num_entities
    num_relations = dataset.num_relations
    train_triples = dataset.train_triples
    n_batches_per_epoch = (len(train_triples) + batch_size - 1) // batch_size

    model = TransEModel(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=dim,
        margin=margin,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    triples_ds = TriplesDataset(train_triples)
    dataloader = DataLoader(
        triples_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    history: Dict[str, List[float]] = {"train_loss": []}

    for epoch in tqdm(range(epochs), desc="Epoch", unit="epoch"):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar_batch = tqdm(
            dataloader,
            desc=f"  Epoch {epoch+1}/{epochs}",
            leave=False,
            unit="batch",
            total=n_batches_per_epoch,
        )
        for batch_h, batch_r, batch_t in pbar_batch:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)
            for _ in range(neg_size):
                neg_h, neg_r, neg_t = negative_sample(batch_h, batch_r, batch_t, num_entities)
                pos_score, neg_score = model(batch_h, batch_r, batch_t, neg_h, neg_r, neg_t)
                loss = torch.relu(margin + pos_score - neg_score).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            running_avg = total_loss / n_batches
            if getattr(pbar_batch, "set_postfix", None) is not None:
                pbar_batch.set_postfix(loss=f"{running_avg:.4f}")
        avg = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg)
        log.info(f"epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")
        if (epoch + 1) % 10 == 0 or epoch == 0:
            state = {
                "entity_emb": model.ent_emb.weight.data.cpu().clone(),
                "relation_emb": model.rel_emb.weight.data.cpu().clone(),
                "num_entities": num_entities,
                "num_relations": num_relations,
                "embedding_dim": dim,
                "margin": margin,
            }
            torch.save(state, ckpt_path)
            log.info(f"  checkpoint saved -> {ckpt_path}")

    # 保存训练曲线数据并绘图（通用）
    plot_dir = base / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / f"transe_{dataset_name.lower()}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    log.info(f"  训练曲线数据已保存: {history_path}")
    plot_path = _plot_embedding_train_history(history, plot_dir, dataset_name, model_name="transe")
    if plot_path is not None:
        log.info(f"  训练损失曲线图已保存: {plot_path}")
    else:
        log.info("  未安装 matplotlib，跳过绘图；可读取 history JSON 自行绘图。")

    # 训练结束：打印并记录简要结果与抽样 S1
    log.info("")
    log.info("=" * 60)
    log.info(f"TransE 训练完成: {dataset_name}")
    log.info(f"  device: {device}  |  checkpoint: {ckpt_path}")
    log.info(f"  最后一轮平均 loss: {avg:.4f}")
    log.info("  抽样 3 条训练三元组的结构置信度 S1 = sigmoid(-score):")
    model.eval()
    for i in [0, len(train_triples) // 2, len(train_triples) - 1]:
        if i >= len(train_triples):
            continue
        h, r, t = train_triples[i]
        s1 = compute_struct_confidence((h, r, t), model, device)
        log.info(f"    triple ({h}, {r}, {t})  ->  S1 = {s1:.4f}")
    log.info("=" * 60)
    return model, dataset


def train_complex_for_dataset(
    dataset_name: str,
    seed: Optional[int] = 42,
    device: Optional[torch.device] = None,
) -> Tuple[ComplExModel, KGDataset]:
    """
    对指定数据集训练 ComplEx，并按配置保存 checkpoint。

    使用与 TransE 相同的 embedding 配置（dim, margin, lr, batch_size, epochs, neg_size）。
    ComplEx：分数越高越合理，loss = relu(margin - pos_score + neg_score)。
    保存到 {output_dir}/embeddings/{dataset_name}_complEx.pt。
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = load_config()
    model_cfg_all = load_model_config()
    emb_cfg = model_cfg_all.get("embedding", {})
    # ComplEx 使用 complex_dim（默认 200），无则回退到 dim，避免与 TransE 同用 100 导致表现偏弱
    dim = int(emb_cfg.get("complex_dim", emb_cfg.get("dim", 100)))
    margin = float(emb_cfg.get("margin", 1.0))
    lr = float(emb_cfg.get("learning_rate", 0.001))
    batch_size = int(emb_cfg.get("batch_size", 1024))
    epochs = int(emb_cfg.get("epochs", 200))
    neg_size = int(emb_cfg.get("negative_sample_size", 1))

    output_dir = ds_cfg.get("output_dir", "./outputs")
    base = _ROOT / output_dir.lstrip("./")
    base.mkdir(parents=True, exist_ok=True)
    embed_dir = base / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = embed_dir / f"{dataset_name.lower()}_complEx.pt"
    log_file = log_dir / f"train_complex_{dataset_name.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = _setup_train_logger(log_file)

    log.info("=" * 60)
    log.info(f"ComplEx 训练开始: dataset={dataset_name}, device={device}")
    log.info(f"  dim={dim} (complex_dim), margin={margin}, lr={lr}, batch_size={batch_size}, epochs={epochs}, neg_size={neg_size}")
    log.info(f"  日志文件: {log_file}")
    log.info("=" * 60)

    dataset = load_dataset_by_name(ds_cfg, dataset_name)
    num_entities = dataset.num_entities
    num_relations = dataset.num_relations
    train_triples = dataset.train_triples
    n_batches_per_epoch = (len(train_triples) + batch_size - 1) // batch_size

    model = ComplExModel(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    triples_ds = TriplesDataset(train_triples)
    dataloader = DataLoader(
        triples_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    history: Dict[str, List[float]] = {"train_loss": []}

    for epoch in tqdm(range(epochs), desc="Epoch", unit="epoch"):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar_batch = tqdm(
            dataloader,
            desc=f"  Epoch {epoch+1}/{epochs}",
            leave=False,
            unit="batch",
            total=n_batches_per_epoch,
        )
        for batch_h, batch_r, batch_t in pbar_batch:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)
            for _ in range(neg_size):
                neg_h, neg_r, neg_t = negative_sample(batch_h, batch_r, batch_t, num_entities)
                pos_score, neg_score = model(batch_h, batch_r, batch_t, neg_h, neg_r, neg_t)
                # ComplEx: 希望 pos_score 高、neg_score 低，margin ranking
                loss = torch.relu(margin - pos_score + neg_score).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            running_avg = total_loss / n_batches
            if getattr(pbar_batch, "set_postfix", None) is not None:
                pbar_batch.set_postfix(loss=f"{running_avg:.4f}")
        avg = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg)
        log.info(f"epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")
        if (epoch + 1) % 10 == 0 or epoch == 0:
            state = {
                "ent_re": model.ent_re.weight.data.cpu().clone(),
                "ent_im": model.ent_im.weight.data.cpu().clone(),
                "rel_re": model.rel_re.weight.data.cpu().clone(),
                "rel_im": model.rel_im.weight.data.cpu().clone(),
                "num_entities": num_entities,
                "num_relations": num_relations,
                "embedding_dim": dim,
            }
            torch.save(state, ckpt_path)
            log.info(f"  checkpoint saved -> {ckpt_path}")

    # 保存训练曲线数据并绘图（与 TransE 共用同一套逻辑）
    plot_dir = base / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / f"complex_{dataset_name.lower()}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    log.info(f"  训练曲线数据已保存: {history_path}")
    plot_path = _plot_embedding_train_history(history, plot_dir, dataset_name, model_name="complex")
    if plot_path is not None:
        log.info(f"  训练损失曲线图已保存: {plot_path}")
    else:
        log.info("  未安装 matplotlib，跳过绘图；可读取 history JSON 自行绘图。")

    log.info("")
    log.info("=" * 60)
    log.info(f"ComplEx 训练完成: {dataset_name}")
    log.info(f"  device: {device}  |  checkpoint: {ckpt_path}")
    log.info(f"  最后一轮平均 loss: {avg:.4f}")
    log.info("  抽样 3 条训练三元组的结构置信度 S_complex = sigmoid(score):")
    model.eval()
    for i in [0, len(train_triples) // 2, len(train_triples) - 1]:
        if i >= len(train_triples):
            continue
        h, r, t = train_triples[i]
        with torch.no_grad():
            ht = torch.tensor([h], dtype=torch.long, device=device)
            rt = torch.tensor([r], dtype=torch.long, device=device)
            tt = torch.tensor([t], dtype=torch.long, device=device)
            sc = complex_struct_confidence(model.score_triple(ht, rt, tt)).item()
        log.info(f"    triple ({h}, {r}, {t})  ->  S_complex = {sc:.4f}")
    log.info("=" * 60)
    return model, dataset


def load_transe_from_checkpoint(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> TransEModel:
    """
    从 outputs/embeddings/{dataset_name}_transE.pt 加载 TransE 权重，返回可用于
    compute_struct_confidence 的模型。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    num_entities = state["num_entities"]
    num_relations = state["num_relations"]
    embedding_dim = state["embedding_dim"]
    margin = state.get("margin", 1.0)
    model = TransEModel(num_entities, num_relations, embedding_dim, margin=margin).to(device)
    model.ent_emb.weight.data = state["entity_emb"].to(device)
    model.rel_emb.weight.data = state["relation_emb"].to(device)
    return model


def load_complex_from_checkpoint(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> ComplExModel:
    """
    从 outputs/embeddings/{dataset_name}_complEx.pt 加载 ComplEx 权重。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    num_entities = state["num_entities"]
    num_relations = state["num_relations"]
    embedding_dim = state["embedding_dim"]
    model = ComplExModel(num_entities, num_relations, embedding_dim).to(device)
    model.ent_re.weight.data = state["ent_re"].to(device)
    model.ent_im.weight.data = state["ent_im"].to(device)
    model.rel_re.weight.data = state["rel_re"].to(device)
    model.rel_im.weight.data = state["rel_im"].to(device)
    return model


def compute_struct_confidence(
    triple: Tuple[int, int, int],
    model: TransEModel,
    device: torch.device,
) -> float:
    """
    对单条三元组 (h_id, r_id, t_id) 计算结构置信度 S1 = sigmoid(-f(h,r,t))。
    """
    h, r, t = triple
    h_t = torch.tensor([h], dtype=torch.long, device=device)
    r_t = torch.tensor([r], dtype=torch.long, device=device)
    t_t = torch.tensor([t], dtype=torch.long, device=device)
    with torch.no_grad():
        score = model.score_triple(h_t, r_t, t_t)
        s1 = torch.sigmoid(-score).item()
    return float(s1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="加载配置/训练 TransE 或 ComplEx/演示结构置信度")
    parser.add_argument("--train", type=str, default=None, help="训练数据集，如 fb15k / fb15k237 / wn18rr / nell995")
    parser.add_argument("--model", type=str, default="transe", choices=["transe", "complex"], help="训练哪个 embedding 模型，默认 transe")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数，不填则用 model.yaml 中的 embedding.epochs（默认 200）")
    parser.add_argument("--demo", action="store_true", help="仅打印三个数据集的统计信息")
    args = parser.parse_args()

    if args.train:
        if args.model == "complex":
            train_complex_for_dataset(args.train)
        else:
            train_transe_for_dataset(args.train, epochs_override=args.epochs)

