"""
在指定知识图谱数据集（如 FB15K-237）上训练子图级 RGCN 编码器：

- 使用已训练好的 TransE 实体/关系嵌入作为节点/关系初始表示；
- 对真实三元组与通过随机替换头/尾实体生成的噪声三元组构造 k-hop 子图；
- 通过 RelationalSubgraphEncoder + SubgraphScorer 对「子图 + 三元组」训练：
  - **bce**（默认）：BCEWithLogits，真=1、噪=0，即学 P(true|x)；
  - **ranking**：对 (干净, 注噪) 成对采样，MarginRankingLoss 约束 score(真) > score(假)+margin，
    更贴近「找异常 / 排序」评测（低分更可疑）。
- 训练完成后，将 GCN 编码器与打分器的参数保存到 outputs/gcn/{dataset}_gcn.pt。

注意：
- 这里的「GCN」具体实现是 ccg_models.gcn_encoder.RelationalGraphConvLayer（RGCN 风格）堆叠而成，
  再接 GraphAttentionLayer，因此本质上是 RGCN 子图编码器。
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing
import random
import sys
from pathlib import Path
from typing import Any, List, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset as SimpleKGDataset
from ccg_data.noise_injector import inject_noise, inject_noise_with_pairs
from ccg_models.gcn_encoder import (
    build_adj_from_triples,
    extract_k_hop_subgraph_from_adj,
    AdjacencyBuilder,
    RelationalSubgraphEncoder,
    encode_triple_subgraph,
    SubgraphScorer,
)
from ccg_pipeline.train_embeddings import (
    load_config,
    load_model_config,
    load_transe_from_checkpoint,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback
    def tqdm(iterable, **kwargs):
        return iterable


TripleId = Tuple[int, int, int]

# 多进程 worker：单样本 BCE — (adj, triples_labels, k, max_nodes)
_ChunkArg = Tuple[Any, List[Tuple[TripleId, int]], int, int]
# 多进程 worker：pairwise — (adj, pairs, k, max_nodes)
_PairChunkArg = Tuple[Any, List[Tuple[TripleId, TripleId]], int, int]


def _worker_extract_chunk(args: _ChunkArg) -> List[Tuple[List[int], List[Tuple[int, int, int]], TripleId, int]]:
    """在子进程中为一批 (triple, y) 提取 k-hop 子图，返回有效 (sub_nodes, sub_edges, triple, y)。"""
    from ccg_models.gcn_encoder import extract_k_hop_subgraph_from_adj

    adj, triples_labels, k, max_nodes = args
    out: List[Tuple[List[int], List[Tuple[int, int, int]], TripleId, int]] = []
    for (triple, y) in triples_labels:
        sub_nodes, sub_edges = extract_k_hop_subgraph_from_adj(adj, triple, k=k, max_nodes=max_nodes)
        if len(sub_nodes) >= 2 and len(sub_edges) > 0:
            out.append((sub_nodes, sub_edges, triple, y))
    return out


def _worker_extract_pair_chunk(
    args: _PairChunkArg,
) -> List[
    Tuple[
        List[int],
        List[Tuple[int, int, int]],
        TripleId,
        List[int],
        List[Tuple[int, int, int]],
        TripleId,
    ]
]:
    """为 (正例, 负例) 对分别提取 k-hop 子图；仅当两侧子图均有效时保留。"""
    from ccg_models.gcn_encoder import extract_k_hop_subgraph_from_adj

    adj, pairs, k, max_nodes = args
    out: List[
        Tuple[
            List[int],
            List[Tuple[int, int, int]],
            TripleId,
            List[int],
            List[Tuple[int, int, int]],
            TripleId,
        ]
    ] = []
    for t_pos, t_neg in pairs:
        sp, se = extract_k_hop_subgraph_from_adj(adj, t_pos, k=k, max_nodes=max_nodes)
        sn, se_n = extract_k_hop_subgraph_from_adj(adj, t_neg, k=k, max_nodes=max_nodes)
        if (
            len(sp) >= 2
            and len(se) > 0
            and len(sn) >= 2
            and len(se_n) > 0
        ):
            out.append((sp, se, t_pos, sn, se_n, t_neg))
    return out


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train_gcn_encoder")
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


def _plot_gcn_loss(
    history: List[float],
    save_path: Path,
) -> bool:
    """绘制 GCN/RGCN 训练损失曲线（若 matplotlib 不可用则返回 False）。"""
    if not history:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, history, color="C0", linewidth=1.6, label="Train Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("GCN/RGCN Subgraph Encoder Training Loss", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=1)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def train_gcn_for_dataset(
    dataset_name: str,
    max_triples: int | None = 50000,
    noise_ratio: float = 1.0,
    seed: int = 42,
    epochs: int | None = None,
    n_workers: int = 0,
    ckpt_suffix: str | None = None,
    log_file: str | None = None,
    loss_type: str = "bce",
    ranking_margin: float = 1.0,
) -> None:
    """
    在指定数据集（如 fb15k237）上训练 RGCN 子图编码器：

    1. 加载 TransE checkpoint，得到实体/关系嵌入（固定，不反向传播）；
    2. 从训练集中采样若干真实三元组，并注噪；
    3. 对每条（或成对）三元组构造 k-hop 子图，经 Encoder + Scorer 得到 logit；
    4. **bce**：BCEWithLogits，真=1、噪=0；**ranking**：(真, 假) 成对，MarginRankingLoss(logit_pos, logit_neg, +1)；
    5. 保存 encoder/scorer 参数到 outputs/gcn/{dataset}_gcn.pt。
    """
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    model_cfg = load_model_config()

    gcn_cfg = model_cfg.get("gcn", {})
    hidden_dim = int(gcn_cfg.get("hidden_dim", 64))
    # 将 gcn.num_layers 解释为子图提取的 k-hop 深度
    k_hop = int(gcn_cfg.get("num_layers", 2))
    max_subgraph_nodes = int(gcn_cfg.get("max_subgraph_nodes", 2000))
    if epochs is None:
        epochs = int(gcn_cfg.get("epochs", 10))

    output_dir = cfg.get("output_dir", "./outputs")
    base = _ROOT / output_dir.lstrip("./")
    gcn_dir = base / "gcn"
    gcn_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = base / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_file) if log_file else (log_dir / f"train_gcn_{dataset_name.lower()}_{ts}.log")
    logger = _setup_logger(log_path)

    logger.info("=" * 70)
    logger.info(f"GCN/RGCN 子图编码器训练开始: dataset={dataset_name}, device={device}")
    suf = (ckpt_suffix or "").strip()
    suf_safe = "".join(c for c in suf if c.isalnum() or c in "._-")[:80] if suf else ""
    gcn_stem = f"{dataset_name.lower()}_gcn_{suf_safe}" if suf_safe else f"{dataset_name.lower()}_gcn"

    lt = (loss_type or "bce").strip().lower()
    if lt not in ("bce", "ranking"):
        raise ValueError(f"loss_type 须为 bce 或 ranking，当前: {loss_type}")
    logger.info(
        f"  hidden_dim={hidden_dim}, k_hop={k_hop}, max_subgraph_nodes={max_subgraph_nodes}, epochs={epochs}, "
        f"max_triples={max_triples}, noise_ratio={noise_ratio}, loss_type={lt}, ranking_margin={ranking_margin}"
    )
    logger.info(f"  输出权重文件名: {gcn_stem}.pt" + ("" if suf_safe else "（默认名；可加 --ckpt_suffix 区分多组实验）"))
    logger.info(f"  日志文件: {log_path}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. 加载数据集与 TransE checkpoint（实体/关系嵌入）
    # ------------------------------------------------------------------
    ds_cfg = cfg.get(dataset_name.lower())
    if not isinstance(ds_cfg, dict) or "train_path" not in ds_cfg:
        raise KeyError(f"配置中未找到数据集 {dataset_name}，或缺少 train_path")

    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")

    dataset = SimpleKGDataset.load_from_files(str(train_path), str(valid_path), str(test_path))
    train_triples: List[TripleId] = dataset.train_triples
    num_entities = dataset.num_entities
    num_relations = dataset.num_relations

    logger.info(f"  训练三元组数: {len(train_triples)}, num_entities={num_entities}, num_relations={num_relations}")

    # TransE checkpoint 默认位置与 train_embeddings.py 一致
    embed_dir = base / "embeddings"
    ckpt_path = embed_dir / f"{dataset_name.lower()}_transE.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 TransE checkpoint: {ckpt_path}，请先运行 train_embeddings.py 训练 TransE")

    transe_model = load_transe_from_checkpoint(str(ckpt_path), device=device)
    transe_model.eval()
    entity_emb = transe_model.ent_emb.weight.detach()
    rel_emb = transe_model.rel_emb  # 直接作为 Embedding 模块传给 encode_triple_subgraph
    ent_dim = entity_emb.size(1)

    logger.info(f"  已加载 TransE checkpoint: {ckpt_path}, ent_dim={ent_dim}")

    # ------------------------------------------------------------------
    # 2. 采样训练集三元组（与注噪共用）
    # ------------------------------------------------------------------
    total_train = len(train_triples)
    if max_triples is not None and max_triples < total_train:
        train_triples = random.sample(train_triples, max_triples)
        logger.info(f"  采样训练三元组子集: {len(train_triples)} / {total_train} 条")

    # ------------------------------------------------------------------
    # 3. 构建 GCN 编码器与打分器（两种 loss 共用）
    # ------------------------------------------------------------------
    encoder = RelationalSubgraphEncoder(
        ent_dim=ent_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
    ).to(device)
    triple_vec_dim = hidden_dim * 2 + ent_dim
    scorer = SubgraphScorer(triple_vec_dim=triple_vec_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(scorer.parameters()),
        lr=float(model_cfg.get("embedding", {}).get("learning_rate", 0.001)),
    )
    builder = AdjacencyBuilder()
    logger.info("  预构建训练图邻接表...")
    adj = build_adj_from_triples(train_triples)
    logger.info("  邻接表已构建")

    loss_history: List[float] = []

    if lt == "ranking":
        # ---------- Pairwise：score(真) > score(假) + margin，对齐排序式异常检测 ----------
        clean_triples, noisy_triples, _, pos_neg_pairs = inject_noise_with_pairs(
            train_triples,
            num_entities=num_entities,
            noise_ratio=noise_ratio,
            mode="head_tail",
        )
        logger.info(
            f"  [ranking] 真实条数={len(clean_triples)}, 噪声条数={len(noisy_triples)}, "
            f"(正,负) 对数={len(pos_neg_pairs)}"
        )
        if not pos_neg_pairs:
            raise ValueError(
                "ranking 模式需要至少一对 (干净, 注噪)：请增大 noise_ratio 或检查注噪是否全部失败。"
            )

        criterion = torch.nn.MarginRankingLoss(margin=float(ranking_margin))
        n_pairs = len(pos_neg_pairs)
        n_procs = n_workers if n_workers > 0 else min(multiprocessing.cpu_count(), 16)
        n_procs = max(1, min(n_procs, n_pairs))
        chunk_size = (n_pairs + n_procs - 1) // n_procs
        pair_chunks: List[List[Tuple[TripleId, TripleId]]] = [
            pos_neg_pairs[i : i + chunk_size] for i in range(0, n_pairs, chunk_size)
        ]
        pair_chunk_args: List[_PairChunkArg] = [(adj, ch, k_hop, max_subgraph_nodes) for ch in pair_chunks]

        precomputed_pairs: List[
            Tuple[
                List[int],
                List[Tuple[int, int, int]],
                TripleId,
                List[int],
                List[Tuple[int, int, int]],
                TripleId,
            ]
        ] = []
        logger.info(f"  [ranking] 多进程预计算成对子图: n_workers={n_procs}, 对数={n_pairs}...")
        with multiprocessing.Pool(n_procs) as pool:
            for res in tqdm(
                pool.imap_unordered(_worker_extract_pair_chunk, pair_chunk_args),
                total=len(pair_chunk_args),
                desc="  预计算 pair 子图",
                unit="chunk",
                dynamic_ncols=True,
            ):
                precomputed_pairs.extend(res)

        n_valid = len(precomputed_pairs)
        logger.info(f"  [ranking] 两侧子图均有效的对数: {n_valid} / {n_pairs}")
        if n_valid == 0:
            raise RuntimeError("ranking：没有有效的成对子图，请检查 k_hop / max_subgraph_nodes。")

        epoch_pbar = tqdm(range(epochs), desc="Epoch", unit="epoch", leave=True)
        for epoch in epoch_pbar:
            encoder.train()
            scorer.train()
            total_loss = 0.0
            n_used = 0
            order = precomputed_pairs.copy()
            random.shuffle(order)

            prog = tqdm(
                order,
                total=len(order),
                desc=f"  Epoch {epoch+1}/{epochs} [ranking]",
                unit="pair",
                leave=False,
                dynamic_ncols=True,
            )
            for sp, se, t_pos, sn, se_n, t_neg in prog:
                optimizer.zero_grad(set_to_none=True)

                Xp, edge_index_p, edge_type_p, node2id_p = builder.build(sp, se, entity_emb.to(device))
                triple_vec_p = encode_triple_subgraph(
                    encoder=encoder,
                    X=Xp,
                    edge_index=edge_index_p,
                    edge_type=edge_type_p,
                    node2id=node2id_p,
                    triple=t_pos,
                    rel_emb=rel_emb,
                )
                logit_pos = scorer(triple_vec_p)

                Xn, edge_index_n, edge_type_n, node2id_n = builder.build(sn, se_n, entity_emb.to(device))
                triple_vec_n = encode_triple_subgraph(
                    encoder=encoder,
                    X=Xn,
                    edge_index=edge_index_n,
                    edge_type=edge_type_n,
                    node2id=node2id_n,
                    triple=t_neg,
                    rel_emb=rel_emb,
                )
                logit_neg = scorer(triple_vec_n)

                lp = logit_pos.reshape(1)
                ln = logit_neg.reshape(1)
                loss = criterion(lp, ln, torch.ones(1, device=device, dtype=torch.float32))
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                n_used += 1
                if n_used > 0 and getattr(prog, "set_postfix", None) is not None:
                    prog.set_postfix(loss=f"{total_loss / n_used:.4f}", n_used=n_used)

            avg_loss = total_loss / max(n_used, 1)
            loss_history.append(float(avg_loss))
            epoch_pbar.set_postfix(avg_loss=f"{avg_loss:.4f}", n_used=n_used)
            logger.info(f"epoch {epoch+1}/{epochs}  [ranking] avg_loss={avg_loss:.4f}  有效对数={n_used}")

    else:
        # ---------- BCE：P(true|x) ----------
        clean_triples, noisy_triples, _ = inject_noise(
            train_triples,
            num_entities=num_entities,
            noise_ratio=noise_ratio,
            mode="head_tail",
        )

        all_triples: List[TripleId] = []
        labels: List[int] = []
        all_triples.extend(clean_triples)
        labels.extend([1] * len(clean_triples))
        all_triples.extend(noisy_triples)
        labels.extend([0] * len(noisy_triples))

        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        all_triples = [all_triples[i] for i in indices]
        labels = [labels[i] for i in indices]

        logger.info(f"  [bce] 真实样本数: {len(clean_triples)}, 噪声样本数: {len(noisy_triples)}, 训练总样本: {len(all_triples)}")

        criterion = torch.nn.BCEWithLogitsLoss()

        triples_with_labels = list(zip(all_triples, labels))
        n_total = len(triples_with_labels)
        n_procs = n_workers if n_workers > 0 else min(multiprocessing.cpu_count(), 16)
        n_procs = max(1, min(n_procs, n_total))

        logger.info(f"  [bce] 多进程预计算子图: n_workers={n_procs}, 样本数={n_total}...")
        chunk_size = (n_total + n_procs - 1) // n_procs
        chunks: List[List[Tuple[TripleId, int]]] = [
            triples_with_labels[i : i + chunk_size]
            for i in range(0, n_total, chunk_size)
        ]
        chunk_args: List[_ChunkArg] = [(adj, chunk, k_hop, max_subgraph_nodes) for chunk in chunks]

        precomputed_subgraphs: List[Tuple[List[int], List[Tuple[int, int, int]], TripleId, int]] = []
        with multiprocessing.Pool(n_procs) as pool:
            for res in tqdm(
                pool.imap_unordered(_worker_extract_chunk, chunk_args),
                total=len(chunk_args),
                desc="  预计算子图",
                unit="chunk",
                dynamic_ncols=True,
            ):
                precomputed_subgraphs.extend(res)

        n_valid = len(precomputed_subgraphs)
        logger.info(f"  [bce] 预计算子图完成: 有效 {n_valid} / {n_total}（训练时每轮遍历 {n_valid} 条）")

        epoch_pbar = tqdm(range(epochs), desc="Epoch", unit="epoch", leave=True)
        for epoch in epoch_pbar:
            encoder.train()
            scorer.train()
            total_loss = 0.0
            n_used = 0

            prog = tqdm(
                precomputed_subgraphs,
                total=len(precomputed_subgraphs),
                desc=f"  Epoch {epoch+1}/{epochs} [bce]",
                unit="triple",
                leave=False,
                dynamic_ncols=True,
            )
            for sub_nodes, sub_edges, triple, y in prog:
                X, edge_index, edge_type, node2id = builder.build(sub_nodes, sub_edges, entity_emb.to(device))

                optimizer.zero_grad(set_to_none=True)
                triple_vec = encode_triple_subgraph(
                    encoder=encoder,
                    X=X,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    node2id=node2id,
                    triple=triple,
                    rel_emb=rel_emb,
                )
                logit = scorer(triple_vec)
                target = torch.tensor([float(y)], dtype=torch.float32, device=device)
                loss = criterion(logit.unsqueeze(0), target)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                n_used += 1

                if n_used > 0 and getattr(prog, "set_postfix", None) is not None:
                    prog.set_postfix(loss=f"{total_loss / n_used:.4f}", n_used=n_used)

            avg_loss = total_loss / max(n_used, 1)
            loss_history.append(float(avg_loss))
            epoch_pbar.set_postfix(avg_loss=f"{avg_loss:.4f}", n_used=n_used)
            logger.info(f"epoch {epoch+1}/{epochs}  [bce] avg_loss={avg_loss:.4f}  有效样本数={n_used}")

    # ------------------------------------------------------------------
    # 5. 保存损失历史与 checkpoint
    # ------------------------------------------------------------------
    history_path = log_dir / f"{gcn_stem}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": loss_history,
                "loss_type": lt,
                "ranking_margin": float(ranking_margin) if lt == "ranking" else None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"训练历史已保存: {history_path}")

    plot_path = plot_dir / f"{gcn_stem}_train_loss.png"
    if _plot_gcn_loss(loss_history, plot_path):
        logger.info(f"训练损失曲线已保存: {plot_path}")
    else:
        logger.info("未安装 matplotlib，跳过绘图；可使用 history.json 自行绘制。")

    ckpt_path_gcn = gcn_dir / f"{gcn_stem}.pt"
    state = {
        "encoder_state": encoder.state_dict(),
        "scorer_state": scorer.state_dict(),
        "ent_dim": ent_dim,
        "hidden_dim": hidden_dim,
        "num_relations": num_relations,
        "k_hop": k_hop,
        "loss_type": lt,
        "ranking_margin": float(ranking_margin) if lt == "ranking" else None,
    }
    torch.save(state, ckpt_path_gcn)
    logger.info(f"GCN/RGCN 子图编码器已保存到 {ckpt_path_gcn}")
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="在指定数据集上训练 RGCN 子图编码器（GCN）")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fb15k237",
        help="数据集名称，对应 ccg_config/datasets.yaml 中的键，如 fb15k / fb15k237 / wn18rr / nell995",
    )
    parser.add_argument(
        "--max_triples",
        type=int,
        default=50000,
        help="用于训练的真实三元组最大数量（同时生成同量级噪声）；设为 0 或负数表示使用全部训练三元组",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=1.0,
        help="噪声比例，noise_num = int(len(triples) * noise_ratio)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="GCN 训练轮数；不传则用 ccg_config/model.yaml 中 gcn.epochs（默认 30）。每轮遍历全部真实+噪声样本。",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="预计算子图时的进程数；0 表示 min(cpu_count(), 16)。",
    )
    parser.add_argument(
        "--ckpt_suffix",
        type=str,
        default=None,
        metavar="SUFFIX",
        help=(
            "保存后缀：输出 outputs/gcn/{dataset}_gcn_{SUFFIX}.pt；"
            "并同步保存 history/log/plot 同名文件。"
        ),
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help=(
            "训练日志文件路径；不传则自动写 outputs/logs/train_gcn_<dataset>_<时间戳>.log。"
        ),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce",
        choices=["bce", "ranking"],
        help="bce=二分类 BCE（默认）；ranking=成对 MarginRanking，使 score(真)>score(假)，更贴近排序式异常检测。",
    )
    parser.add_argument(
        "--ranking_margin",
        type=float,
        default=1.0,
        help="仅 loss_type=ranking 时有效：MarginRankingLoss 的 margin（默认 1.0）。",
    )
    args = parser.parse_args()

    max_triples = args.max_triples
    if max_triples is not None and max_triples <= 0:
        max_triples = None

    train_gcn_for_dataset(
        dataset_name=args.dataset,
        max_triples=max_triples,
        noise_ratio=args.noise_ratio,
        epochs=args.epochs,
        n_workers=args.n_workers,
        ckpt_suffix=args.ckpt_suffix,
        log_file=args.log_file,
        loss_type=args.loss_type,
        ranking_margin=args.ranking_margin,
    )


if __name__ == "__main__":
    main()

