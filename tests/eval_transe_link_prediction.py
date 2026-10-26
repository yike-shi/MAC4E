"""
TransE 链接预测评估脚本：在测试集上计算 Hits@K / Precision@K / MRR。

- 对每个测试三元组 (h, r, t)，分别做 tail prediction（预测 t）和 head prediction（预测 h）：
  - tail：固定 (h, r)，遍历所有实体 e 作为候选 tail，计算 score(h,r,e)，看真实 t 的排序位置；
  - head：固定 (r, t)，遍历所有实体 e 作为候选 head，计算 score(e,r,t)，看真实 h 的排序位置。
- 对所有测试三元组统计：
  - MRR: Mean Reciprocal Rank
  - Hits@K / Precision@K：rank <= K 的比例（单一 ground truth 时两者数值相同）
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Sequence

import torch

# 确保以脚本方式运行时也能找到 ccg_data / ccg_pipeline 包
_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_pipeline.train_embeddings import load_transe_from_checkpoint, load_config

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


TripleId = Tuple[int, int, int]


def _build_all_triples_set(
    train: Sequence[TripleId],
    valid: Sequence[TripleId],
    test: Sequence[TripleId],
) -> set[TripleId]:
    """构建包含 train/valid/test 所有三元组的集合，用于 filtered 评估。"""
    all_set: set[TripleId] = set()
    all_set.update(train)
    all_set.update(valid)
    all_set.update(test)
    return all_set


def _ranks_for_side(
    model,
    device: torch.device,
    triples: Sequence[TripleId],
    num_entities: int,
    all_triples: set[TripleId] | None,
    side: str,
) -> List[int]:
    """
    计算给定 side（'tail' 或 'head'）下，每条测试三元组的 rank（1 为最好）。
    - tail：预测 t，固定 (h,r)，对所有 e 计算 score(h,r,e)
    - head：预测 h，固定 (r,t)，对所有 e 计算 score(e,r,t)
    """
    ranks: List[int] = []

    ent_ids = torch.arange(num_entities, dtype=torch.long, device=device)

    for h, r, t in tqdm(triples, desc=f"  Eval {side}", unit="triple"):
        if side == "tail":
            # 固定 (h,r)，候选 tail = 所有实体
            h_batch = torch.full_like(ent_ids, h)
            r_batch = torch.full_like(ent_ids, r)
            t_batch = ent_ids
        elif side == "head":
            # 固定 (r,t)，候选 head = 所有实体
            h_batch = ent_ids
            r_batch = torch.full_like(ent_ids, r)
            t_batch = torch.full_like(ent_ids, t)
        else:
            raise ValueError(f"未知 side: {side}")

        with torch.no_grad():
            scores = model.score_triple(h_batch, r_batch, t_batch)  # 越小越好

        # filtered setting：对除了真实三元组之外的 (h,r,*) 或 (*,r,t) 如果在图中存在，
        # 则把它们的分数设为一个很大的值，这样它们不会排在真实三元组前面。
        if all_triples is not None:
            if side == "tail":
                for e in range(num_entities):
                    if e == t:
                        continue
                    if (h, r, e) in all_triples:
                        scores[e] = 1e9
            else:  # head
                for e in range(num_entities):
                    if e == h:
                        continue
                    if (e, r, t) in all_triples:
                        scores[e] = 1e9

        # rank = 1 表示最好（分数最小）
        # argsort 得到索引从小到大排列
        sorted_indices = torch.argsort(scores, dim=0)  # (num_entities,)
        # 找到真实实体的排名位置
        target = t if side == "tail" else h
        # 转成 Python 列表后用 index 找位置（+1 因为从 1 开始计）
        rank = (sorted_indices == target).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)

    return ranks


def evaluate_link_prediction(
    dataset_name: str,
    ks: Sequence[int] = (1, 3, 10),
    max_queries: int | None = None,
    filtered: bool = False,
) -> Dict[str, float]:
    """
    在指定数据集上评估 TransE 链接预测性能（head + tail 平均）。

    指标：
    - MRR：mean(1/rank)
    - Hits@K / Precision@K：mean( [rank <= K] )
    """
    cfg = load_config()
    ds_cfg = cfg[dataset_name.lower()]

    # 加载数据
    ds = KGDataset.load_from_files(
        ds_cfg["train_path"],
        ds_cfg["valid_path"],
        ds_cfg["test_path"],
    )
    train_triples = ds.train_triples
    valid_triples = ds.valid_triples
    test_triples = ds.test_triples
    num_entities = ds.num_entities

    if max_queries is not None and max_queries < len(test_triples):
        test_triples = random.sample(test_triples, max_queries)

    # 加载模型
    output_dir = cfg.get("output_dir", "./outputs")
    ckpt_path = Path(output_dir) / "embeddings" / f"{dataset_name.lower()}_transE.pt"
    ckpt_path = (Path(__file__).resolve().parent.parent / ckpt_path).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_transe_from_checkpoint(str(ckpt_path), device=device)
    model.eval()

    # filtered 设置：包含 train/valid/test 的所有三元组
    all_triples = (
        _build_all_triples_set(train_triples, valid_triples, test_triples)
        if filtered
        else None
    )

    # 计算 head / tail 的 rank 列表
    tail_ranks = _ranks_for_side(model, device, test_triples, num_entities, all_triples, side="tail")
    head_ranks = _ranks_for_side(model, device, test_triples, num_entities, all_triples, side="head")

    def _metrics_from_ranks(ranks: List[int]) -> Dict[str, float]:
        n = len(ranks)
        mrr = sum(1.0 / r for r in ranks) / n
        out: Dict[str, float] = {"MRR": mrr}
        for k in ks:
            hits = sum(1 for r in ranks if r <= k) / n
            # 单一 ground truth 时，Hits@K 与 Precision@K 数值相同
            out[f"Hits@{k}"] = hits
            out[f"P@{k}"] = hits
        return out

    tail_metrics = _metrics_from_ranks(tail_ranks)
    head_metrics = _metrics_from_ranks(head_ranks)

    # head / tail 平均
    avg_metrics: Dict[str, float] = {}
    keys = set(tail_metrics.keys()) | set(head_metrics.keys())
    for k in keys:
        avg_metrics[k] = 0.5 * (tail_metrics.get(k, 0.0) + head_metrics.get(k, 0.0))

    # 打印结果
    setting = "filtered" if filtered else "raw"
    print("=" * 60)
    print(f"TransE 链接预测评估: dataset={dataset_name}, setting={setting}, device={device}")
    if max_queries is not None:
        print(f"  使用测试三元组子集: {len(test_triples)} / {len(ds.test_triples)}")
    print("- head prediction:")
    for k, v in sorted(head_metrics.items()):
        print(f"  {k:8s}: {v:.4f}")
    print("- tail prediction:")
    for k, v in sorted(tail_metrics.items()):
        print(f"  {k:8s}: {v:.4f}")
    print("- avg (head+tail)/2:")
    for k, v in sorted(avg_metrics.items()):
        print(f"  {k:8s}: {v:.4f}")
    print("=" * 60)

    return avg_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="TransE 链接预测评估 (MRR / Hits@K / Precision@K)")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fb15k", "fb15k237", "wn18rr", "nell995"],
        help="数据集名称（已支持 fb15k237）",
    )
    parser.add_argument(
        "--K",
        type=int,
        nargs="+",
        default=[1, 3, 10],
        help="评估的 K 列表，例如 --K 1 3 10",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="最多使用多少条测试三元组（不设则用全部）",
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="是否使用 filtered 评估（过滤其它真三元组）",
    )
    args = parser.parse_args()

    evaluate_link_prediction(
        dataset_name=args.dataset,
        ks=args.K,
        max_queries=args.max_queries,
        filtered=args.filtered,
    )


if __name__ == "__main__":
    main()

