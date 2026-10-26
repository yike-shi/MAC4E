"""
ComplEx 链接预测评估脚本：在测试集上计算 Hits@K / Precision@K / MRR。

- 与 eval_transe_link_prediction 流程一致，仅评分语义不同：ComplEx 分数越高越合理。
- 对每个测试三元组分别做 tail / head prediction，filtered 时将其他真三元组分数置为 -inf，
  再按分数降序排序得到 rank。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import torch

_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_pipeline.train_embeddings import load_complex_from_checkpoint, load_config

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
) -> set:
    """构建 train/valid/test 所有三元组集合，用于 filtered 评估。"""
    all_set = set()
    all_set.update(train)
    all_set.update(valid)
    all_set.update(test)
    return all_set


def _ranks_for_side(
    model,
    device: torch.device,
    triples: Sequence[TripleId],
    num_entities: int,
    all_triples: set | None,
    side: str,
) -> List[int]:
    """
    计算 tail 或 head 预测的 rank（1 为最好）。
    ComplEx：分数越高越合理，故降序排序；filtered 时其他真三元组分数置为 -inf。
    """
    ranks: List[int] = []
    ent_ids = torch.arange(num_entities, dtype=torch.long, device=device)

    for h, r, t in tqdm(triples, desc=f"  Eval {side}", unit="triple"):
        if side == "tail":
            h_batch = torch.full_like(ent_ids, h)
            r_batch = torch.full_like(ent_ids, r)
            t_batch = ent_ids
        elif side == "head":
            h_batch = ent_ids
            r_batch = torch.full_like(ent_ids, r)
            t_batch = torch.full_like(ent_ids, t)
        else:
            raise ValueError(f"未知 side: {side}")

        with torch.no_grad():
            scores = model.score_triple(h_batch, r_batch, t_batch)  # 越大越好

        # filtered：除当前真实三元组外的其它已存在三元组分数置为很小，避免排在真实答案前
        if all_triples is not None:
            if side == "tail":
                for e in range(num_entities):
                    if e == t:
                        continue
                    if (h, r, e) in all_triples:
                        scores[e] = -1e9
            else:
                for e in range(num_entities):
                    if e == h:
                        continue
                    if (e, r, t) in all_triples:
                        scores[e] = -1e9

        # 降序：分数越高排名越前，rank=1 为最好
        sorted_indices = torch.argsort(scores, dim=0, descending=True)
        target = t if side == "tail" else h
        pos = (sorted_indices == target).nonzero(as_tuple=True)[0][0].item()
        rank = pos + 1
        ranks.append(rank)

    return ranks


def evaluate_link_prediction(
    dataset_name: str,
    ks: Sequence[int] = (1, 3, 10),
    max_queries: int | None = None,
    filtered: bool = False,
) -> Dict[str, float]:
    """
    在指定数据集上评估 ComplEx 链接预测（head + tail 平均）。
    指标：MRR、Hits@K、P@K。
    """
    cfg = load_config()
    ds_cfg = cfg[dataset_name.lower()]

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

    output_dir = cfg.get("output_dir", "./outputs")
    ckpt_path = Path(output_dir) / "embeddings" / f"{dataset_name.lower()}_complEx.pt"
    ckpt_path = (_ROOT / ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"ComplEx checkpoint 不存在: {ckpt_path}，请先训练 ComplEx。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_complex_from_checkpoint(str(ckpt_path), device=device)
    model.eval()

    all_triples = (
        _build_all_triples_set(train_triples, valid_triples, test_triples)
        if filtered
        else None
    )

    tail_ranks = _ranks_for_side(model, device, test_triples, num_entities, all_triples, side="tail")
    head_ranks = _ranks_for_side(model, device, test_triples, num_entities, all_triples, side="head")

    def _metrics_from_ranks(ranks: List[int]) -> Dict[str, float]:
        n = len(ranks)
        mrr = sum(1.0 / r for r in ranks) / n
        out: Dict[str, float] = {"MRR": mrr}
        for k in ks:
            hits = sum(1 for r in ranks if r <= k) / n
            out[f"Hits@{k}"] = hits
            out[f"P@{k}"] = hits
        return out

    tail_metrics = _metrics_from_ranks(tail_ranks)
    head_metrics = _metrics_from_ranks(head_ranks)

    avg_metrics: Dict[str, float] = {}
    keys = set(tail_metrics.keys()) | set(head_metrics.keys())
    for k in keys:
        avg_metrics[k] = 0.5 * (tail_metrics.get(k, 0.0) + head_metrics.get(k, 0.0))

    setting = "filtered" if filtered else "raw"
    print("=" * 60)
    print(f"ComplEx 链接预测评估: dataset={dataset_name}, setting={setting}, device={device}")
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
    parser = argparse.ArgumentParser(description="ComplEx 链接预测评估 (MRR / Hits@K / P@K)")
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
        help="最多使用多少条测试三元组",
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="是否使用 filtered 评估",
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
