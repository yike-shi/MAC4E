#!/usr/bin/env python3
"""
评测 TransE 在噪声注入（错误检测）任务上的 Top-K 指标，并汇总成表格（CSV）。

需求：
- FB15k237：5% / 10% / 15% 噪声
- WN18RR：5% / 10% 噪声
- NELL-995：5% 噪声

收集指标（与 CAGED Top-K 计算方法保持一致）：
- 对每个 setting 计算：
  - Precision@K, Recall@K，其中 K = 1%/2%/3%/4%/5% × num_original_triples
  - 排序依据：anomaly_score = 1 - score（score 越高越“正确”，越低越“异常”）

重要：为保证与 CAGED 的注入与 K 口径一致，本脚本：
- 使用 CC_gra 的 KGDataset 读 train/valid/test 并建立 ent/rel id；
- 在同一 id 空间中复现 CAGED/dataset.py 的噪声注入逻辑；
- 用 TransE score 得到 “正确性分数” score = sigmoid(-f(h,r,t))（与 CAGED 约定一致）；
- Top-K 的 K 基于 num_original_triples（不是 原+注入 的总长度）。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_pipeline.train_embeddings import (  # noqa: E402
    load_config,
    load_dataset_by_name,
    load_transe_from_checkpoint,
)


TripleId = Tuple[int, int, int]


def compute_precision_recall_at_k_anomaly(
    labels_pos: List[int],
    scores: List[float],
    k_ratios: Tuple[float, ...],
    k_ref_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    与 CAGED/Our_TopK%_RankingList.py 一致（错误检测 Top-K）：
    - labels_pos: 1=正确三元组，0=异常/错误；
    - scores 越高越正确，anomaly_score = 1 - score 越高越可能为错误；
    - K = ratio * k_ref_size（通常为 num_original_triples），并截断到 n。
    """
    n = len(labels_pos)
    labels_anom = [1 - p for p in labels_pos]
    n_anom = sum(labels_anom)
    result: Dict[str, float] = {}
    if n == 0:
        for r in k_ratios:
            pct = int(r * 100)
            result[f"precision_at_{pct}"] = 0.0
            result[f"recall_at_{pct}"] = 0.0
        return result

    ref = int(k_ref_size) if k_ref_size is not None else n
    anomaly_scores = [1.0 - s for s in scores]
    order = sorted(range(n), key=lambda i: -anomaly_scores[i])
    for r in k_ratios:
        pct = int(r * 100)
        n_top = max(1, min(int(r * ref), n))
        top_indices = order[:n_top]
        n_anom_in_top = sum(1 for i in top_indices if labels_anom[i] == 1)
        precision_at_k = n_anom_in_top / n_top if n_top > 0 else 0.0
        recall_at_k = n_anom_in_top / n_anom if n_anom > 0 else 0.0
        result[f"precision_at_{pct}"] = float(precision_at_k)
        result[f"recall_at_{pct}"] = float(recall_at_k)
        result[f"k_{pct}"] = int(n_top)
    return result


def _generate_anomalous_triples_like_caged(
    pos_triples: Sequence[TripleId],
    num_entities: int,
    num_relations: int,
    triple_ori_set: set[TripleId],
) -> List[TripleId]:
    """复现 CAGED/dataset.py 的 generate_anomalous_triples。"""
    neg_triples: List[TripleId] = []
    for head, rel, tail in pos_triples:
        head_or_tail = random.randint(0, 2)
        if head_or_tail == 0:
            new_head = random.randint(0, num_entities - 1)
            new_relation = rel
            new_tail = tail
        elif head_or_tail == 1:
            new_head = head
            new_relation = random.randint(0, num_relations - 1)
            new_tail = tail
        else:
            new_head = head
            new_relation = rel
            new_tail = random.randint(0, num_entities - 1)

        anomaly = (new_head, new_relation, new_tail)
        while anomaly in triple_ori_set:
            if head_or_tail == 0:
                new_head = random.randint(0, num_entities - 1)
                new_relation = rel
                new_tail = tail
            elif head_or_tail == 1:
                new_head = head
                new_relation = random.randint(0, num_relations - 1)
                new_tail = tail
            else:
                new_head = head
                new_relation = rel
                new_tail = random.randint(0, num_entities - 1)
            anomaly = (new_head, new_relation, new_tail)
        neg_triples.append(anomaly)
    return neg_triples


def _generate_anomalous_triples_2_like_caged(
    num_anomaly: int,
    num_entities: int,
    num_relations: int,
    triple_ori_set: set[TripleId],
) -> List[TripleId]:
    """复现 CAGED/dataset.py 的 generate_anomalous_triples_2。"""
    neg_triples: List[TripleId] = []
    for _ in range(num_anomaly):
        new_head = random.randint(0, num_entities - 1)
        new_relation = random.randint(0, num_relations - 1)
        new_tail = random.randint(0, num_entities - 1)
        anomaly = (new_head, new_relation, new_tail)
        while anomaly in triple_ori_set:
            new_head = random.randint(0, num_entities - 1)
            new_relation = random.randint(0, num_relations - 1)
            new_tail = random.randint(0, num_entities - 1)
            anomaly = (new_head, new_relation, new_tail)
        neg_triples.append(anomaly)
    return neg_triples


def inject_anomaly_like_caged(
    all_triples: Sequence[TripleId],
    num_entities: int,
    num_relations: int,
    anomaly_ratio: float,
    exclude_triple_set: Optional[set[TripleId]] = None,
) -> Tuple[List[TripleId], List[int], int]:
    """
    复现 CAGED Reader 的 inject_anomaly + get_data_test 行为：
    - 原始三元组 label=0
    - 注入异常 label=1
    - 返回 bp_triples（原+异常，已 shuffle）及 labels_anom（0/1），并返回 num_original_triples
    """
    # 只在某个 split 注入时，也要避免生成落在“全图真三元组”里的样本，
    # 否则会把真实三元组错误标成异常，导致 P@K/R@K 异常偏低（尤其在 WN18RR 等数据上更明显）。
    triple_ori_set: set[TripleId] = exclude_triple_set if exclude_triple_set is not None else set(all_triples)
    num_original_triples = len(all_triples)
    num_anomalies = int(anomaly_ratio * num_original_triples)

    # 与 CAGED 一致：采样 (num_anomalies//2) 条原三元组做“定向腐化”，再加 (num_anomalies//2) 条全随机异常
    idx = random.sample(range(0, max(1, num_original_triples - 1)), max(1, num_anomalies // 2))
    selected_triples = [all_triples[i] for i in idx]
    anomalies = _generate_anomalous_triples_like_caged(selected_triples, num_entities, num_relations, triple_ori_set) + _generate_anomalous_triples_2_like_caged(
        num_anomalies // 2, num_entities, num_relations, triple_ori_set
    )

    triple_label = [(tr, 0) for tr in all_triples]
    anomaly_label = [(tr, 1) for tr in anomalies]
    triple_anomaly_label = triple_label + anomaly_label
    random.shuffle(triple_anomaly_label)

    bp_triples = [x[0] for x in triple_anomaly_label]
    labels_anom = [int(x[1]) for x in triple_anomaly_label]
    return bp_triples, labels_anom, num_original_triples


def _batch_score_transe(
    model,
    triples: Sequence[TripleId],
    device: torch.device,
    batch_size: int = 8192,
) -> List[float]:
    """
    返回 score（越高越正确）= -raw_score（直接使用 TransE 原始打分，不做 sigmoid/归一化）。
    说明：TransE 的 raw_score 越小越好，所以取负号后满足“越大越正确”的约定。
    """
    scores: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            chunk = triples[i : i + batch_size]
            h = torch.tensor([x[0] for x in chunk], dtype=torch.long, device=device)
            r = torch.tensor([x[1] for x in chunk], dtype=torch.long, device=device)
            t = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)
            raw = model.score_triple(h, r, t)
            sc = (-raw).detach().cpu().numpy().astype(np.float64)
            scores.extend(sc.tolist())
    return scores


@dataclass(frozen=True)
class Spec:
    dataset: str
    anomaly_ratio: float


def run_one(
    dataset: str,
    anomaly_ratio: float,
    k_ratios: Tuple[float, ...],
    seed: int,
    device: torch.device,
    k_base: str,
    inject_split: str,
) -> dict:
    cfg = load_config()
    out_dir = cfg.get("output_dir", "./outputs")
    base = (_ROOT / out_dir.lstrip("./")).resolve()
    ckpt_path = (base / "embeddings" / f"{dataset.lower()}_transE.pt").resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TransE checkpoint 不存在: {ckpt_path}。请先训练 TransE。")

    ds = load_dataset_by_name(cfg, dataset)
    all_true_triples: set[TripleId] = set(ds.train_triples) | set(ds.valid_triples) | set(ds.test_triples)
    if inject_split == "train":
        clean_triples: List[TripleId] = list(ds.train_triples)
    elif inject_split == "valid":
        clean_triples = list(ds.valid_triples)
    elif inject_split == "test":
        clean_triples = list(ds.test_triples)
    elif inject_split == "all":
        clean_triples = list(ds.train_triples) + list(ds.valid_triples) + list(ds.test_triples)
    else:
        raise ValueError("--inject_split must be train|valid|test|all")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bp_triples, labels_anom, num_original_triples = inject_anomaly_like_caged(
        all_triples=clean_triples,
        num_entities=ds.num_entities,
        num_relations=ds.num_relations,
        anomaly_ratio=anomaly_ratio,
        exclude_triple_set=all_true_triples,
    )
    labels_pos = [1 - int(v) for v in labels_anom]

    model = load_transe_from_checkpoint(str(ckpt_path), device=device)
    scores = _batch_score_transe(model, bp_triples, device=device, batch_size=8192)
    if k_base == "original":
        ref = num_original_triples
    elif k_base == "total":
        ref = len(bp_triples)
    else:
        raise ValueError("--k_base must be original|total")
    pk = compute_precision_recall_at_k_anomaly(labels_pos, scores, k_ratios=k_ratios, k_ref_size=ref)

    row = {
        "dataset": dataset,
        "anomaly_ratio": float(anomaly_ratio),
        "seed": int(seed),
        "inject_split": inject_split,
        "num_original_triples": int(num_original_triples),
        "k_base": k_base,
        "k_ref_size": int(ref),
        "transE_ckpt": str(ckpt_path),
    }
    for r in k_ratios:
        pct = int(r * 100)
        row[f"k_{pct}%"] = pk.get(f"k_{pct}", "")
        row[f"p_at_{pct}%"] = pk.get(f"precision_at_{pct}", "")
        row[f"r_at_{pct}%"] = pk.get(f"recall_at_{pct}", "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="TransE Top-K (1%~5%) 评测并汇总 CSV（口径与 CAGED 一致）")
    parser.add_argument("--device", type=str, default=None, help="如 cuda / cuda:0 / cpu；默认自动选择")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（噪声注入与 shuffle）")
    parser.add_argument("--only", type=str, default=None, help="只跑某个数据集：fb15k237 / wn18rr / nell995")
    parser.add_argument(
        "--k_base",
        type=str,
        default="original",
        choices=["original", "total"],
        help="K 的基数：original=ratio*num_original_triples（CAGED 口径，默认）; total=ratio*len(原+注入)",
    )
    parser.add_argument(
        "--inject_split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "all"],
        help="在哪个 split 上注入噪声并评测：默认 test（更贴近常规测试设定）；all 则为 train+valid+test（更贴近 CAGED Reader 实现）",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(args.seed)

    specs: List[Spec] = [
        Spec("fb15k237", 0.05),
        Spec("fb15k237", 0.10),
        Spec("fb15k237", 0.15),
        Spec("wn18rr", 0.05),
        Spec("wn18rr", 0.10),
        Spec("nell995", 0.05),
    ]
    if args.only:
        only = args.only.strip().lower()
        specs = [s for s in specs if s.dataset.lower() == only]
        if not specs:
            raise ValueError(f"--only={args.only} 不在可选数据集内：fb15k237 / wn18rr / nell995")

    # K 比例：1%~5%
    k_ratios = (0.01, 0.02, 0.03, 0.04, 0.05)

    rows: List[dict] = []
    for s in specs:
        row = run_one(
            s.dataset,
            s.anomaly_ratio,
            k_ratios=k_ratios,
            seed=seed,
            device=device,
            k_base=args.k_base,
            inject_split=args.inject_split,
        )
        rows.append(row)
        print(
            f"[OK] {s.dataset} anomaly={s.anomaly_ratio:.2f} "
            f"P@1%={row['p_at_1%']:.4f} R@1%={row['r_at_1%']:.4f}  "
            f"P@5%={row['p_at_5%']:.4f} R@5%={row['r_at_5%']:.4f}"
        )

    cfg = load_config()
    out_dir = cfg.get("output_dir", "./outputs")
    base = (_ROOT / out_dir.lstrip("./")).resolve()
    save_dir = base / "transe_topk_metrics"
    save_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = save_dir / f"transe_topk_summary_{ts}.csv"

    fieldnames = [
        "dataset",
        "anomaly_ratio",
        "seed",
        "inject_split",
        "num_original_triples",
        "k_base",
        "k_ref_size",
        "k_1%",
        "p_at_1%",
        "r_at_1%",
        "k_2%",
        "p_at_2%",
        "r_at_2%",
        "k_3%",
        "p_at_3%",
        "r_at_3%",
        "k_4%",
        "p_at_4%",
        "r_at_4%",
        "k_5%",
        "p_at_5%",
        "r_at_5%",
        "transE_ckpt",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n汇总已写入: {csv_path}")


if __name__ == "__main__":
    main()

