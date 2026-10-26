"""
带噪声数据上跑结构智能体，并收集 AUC、准确率、Precision、Recall、F1、P@K/R@K、正/负例分布等指标。

默认使用 FB15K-237、30% 噪声；可通过 --dataset / --noise_ratio 指定。

流程：
  1. 加载指定数据集的指定划分（默认 test），可选 --max_triples 做子集。
  2. 用 noise_injector 按比例注入负例（随机替换头/尾），得到带标签 (triple, label)。
  3. 用结构智能体对每条 triple 打 S_struct 分。
  4. 按阈值计算二分类指标：默认自动选使 F1 最大的黄金阈值（可 --threshold 固定）；并计算 Precision@K、Recall@K（K=5%,10%,15%）。
  5. 输出正例/负例的 S_struct 分布（mean、median、直方图），可选保存 JSONL 与指标汇总。

指标说明（合理性）：
  - 现有 Precision/Recall/F1：依赖单一阈值，S_struct>=threshold 判为正。优点：与分类决策直接对应；
    缺点：阈值敏感，不同阈值下指标差异大，且无法区分「高分段是否更纯净」。
  - Precision@K、Recall@K（K=5%,10%,15%）：按分数从高到低取「前 K%」样本，不依赖阈值。
    · Precision@K = 前 K% 中真正例数 / (K%×N)：衡量「最可信的 K%」里有多少真是正确，适合「只信任模型最自信的一小部分」的场景。
    · Recall@K = 前 K% 中真正例数 / 总正例数：衡量正确三元组中有多少被排进最可信的 K%，适合「人工只复核前 K%」等资源受限场景。
    优点：与 AUC 一样不依赖阈值、强调排序质量，且 K 有明确业务含义；缺点：K 的选取较主观，K 过小时分母小、波动可能略大。

依赖：TransE 已训练，outputs/embeddings/{dataset}_transE.pt 存在。

在项目根目录 CC_gra 下执行：
  python examples/run_structure_agent_noisy_fb15k.py
  python examples/run_structure_agent_noisy_fb15k.py --dataset fb15k237 --split test --noise_ratio 0.3 --out outputs/structure_noisy_fb15k237_test.jsonl
  python examples/run_structure_agent_noisy_fb15k.py --dataset fb15k237 --noise_ratio 0.3 --threshold 0.5
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from ccg_data.kg_dataset import KGDataset
from ccg_data.path_mining import build_graph
from ccg_data.noise_injector import inject_noise
from ccg_pipeline.train_embeddings import load_config, load_transe_from_checkpoint
from ccg_agents.structure_agent import StructureConstraintAgent, StructureAgentConfig

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    def roc_auc_score(y_true, y_score):
        """备用 AUC：按分数降序排列，用秩和（Mann-Whitney）计算。正类 label=1。"""
        if len(y_true) == 0 or sum(y_true) == 0 or sum(y_true) == len(y_true):
            return 0.0
        order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        S = sum((rank + 1) for rank, i in enumerate(order) if y_true[i] == 1)
        return (S - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def compute_metrics(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
) -> dict:
    """
    根据真实标签与 S_struct 分数计算二分类指标。
    label=1 表示正确三元组，0 表示噪声；S_struct 越高越判为正（正确）。
    """
    preds = [1 if s >= threshold else 0 for s in scores]
    n = len(labels)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

    tp = sum(1 for i in range(n) if preds[i] == 1 and labels[i] == 1)
    tn = sum(1 for i in range(n) if preds[i] == 0 and labels[i] == 0)
    fp = sum(1 for i in range(n) if preds[i] == 1 and labels[i] == 0)
    fn = sum(1 for i in range(n) if preds[i] == 0 and labels[i] == 1)

    accuracy = (tp + tn) / n  # 准确率 HR / Hit
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_precision_recall_at_k_anomaly(
    labels: List[int],
    scores: List[float],
    k_ratios: Tuple[float, ...] = (0.05, 0.10, 0.15),
    k_ref_size: int | None = None,
) -> dict:
    """
    排序度量（Ranking Metrics），在「错误检测」视角下的 P@K / R@K：
    - labels: 1 表示正确三元组，0 表示错误/噪声；
    - scores: S_struct，越高越「正确」；
    - 先将 scores 映射为异常分数 anomaly_score = 1 - score（越大越可能是错误）；
    - 再按异常分数从高到低排序，取「最可能为错误」的前 K 个样本：
        · K = ratio * k_ref_size（若提供，例如原始三元组数），否则 K = ratio * n；
        · Precision@K = 前 K 个中真实错误数 / K；
        · Recall@K    = 前 K 个中真实错误数 / 全部真实错误数。
    """
    n = len(labels)
    labels_anom = [1 - y for y in labels]
    n_anom = sum(labels_anom)
    result: dict = {}
    if n == 0 or n_anom == 0:
        for r in k_ratios:
            pct = int(r * 100)
            result[f"precision_at_{pct}"] = 0.0
            result[f"recall_at_{pct}"] = 0.0
        return result

    ref = k_ref_size if k_ref_size is not None else n
    anomaly_scores = [1.0 - s for s in scores]
    order = sorted(range(n), key=lambda i: -anomaly_scores[i])

    for r in k_ratios:
        n_top = max(1, min(int(ref * r), n))
        top_indices = order[:n_top]
        n_anom_in_top = sum(1 for i in top_indices if labels_anom[i] == 1)
        precision_at_k = n_anom_in_top / n_top if n_top > 0 else 0.0
        recall_at_k = n_anom_in_top / n_anom if n_anom > 0 else 0.0
        pct = int(r * 100)
        result[f"precision_at_{pct}"] = precision_at_k
        result[f"recall_at_{pct}"] = recall_at_k
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="带噪声数据：注入负例 → 结构智能体打分 → AUC 与分布统计（默认 FB15K-237）"
    )
    parser.add_argument("--dataset", type=str, default="fb15k237", help="数据集名，如 fb15k237 / fb15k / wn18rr / nell995")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "valid", "test"]
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.3,
        help="负例比例（在划分内按比例从正例中随机替换头/尾生成负例），默认 0.3（30%%噪声）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="T",
        help="固定阈值；不传则自动遍历候选阈值并选取使 F1 最大的黄金阈值（自适应模式）",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.05,
        help="黄金阈值搜索步长（仅在不传 --threshold 时有效），默认 0.05",
    )
    parser.add_argument(
        "--max_triples",
        type=int,
        default=None,
        help="最多使用多少条正例（用于快速测试），默认不限制",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="保存 (triple, label, S_struct) 的 JSONL 路径",
    )
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if hasattr(torch, "manual_seed_all"):
        torch.manual_seed_all(args.seed)

    # ---------- 加载配置与数据 ----------
    cfg = load_config()
    ds_cfg = cfg.get(dataset_name)
    if not isinstance(ds_cfg, dict) or "train_path" not in ds_cfg:
        raise KeyError(f"配置中未找到数据集 {dataset_name}，或缺少 train_path")
    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")

    ds = KGDataset.load_from_files(str(train_path), str(valid_path), str(test_path))
    triples = ds.get_all_triples(args.split)
    if args.max_triples is not None and len(triples) > args.max_triples:
        random.seed(args.seed)
        triples = random.sample(triples, args.max_triples)
        print(f"使用子集: {len(triples)} 条 (--max_triples={args.max_triples})")

    num_entities = ds.num_entities
    clean_triples, noisy_triples, labeled = inject_noise(
        triples,
        num_entities,
        args.noise_ratio,
        mode="head_tail",
    )
    n_pos = len(clean_triples)
    n_neg = len(noisy_triples)
    print(f"数据集: {dataset_name}, 划分: {args.split}")
    print(f"正例: {n_pos}, 负例: {n_neg}, 合计: {len(labeled)}, noise_ratio={args.noise_ratio}")

    # ---------- 结构智能体 ----------
    graph = build_graph(ds.train_triples)
    ckpt_path = base / "embeddings" / f"{dataset_name}_transE.pt"
    if not ckpt_path.exists():
        print(f"未找到 TransE 权重: {ckpt_path}，请先训练 TransE。")
        return
    transe = load_transe_from_checkpoint(str(ckpt_path), device=device)
    transe.eval()

    agent = StructureConstraintAgent(
        transe_model=transe,
        graph=graph,
        llm_explainer=None,
        id2entity=None,
        id2relation=None,
        save_explanation_path=None,
        config=StructureAgentConfig(emb_mode="transe"),
    )

    # ---------- 对带标签三元组打分 ----------
    scores: List[float] = []
    labels: List[int] = []
    records: List[Tuple[List[int], int, float]] = []  # (triple, label, S_struct)

    for (h, r, t, label) in tqdm(labeled, desc="打分", unit="triple"):
        result = agent.score_triple((h, r, t))
        s = result["S_struct"]
        scores.append(s)
        labels.append(label)
        records.append(([h, r, t], label, s))

    # ---------- 阈值：不传 --threshold 则自适应选黄金阈值 ----------
    if args.threshold is None:
        step = max(0.01, min(0.5, args.threshold_step))
        candidates = []
        t = step
        while t < 1.0:
            candidates.append(t)
            t += step
        if not candidates:
            candidates = [0.5]
        best_f1 = -1.0
        best_threshold = 0.5
        best_metrics = None
        for th in candidates:
            m = compute_metrics(labels, scores, threshold=th)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_threshold = th
                best_metrics = m
        args.threshold = best_threshold
        metrics = best_metrics
        used_gold_threshold = True
        #print("\n黄金阈值搜索（按 F1 最大）: 候选 {} 个，最优 threshold = {:.3f}, F1 = {:.4f}".format(
        #    len(candidates), best_threshold, best_f1))
    else:
        metrics = compute_metrics(labels, scores, threshold=args.threshold)
        used_gold_threshold = False

    # ---------- 二分类指标 ----------
    pk_ratios = (0.05, 0.10, 0.15)
    # 排序度量：按异常分数从高到低取前 K 个（最可能为错误的 K 个），
    # K 的基准为原始正例数量 n_pos，与 CAGED 设置一致。
    pk_metrics = compute_precision_recall_at_k_anomaly(
        labels, scores, k_ratios=pk_ratios, k_ref_size=n_pos
    )

    print("\n" + "=" * 60)
    print("结构智能体 — 错误知识检测指标（阈值 = {:.3f}）".format(args.threshold))
    print("=" * 60)
    print("  Accuracy       = {:.4f}".format(metrics["accuracy"]))
    print("  Precision      = {:.4f}".format(metrics["precision"]))
    print("  Recall         = {:.4f}".format(metrics["recall"]))
    print("  F1             = {:.4f}".format(metrics["f1"]))
    print("  AUC            = {:.4f}".format(metrics["auc"]))
    print("  (TP={}, TN={}, FP={}, FN={})".format(
        metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]))
    print("  --- Precision@K / Recall@K ---")
    for r in pk_ratios:
        pct = int(r * 100)
        print("  P@{:2d}% = {:.4f}   R@{:2d}% = {:.4f}".format(
            pct, pk_metrics[f"precision_at_{pct}"], pct, pk_metrics[f"recall_at_{pct}"]))

    # ---------- 正例 / 负例 S_struct 分布（mean、median、直方图）----------
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    def mean_median_hist(name: str, arr: List[float]) -> Tuple[float, float]:
        if not arr:
            print(f"  {name}: 无数据")
            return 0.0, 0.0
        n = len(arr)
        mean_s = sum(arr) / n
        sorted_s = sorted(arr)
        mid = n // 2
        median_s = (
            (sorted_s[mid] + sorted_s[mid - 1]) / 2.0
            if n >= 2 and n % 2 == 0
            else (sorted_s[mid] if n else 0.0)
        )
        print(f"  {name}  mean = {mean_s:.4f}, median = {median_s:.4f}")
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in bins:
            count = sum(1 for x in arr if low <= x < high)
            pct = 100.0 * count / n
            bar = "#" * int(pct / 2) + " " * (50 - int(pct / 2))
            print(f"    [{low:.1f}-{high:.1f})  {pct:5.1f}%  {bar}")
        return mean_s, median_s

    print("\nS_struct 分布（正例）")
    pos_mean, pos_median = mean_median_hist("正例", pos_scores)
    print("\nS_struct 分布（负例）")
    neg_mean, neg_median = mean_median_hist("负例", neg_scores)

    # ---------- 保存 JSONL 与指标汇总 ----------
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for triple, label, s in records:
                f.write(
                    json.dumps(
                        {"triple": triple, "label": label, "S_struct": s},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"\n结果已写入: {out_path}")

        # 同目录下保存指标汇总（便于后续整理实验数据）
        summary_path = out_path.parent / (out_path.stem + "_metrics.json")
        summary = {
            "dataset": dataset_name,
            "split": args.split,
            "noise_ratio": args.noise_ratio,
            "threshold": args.threshold,
            "gold_threshold": used_gold_threshold,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "tp": metrics["tp"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"],
            "precision_at_5": pk_metrics["precision_at_5"],
            "recall_at_5": pk_metrics["recall_at_5"],
            "precision_at_10": pk_metrics["precision_at_10"],
            "recall_at_10": pk_metrics["recall_at_10"],
            "precision_at_15": pk_metrics["precision_at_15"],
            "recall_at_15": pk_metrics["recall_at_15"],
            "pos_mean": pos_mean, "pos_median": pos_median,
            "neg_mean": neg_mean, "neg_median": neg_median,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"指标汇总已写入: {summary_path}")


if __name__ == "__main__":
    main()
