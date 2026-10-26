"""
DKRL（CNN 描述向量 + TransE 式 L1）在 WN18RR 上的噪声检测评测。

与 examples/run_transe_baseline_noisy_wn18rr.py 流程一致，仅将结构置信度从 TransE 换为
DKRL 官方 Test_cnn 同款：先由 word2vec + CNN 权重得到 cnn_vec，再
S = sigmoid(sum_i -|cnn[t]-cnn[h]-rel[r]|_i)。

依赖：
  - 已用 DKRL 在 WN18RR 上训练完成，并在 sky/res（或 --dkrl-res-dir）下生成
    relation2vec.{bern|unif}, word2vec.*, weight1.*, weight2.*
  - sky/data（或 --dkrl-data-dir）下为 prepare 脚本生成的 entity2id / relation2id /
    word2id / entityWords.txt（与训练时一致）

示例（在 CC_gra 根目录）：
  python examples/run_dkrl_noisy_wn18rr.py \\
    --dkrl-data-dir ../data --dkrl-res-dir ../res --version bern
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.noise_injector import inject_noise
from ccg_pipeline.train_embeddings import load_config
from ccg_utils.dkrl_cnn_forward import (
    build_all_cnn_vec,
    dkrl_struct_confidence_sigmoid,
)

TripleStr = Tuple[str, str, str]
TripleId = Tuple[int, int, int]


def _read_triples_raw(path: Path) -> List[TripleStr]:
    triples: List[TripleStr] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _load_dkrl_maps(data_dir: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str], int, int]:
    entity2id: Dict[str, int] = {}
    relation2id: Dict[str, int] = {}
    with (data_dir / "entity2id.txt").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split("\t") if "\t" in line else line.split()
            entity2id[a] = int(b)
    with (data_dir / "relation2id.txt").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split("\t") if "\t" in line else line.split()
            relation2id[a] = int(b)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    n_ent = len(entity2id)
    n_rel = len(relation2id)
    return entity2id, relation2id, id2entity, id2relation, n_ent, n_rel


def _load_word2id(data_dir: Path) -> Dict[str, int]:
    m: Dict[str, int] = {}
    with (data_dir / "word2id.txt").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            w, wid = parts[0], int(parts[1])
            m[w] = wid
    return m


def _load_entity_words_lists(
    data_dir: Path,
    entity2id: Dict[str, int],
    word2id: Dict[str, int],
    entity_num: int,
) -> List[List[int]]:
    path = data_dir / "entityWords.txt"
    entity_words: List[List[int]] = [[] for _ in range(entity_num)]
    with path.open(encoding="utf-8") as f:
        while True:
            hdr = f.readline()
            if not hdr:
                break
            hdr = hdr.strip()
            if not hdr:
                continue
            parts = hdr.split("\t") if "\t" in hdr else hdr.split()
            ent, cnt_s = parts[0], int(parts[1])
            eid = entity2id[ent]
            wids: List[int] = []
            for _ in range(cnt_s):
                wline = f.readline()
                if not wline:
                    break
                tok = wline.strip()
                wids.append(word2id[tok])
            entity_words[eid] = wids
    return entity_words


def _map_triple_str_to_dkrl(
    h: str,
    r: str,
    t: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> Optional[TripleId]:
    if h not in entity2id or t not in entity2id or r not in relation2id:
        return None
    return (entity2id[h], relation2id[r], entity2id[t])


def _load_dkrl_res(
    res_dir: Path,
    version: str,
    relation_num: int,
    word_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    rel_p = res_dir / f"relation2vec.{version}"
    w_p = res_dir / f"word2vec.{version}"
    w1_p = res_dir / f"weight1.{version}"
    w2_p = res_dir / f"weight2.{version}"
    for p in (rel_p, w_p, w1_p, w2_p):
        if not p.exists():
            raise FileNotFoundError(f"缺少 DKRL 输出文件: {p}")

    rel_flat = np.loadtxt(rel_p, dtype=np.float64)
    n = int(rel_flat.size // relation_num)
    relation_vec = rel_flat.reshape(relation_num, n)

    w_flat = np.loadtxt(w_p, dtype=np.float64)
    n_w = int(w_flat.size // word_num)
    word_vec = w_flat.reshape(word_num, n_w)

    conv_w1 = np.loadtxt(w1_p, dtype=np.float64)
    n_1 = int(conv_w1.shape[0])

    conv_w2 = np.loadtxt(w2_p, dtype=np.float64)
    if conv_w2.shape[0] != n:
        raise ValueError(
            f"weight2 行数 {conv_w2.shape[0]} 与嵌入维 n={n} 不一致，请检查训练参数与数据。"
        )
    return relation_vec, word_vec, conv_w1, conv_w2, n, n_1, n_w


try:
    from sklearn.metrics import roc_auc_score
except ImportError:

    def roc_auc_score(y_true, y_score):  # type: ignore
        if len(y_true) == 0 or sum(y_true) == 0 or sum(y_true) == len(y_true):
            return 0.0
        order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        s = sum((rank + 1) for rank, i in enumerate(order) if y_true[i] == 1)
        return (s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_metrics(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
) -> dict:
    preds = [1 if s >= threshold else 0 for s in scores]
    n = len(labels)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

    tp = sum(1 for i in range(n) if preds[i] == 1 and labels[i] == 1)
    tn = sum(1 for i in range(n) if preds[i] == 0 and labels[i] == 0)
    fp = sum(1 for i in range(n) if preds[i] == 1 and labels[i] == 0)
    fn = sum(1 for i in range(n) if preds[i] == 0 and labels[i] == 1)

    accuracy = (tp + tn) / n
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_precision_recall_at_k_anomaly(
    labels: List[int],
    scores: List[float],
    k_ratios: Tuple[float, ...] = (0.05, 0.10, 0.15),
    k_ref_size: int | None = None,
) -> dict:
    """与 run_transe_baseline_noisy_wn18rr 一致：label=1 正确，score 越高越正确。"""
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
    sky_root = _ROOT.parent
    # 与 DKRL 训练一致：数据在 sky/DKRL/data，权重在 sky/DKRL/res（repo 下 ../data / ../res）
    default_data = sky_root / "DKRL" / "data"
    default_res = sky_root / "DKRL" / "res"

    parser = argparse.ArgumentParser(
        description="DKRL（CNN+TransE）在 WN18RR 噪声数据上的错误检测 + P@K/R@K"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="WN18RR 划分（路径来自 ccg_config/datasets.yaml）",
    )
    parser.add_argument("--noise_ratio", type=float, default=0.3, help="注入负例比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_triples", type=int, default=None)
    parser.add_argument(
        "--dkrl-data-dir",
        type=Path,
        default=default_data,
        help="DKRL 数据目录（entity2id / word2id / entityWords 等），默认 sky/data",
    )
    parser.add_argument(
        "--dkrl-res-dir",
        type=Path,
        default=default_res,
        help="DKRL 训练输出目录，默认 sky/res",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="bern",
        choices=["bern", "unif"],
        help="与 Train_cnn_multi -method 一致：bern 或 unif",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--out", type=str, default=None, help="JSONL：triple, label, score")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config()
    ds_cfg = cfg.get("wn18rr")
    if not isinstance(ds_cfg, dict) or "train_path" not in ds_cfg:
        raise KeyError("ccg_config/datasets.yaml 中缺少 wn18rr 配置")

    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")
    split_path = {"train": train_path, "valid": valid_path, "test": test_path}[args.split]

    entity2id, relation2id, _id2e, _id2r, n_ent, n_rel = _load_dkrl_maps(args.dkrl_data_dir)
    word2id = _load_word2id(args.dkrl_data_dir)
    word_num = len(word2id)
    entity_words = _load_entity_words_lists(args.dkrl_data_dir, entity2id, word2id, n_ent)

    if n_ent != len(entity_words):
        raise ValueError("entity2id 与 entityWords 实体数不一致")

    triples_str = _read_triples_raw(split_path)
    if args.max_triples is not None and len(triples_str) > args.max_triples:
        triples_str = random.sample(triples_str, args.max_triples)

    triples_dkrl: List[TripleId] = []
    skipped = 0
    for h, r, t in triples_str:
        mid = _map_triple_str_to_dkrl(h, r, t, entity2id, relation2id)
        if mid is None:
            skipped += 1
            continue
        triples_dkrl.append(mid)
    if skipped:
        print(f"警告: {skipped} 条三元组无法映射到 DKRL 词表（已跳过）")
    if not triples_dkrl:
        raise RuntimeError("没有可评测的三元组，请检查 dkrl-data-dir 与 WN18RR 划分是否一致")

    clean_triples, noisy_triples, labeled = inject_noise(
        triples_dkrl,
        n_ent,
        args.noise_ratio,
        mode="head_tail",
    )
    n_pos = len(clean_triples)
    n_neg = len(noisy_triples)
    print(
        f"数据集: wn18rr  split={args.split}  DKRL实体数={n_ent}  关系数={n_rel}  词表={word_num}"
    )
    print(f"正例: {n_pos}, 负例: {n_neg}, 合计: {len(labeled)}, noise_ratio={args.noise_ratio}")

    relation_vec, word_vec, conv_w1, conv_w2, n, n_1, n_w = _load_dkrl_res(
        args.dkrl_res_dir, args.version, n_rel, word_num
    )
    if relation_vec.shape[1] != n:
        raise ValueError("relation2vec 维度与 weight2 不一致")
    print(
        f"已加载 DKRL 权重: n={n}, n_1={n_1}, n_w={n_w}, version={args.version} "
        f"from {args.dkrl_res_dir}"
    )

    print("构建 cnn_vec（全实体）…")
    cnn_vec = build_all_cnn_vec(entity_words, word_vec, conv_w1, conv_w2, n, n_1, n_w)

    scores: List[float] = []
    labels: List[int] = []
    records: List[Tuple[List[int], int, float]] = []

    for h, r, t, label in labeled:
        s = dkrl_struct_confidence_sigmoid(cnn_vec, relation_vec, h, r, t)
        scores.append(s)
        labels.append(label)
        records.append(([h, r, t], label, s))

    if args.threshold is None:
        step = max(0.01, min(0.5, args.threshold_step))
        candidates: List[float] = []
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
        used_gold = True
    else:
        metrics = compute_metrics(labels, scores, threshold=args.threshold)
        used_gold = False

    pk_ratios = (0.05, 0.10, 0.15)
    pk_metrics = compute_precision_recall_at_k_anomaly(
        labels, scores, k_ratios=pk_ratios, k_ref_size=n_pos
    )

    print("\n" + "=" * 60)
    print(
        "DKRL（CNN+TransE）— WN18RR 错误检测（阈值 = {:.3f}，{}）".format(
            args.threshold, "黄金阈值" if used_gold else "固定阈值"
        )
    )
    print("=" * 60)
    print(f"  Accuracy       = {metrics['accuracy']:.4f}")
    print(f"  Precision      = {metrics['precision']:.4f}")
    print(f"  Recall         = {metrics['recall']:.4f}")
    print(f"  F1             = {metrics['f1']:.4f}")
    print(f"  AUC            = {metrics['auc']:.4f}")
    print(
        f"  (TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']})"
    )
    print("  --- P@K / R@K（K=比例×原始正例数，与 TransE 噪声脚本一致）---")
    for r in pk_ratios:
        pct = int(r * 100)
        print(
            f"  P@{pct:2d}% = {pk_metrics[f'precision_at_{pct}']:.4f}   "
            f"R@{pct:2d}% = {pk_metrics[f'recall_at_{pct}']:.4f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for triple, label, score in records:
                f.write(
                    json.dumps(
                        {"triple": triple, "label": int(label), "score": float(score)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"\n已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
