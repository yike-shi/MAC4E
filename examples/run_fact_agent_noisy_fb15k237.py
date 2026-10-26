"""
事实验证智能体在带噪声 FB15K-237 上的实验：抽样 → 注入负例 → 事实智能体打分 → 评估指标。

流程：
  1. 加载 FB15K-237 指定划分（默认 test），可选 --max_triples 做子集。
  2. 用 noise_injector 按比例注入负例（随机替换头/尾），得到带标签 (triple, label)，label=1 为正例，0 为负例。
  3. 用事实智能体对每条 triple 打 S_fact 分（需实体/关系可读映射与 LLM）。
  4. 按阈值将 S_fact 转为二类预测，计算 Accuracy(HIT)、Precision、Recall、F1；并计算 AUC。

依赖：
  - CC_gra/data/FB15k 下 fb15k237-*.txt 及 entity2text.txt、relation2text.txt
  - 本地 Qwen（transformers）；若加载失败则用 Mock，指标无参考价值

建议先验证 LLM 是否正常：运行（只跑 5 条并打印每条结果）：
  python examples/run_fact_agent_fb15k237.py
再跑本脚本做带噪声实验；或先用少量样本试跑：
  python examples/run_fact_agent_noisy_fb15k237.py --max_triples 5

在项目根目录 CC_gra 下执行：
  python examples/run_fact_agent_noisy_fb15k237.py
  python examples/run_fact_agent_noisy_fb15k237.py --split test --noise_ratio 0.3 --max_triples 500
  python examples/run_fact_agent_noisy_fb15k237.py --threshold 0.5 --out outputs/fact_noisy_fb15k237_test.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.text_mapping import load_fb15k237_text_maps
from ccg_data.noise_injector import inject_noise
from ccg_agents.llm_client import LocalQwenClient, MockLLMClient
from ccg_agents.fact_agent import FactVerificationAgent, FactAgentConfig
from ccg_pipeline.train_embeddings import load_config

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    def roc_auc_score(y_true, y_score):
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


def _create_llm_client():
    """与 run_fact_agent_fb15k237 一致：本地 Qwen 优先，失败则 Mock。"""
    model_name = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    try:
        client = LocalQwenClient(
            model_name=model_name,
            device_map="auto",
        )
        return client, f"本地 {model_name}"
    except Exception:
        return MockLLMClient(), None


def compute_metrics(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
) -> dict:
    """
    根据真实标签与 S_fact 分数计算二分类指标。
    label=1 表示正确三元组，0 表示噪声；S_fact 越高越判为正（正确）。
    """
    preds = [1 if s >= threshold else 0 for s in scores]
    n = len(labels)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

    tp = sum(1 for i in range(n) if preds[i] == 1 and labels[i] == 1)
    tn = sum(1 for i in range(n) if preds[i] == 0 and labels[i] == 0)
    fp = sum(1 for i in range(n) if preds[i] == 1 and labels[i] == 0)
    fn = sum(1 for i in range(n) if preds[i] == 0 and labels[i] == 1)

    accuracy = (tp + tn) / n  # HIT / 准确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0

    return {
        "accuracy": accuracy,   # HIT
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="带噪声 FB15K-237：注入负例 → 事实智能体打分 → HIT/Precision/Recall/F1/AUC"
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.2,
        help="负例比例（在划分内按比例从正例中随机替换头/尾生成负例），默认 0.2",
    )
    parser.add_argument(
        "--max_triples",
        type=int,
        default=None,
        help="最多使用多少条正例（抽样后再注噪）；建议先用 2~5 条验证 LLM 是否正常，默认不限制",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="S_fact >= threshold 判为正例（正确三元组），默认 0.5",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="保存 (triple, label, S_fact) 的 JSONL 路径",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_name = "fb15k237"
    cfg = load_config()
    ds_cfg = cfg.get(dataset_name) or cfg.get("fb15k237")
    if not ds_cfg:
        print("未找到 fb15k237 配置。")
        return

    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")

    if not train_path.exists():
        print(f"训练集不存在: {train_path}")
        return

    ds = KGDataset.load_from_files(str(train_path), str(valid_path), str(test_path))
    triples = ds.get_all_triples(args.split)
    if args.max_triples is not None and len(triples) > args.max_triples:
        triples = random.sample(triples, args.max_triples)
        print(f"抽样: {len(triples)} 条 (--max_triples={args.max_triples})")

    num_entities = ds.num_entities
    clean_triples, noisy_triples, labeled = inject_noise(
        triples, num_entities, args.noise_ratio, mode="head_tail",
    )
    n_pos = len(clean_triples)
    n_neg = len(noisy_triples)
    print(f"数据集: {dataset_name}, 划分: {args.split}")
    print(f"正例: {n_pos}, 负例: {n_neg}, 合计: {len(labeled)}, noise_ratio={args.noise_ratio}")

    data_root = _ROOT / "data" / "FB15k"
    try:
        ent2text, rel2text = load_fb15k237_text_maps(root=str(data_root))
        print(f"已加载实体映射 {len(ent2text)} 条、关系映射 {len(rel2text)} 条。")
    except FileNotFoundError as e:
        print(f"映射文件缺失: {e}，事实验证将使用默认分数。")
        ent2text, rel2text = {}, {}

    client, client_src = _create_llm_client()
    if client_src:
        print(f"使用 {client_src} 进行事实验证。")
    else:
        print("本地 Qwen 加载失败，使用 MockLLMClient（分数恒为 0.5）。")
        print("  → 此时指标无参考价值。请先运行: python examples/run_fact_agent_fb15k237.py 验证 LLM 正常后再跑本脚本。")

    agent = FactVerificationAgent(
        llm_client=client,
        id2entity=ds.id2entity,
        id2relation=ds.id2relation,
        ent2text=ent2text,
        rel2text=rel2text,
        config=FactAgentConfig(
            default_score_no_readable=0.5,
            skip_fact_when_no_readable=True,
            max_tokens=300,
        ),
        save_result_path=None,
    )

    scores: List[float] = []
    labels: List[int] = []
    records: List[Tuple[List[int], int, float]] = []

    verbose = len(labeled) <= 10  # 样本很少时逐条打印，便于检查 LLM 是否正常
    it = tqdm(labeled, desc="事实智能体打分", unit="triple")
    for (h, r, t, label) in it:
        result = agent.score_triple((h, r, t))
        s = result["S_fact"]
        scores.append(s)
        labels.append(label)
        records.append(([h, r, t], label, s))
        if verbose:
            rt = result.get("readable_triple")
            rt_str = str(rt) if rt else "(无可读)"
            print("    triple ({}, {}, {}) label={} -> S_fact={:.3f} {}".format(
                h, r, t, label, s, " [skip]" if result.get("skipped") else ""))

    metrics = compute_metrics(labels, scores, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("事实验证智能体 — 错误知识检测指标（阈值 = {:.2f}）".format(args.threshold))
    print("=" * 60)
    print("  HIT (Accuracy) = {:.4f}".format(metrics["accuracy"]))
    print("  Precision      = {:.4f}".format(metrics["precision"]))
    print("  Recall         = {:.4f}".format(metrics["recall"]))
    print("  F1             = {:.4f}".format(metrics["f1"]))
    print("  AUC            = {:.4f}".format(metrics["auc"]))
    print("  (TP={}, TN={}, FP={}, FN={})".format(
        metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for triple, label, s in records:
                f.write(json.dumps(
                    {"triple": triple, "label": label, "S_fact": s},
                    ensure_ascii=False,
                ) + "\n")
        print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
