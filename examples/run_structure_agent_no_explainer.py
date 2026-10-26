"""
结构智能体：对指定划分的 triples 打分并统计分布。

两步逻辑：
  第一步：跑结构智能体，对全部 triples 打分，得到结构评分结果表（triple + S_struct），可写入 JSONL。
  第二步：统计 S_struct 分布（mean、median、直方图），看模型是否有区分能力。

运行前请确保已训练 TransE，并存在 outputs/embeddings/{dataset}_transE.pt。
默认使用 FB15K-237。通过 --split 在 train / valid / test 三个划分上分别测试。

在项目根目录 CC_gra 下执行：
  python examples/run_structure_agent_no_explainer.py
  python examples/run_structure_agent_no_explainer.py --dataset fb15k237 --split train
  python examples/run_structure_agent_no_explainer.py --dataset fb15k237 --split valid
  python examples/run_structure_agent_no_explainer.py --dataset fb15k237 --split test
  指定输出路径：
  python examples/run_structure_agent_no_explainer.py --dataset fb15k237 --split train --out outputs/structure_scores_fb15k237_train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from ccg_data.kg_dataset import KGDataset
from ccg_data.path_mining import build_graph
from ccg_pipeline.train_embeddings import load_config, load_transe_from_checkpoint
from ccg_agents.structure_agent import StructureConstraintAgent, StructureAgentConfig

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def main() -> None:
    parser = argparse.ArgumentParser(description="结构智能体：打分 → 分布统计")
    parser.add_argument("--dataset", type=str, default="fb15k237", help="数据集名，如 fb15k237 / fb15k / wn18rr / nell995")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="数据划分：train / valid / test")
    parser.add_argument("--out", type=str, default=None, help="结构评分结果表路径，如 outputs/structure_scores_fb15k237_train.jsonl")
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载配置与数据
    cfg = load_config()
    ds_cfg = cfg.get(dataset_name)
    if not isinstance(ds_cfg, dict) or "train_path" not in ds_cfg:
        raise KeyError(f"配置中未找到数据集 {dataset_name}，或缺少 train_path")
    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")

    ds = KGDataset.load_from_files(str(train_path), str(valid_path), str(test_path))
    train_triples = ds.train_triples
    graph = build_graph(train_triples)

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

    if args.split == "train":
        triples = ds.train_triples
    elif args.split == "valid":
        triples = ds.valid_triples
    else:
        triples = ds.test_triples

    # ========== 第一步：跑结构智能体，得到结构评分结果表 ==========
    print(f"数据集: {dataset_name}, 划分: {args.split}, 条数: {len(triples)}")

    records = []
    for triple in tqdm(triples, desc="打分", unit="triple"):
        result = agent.score_triple(triple)
        records.append((triple, result))

    # 输出结果表：保存 JSONL（triple_id + S_struct）
    out_path = Path(args.out) if args.out else base / "structure_scores" / f"structure_scores_{dataset_name}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for triple, res in records:
            f.write(json.dumps({
                "triple": list(triple),
                "S_struct": res["S_struct"],
            }, ensure_ascii=False) + "\n")
    print(f"结构评分结果表已写入: {out_path}")

    # ========== 第二步：统计 S_struct 分布 ==========
    print("\n" + "=" * 60)
    print("S_struct 分布")

    s_struct_list = [r[1]["S_struct"] for r in records]
    n = len(s_struct_list)
    mean_s = sum(s_struct_list) / n if n else 0.0
    sorted_s = sorted(s_struct_list)
    mid = n // 2
    median_s = (sorted_s[mid] + sorted_s[mid - 1]) / 2.0 if n >= 2 and n % 2 == 0 else (sorted_s[mid] if n else 0.0)

    print(f"S_struct mean   = {mean_s:.4f}")
    print(f"S_struct median = {median_s:.4f}")

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("\n直方图：")
    for low, high in bins:
        count = sum(1 for x in s_struct_list if low <= x < high)
        pct = 100.0 * count / n if n else 0
        bar = "#" * int(pct / 2) + " " * (50 - int(pct / 2))
        print(f"  {low:.1f}-{high:.1f}  {pct:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
