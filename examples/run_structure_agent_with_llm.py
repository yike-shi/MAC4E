"""
结构智能体 + LLM 解释的完整对接示例。

运行前请确保：
1. 已训练 TransE 并存在 outputs/embeddings/fb15k_transE.pt（或修改 dataset_name）
2. 本地 Qwen（transformers）用于生成结构解释；若加载失败则使用 MockLLMClient

在项目根目录 CC_gra 下执行：
  python examples/run_structure_agent_with_llm.py
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from ccg_data.kg_dataset import KGDataset
from ccg_data.path_mining import build_graph
from ccg_pipeline.train_embeddings import load_config, load_transe_from_checkpoint
from ccg_agents.llm_client import LocalQwenClient, MockLLMClient
from ccg_agents.structure_explainer import make_llm_structure_explainer
from ccg_agents.structure_agent import StructureConstraintAgent, StructureAgentConfig


def main() -> None:
    dataset_name = "fb15k"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载配置与数据
    cfg = load_config()
    ds_cfg = cfg[dataset_name.lower()]
    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")

    ds = KGDataset.load_from_files(
        str(train_path), str(valid_path), str(test_path)
    )
    train_triples = ds.train_triples
    graph = build_graph(train_triples)

    # 2) 加载 TransE
    ckpt_path = base / "embeddings" / f"{dataset_name.lower()}_transE.pt"
    if not ckpt_path.exists():
        print(f"未找到 TransE 权重: {ckpt_path}，请先训练 TransE。")
        return
    transe = load_transe_from_checkpoint(str(ckpt_path), device=device)
    transe.eval()

    # 3) 构造 LLM 客户端与结构解释器（本地 Qwen 优先，失败则 Mock）
    model_name = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    try:
        client = LocalQwenClient(
            model_name=model_name,
            device_map="auto",
        )
        print(f"使用本地 {model_name} 生成结构解释。")
    except Exception:
        client = MockLLMClient()
        print("本地 Qwen 加载失败，使用 MockLLMClient 跑通流程。")

    explainer = make_llm_structure_explainer(client, max_tokens=200)

    # 4) 保存路径：供融合智能体读取
    save_path = base / "structure_explanations.jsonl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 5) 创建结构智能体（接入 LLM 解释 + 落盘）
    agent = StructureConstraintAgent(
        transe_model=transe,
        graph=graph,
        llm_explainer=explainer,
        id2entity=ds.id2entity,
        id2relation=ds.id2relation,
        save_explanation_path=str(save_path),
        config=StructureAgentConfig(emb_mode="transe"),
    )

    # 6) 对若干三元组打分并生成解释
    triples_to_run = [
        train_triples[0],
        train_triples[len(train_triples) // 2],
    ]
    print("\n结构智能体 + LLM 解释 示例输出：")
    print("-" * 60)
    for triple in triples_to_run:
        result = agent.score_triple(triple)
        h, r, t = triple
        head = ds.id2entity.get(h, f"e{h}")
        rel = ds.id2relation.get(r, f"r{r}")
        tail = ds.id2entity.get(t, f"e{t}")
        print(f"三元组: ({head}, {rel}, {tail})")
        print(f"  S_struct={result['S_struct']:.4f}")
        if result.get("explanation"):
            print(f"  解释: {result['explanation']}")
        print()
    print("-" * 60)
    print(f"解释已追加写入: {save_path}")
    print("融合智能体可逐行读取该 JSONL，结合 S_struct 与 explanation 做综合决策。")


if __name__ == "__main__":
    main()
