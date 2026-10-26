"""
事实验证智能体在 FB15k-237 上的运行示例。
适合先跑少量样本（默认 5 条）验证 LLM 是否正常执行，再跑带噪声实验。

运行前请确保：
1. 数据与映射：CC_gra/data/FB15k 下存在 fb15k237-train/valid/test.txt 及 entity2text.txt、relation2text.txt
2. 本地 Qwen：已安装 transformers，模型会懒加载；若加载失败则退回 MockLLMClient

在项目根目录 CC_gra 下执行：
  python examples/run_fact_agent_fb15k237.py
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.text_mapping import load_fb15k237_text_maps
from ccg_agents.llm_client import LocalQwenClient, MockLLMClient
from ccg_agents.fact_agent import FactVerificationAgent, FactAgentConfig
from ccg_pipeline.train_embeddings import load_config


def _create_llm_client():
    """
    使用本地 Qwen2.5 优先；若 transformers 未安装或加载失败，则退回 Mock。
    模型名可通过环境变量 LOCAL_LLM_MODEL 覆盖（如 Qwen/Qwen2.5-1.8B-Instruct 以节省显存）。
    可通过 CUDA_VISIBLE_DEVICES 环境变量指定使用的 GPU（如 CUDA_VISIBLE_DEVICES=1）。
    """
    model_name = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    # 如果设置了 CUDA_VISIBLE_DEVICES，使用 "cuda:0"（逻辑 GPU 0 对应物理 GPU）
    # 否则使用 "auto" 让 transformers 自动选择
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        device_map = "cuda:0"  # 设置了 CUDA_VISIBLE_DEVICES 后，逻辑 GPU 0 就是指定的物理 GPU
    else:
        device_map = "auto"
    try:
        client = LocalQwenClient(
            model_name=model_name,
            device_map=device_map,
        )
        return client, f"本地 {model_name}"
    except Exception:
        return MockLLMClient(), None


def main() -> None:
    dataset_name = "fb15k237"
    # 仅测试时用：只跑前 N 条三元组，设为 None 或 0 则跑全部（慎用，数据量大时很慢）
    NUM_TEST_SAMPLES = 5

    # 1) 加载配置与数据
    cfg = load_config()
    ds_cfg = cfg.get(dataset_name) or cfg.get("fb15k237")
    if not ds_cfg:
        print("未找到 fb15k237 配置，请检查 ccg_config/datasets.yaml 或 model 配置。")
        return

    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    train_path = _ROOT / ds_cfg["train_path"].lstrip("./")
    valid_path = _ROOT / ds_cfg["valid_path"].lstrip("./")
    test_path = _ROOT / ds_cfg["test_path"].lstrip("./")

    if not train_path.exists():
        print(f"训练集不存在: {train_path}")
        return

    ds = KGDataset.load_from_files(
        str(train_path), str(valid_path), str(test_path)
    )
    train_triples = ds.train_triples

    # 2) 加载实体/关系可读映射（data/FB15k 下）
    data_root = _ROOT / "data" / "FB15k"
    try:
        ent2text, rel2text = load_fb15k237_text_maps(root=str(data_root))
        print(f"已加载实体映射 {len(ent2text)} 条、关系映射 {len(rel2text)} 条。")
    except FileNotFoundError as e:
        print(f"映射文件缺失: {e}")
        print("事实验证智能体将使用默认分数（无可读表示时跳过）。")
        ent2text, rel2text = {}, {}

    # 3) LLM 客户端：本地 Qwen 优先，失败则 Mock
    client, client_src = _create_llm_client()
    if client_src:
        print(f"使用 {client_src} 进行事实验证。")
    else:
        print("本地 Qwen 加载失败，使用 MockLLMClient 跑通流程。")

    # 4) 事实验证智能体
    config = FactAgentConfig(
        default_score_no_readable=0.5,
        skip_fact_when_no_readable=True,
        max_tokens=300,
    )
    save_path = base / "fact_scores_fb15k237.jsonl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    agent = FactVerificationAgent(
        llm_client=client,
        id2entity=ds.id2entity,
        id2relation=ds.id2relation,
        ent2text=ent2text,
        rel2text=rel2text,
        config=config,
        save_result_path=str(save_path),
    )

    # 5) 对若干三元组打分（测试时只跑前 NUM_TEST_SAMPLES 条）
    n_total = len(train_triples)
    if NUM_TEST_SAMPLES and NUM_TEST_SAMPLES > 0:
        n_run = min(NUM_TEST_SAMPLES, n_total)
        triples_to_run = train_triples[:n_run]
        print(f"\n仅测试前 {n_run} 条三元组（共 {n_total} 条）。要跑全部请将 NUM_TEST_SAMPLES 设为 0 或 None。")
    else:
        triples_to_run = train_triples
        print(f"\n对全部 {n_total} 条三元组进行事实验证。")

    print("\n事实验证智能体示例输出：")
    print("-" * 60)
    for triple in triples_to_run:
        result = agent.score_triple(triple)
        h, r, t = triple
        head_str = ds.id2entity.get(h, f"e{h}")
        rel_str = ds.id2relation.get(r, f"r{r}")
        tail_str = ds.id2entity.get(t, f"e{t}")
        print(f"三元组(ID): ({head_str}, {rel_str}, {tail_str})")
        if result.get("readable_triple"):
            rh, rr, rt = result["readable_triple"]
            print(f"  可读形式: ({rh}, {rr}, {rt})")
        print(f"  S_fact = {result['S_fact']:.4f}")
        if result.get("skipped"):
            print("  [已跳过事实核查]")
        if result.get("explanation"):
            expl = result["explanation"]
            print(f"  说明: {expl[:200]}{'...' if len(expl) > 200 else ''}")
        print()
    print("-" * 60)
    print(f"共处理 {len(triples_to_run)} 条，结果已写入: {save_path}")
    print("融合决策智能体可读取该 JSONL，结合 S_fact 与 explanation 做综合决策。")


if __name__ == "__main__":
    main()
