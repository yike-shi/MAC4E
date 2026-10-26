"""
融合决策智能体调用示例：演示如何将任务分配权重与结构/事实/逻辑三路结果交给融合决策智能体，
得到最终置信度分数 S_fusion 与综合理由。

用法（在项目根目录 CC_gra 下）：
  python examples/run_fusion_agent_example.py
  python examples/run_fusion_agent_example.py --no_llm   # 仅用加权平均作为置信度，不调 LLM
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_agents.llm_client import LocalQwenClient, MockLLMClient
from ccg_agents.fusion_agent import FusionDecisionAgent, FusionAgentConfig


def _create_llm_client():
    """与 run_fact_agent_fb15k237 一致：本地 Qwen 优先，失败则 Mock。"""
    model_name = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_map = "cuda:0" if cuda_visible is not None else "auto"
    try:
        client = LocalQwenClient(model_name=model_name, device_map=device_map)
        return client, f"本地 {model_name}"
    except Exception:
        return MockLLMClient(), None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="融合决策智能体示例")
    parser.add_argument("--no_llm", action="store_true", help="不调用 LLM，仅用加权平均作为置信度")
    parser.add_argument("--out", type=str, default=None, help="可选：融合结果 JSONL 路径")
    args = parser.parse_args()

    # 1) LLM 客户端（--no_llm 时也会创建，但 FusionAgentConfig.use_llm=False 不会真正调用）
    llm_client, client_src = _create_llm_client()
    if client_src:
        print(f"LLM 客户端: {client_src}")
    else:
        print("LLM 客户端: Mock（未配置或加载失败）")

    # 2) 融合决策智能体
    config = FusionAgentConfig(
        use_llm=not args.no_llm,
        max_tokens=350,
    )
    agent = FusionDecisionAgent(
        llm_client=llm_client,
        config=config,
        save_result_path=args.out,
    )

    # 3) 模拟一条三元组的三路输出 + 分配权重，调用融合决策
    allocator_weights = [0.2, 0.5, 0.3]  # 示例：本三元组更依赖事实与逻辑
    structure_result = {
        "S_struct": 0.35,
        "explanation": "嵌入空间中 head+rel 与 tail 距离较大，结构置信度偏低。",
    }
    fact_result = {
        "S_fact": 0.82,
        "explanation": "该关系在现实世界中常见，头尾实体类型与关系匹配，无明显矛盾。",
    }
    logic_result = {
        "S_logic": 0.68,
        "explanation": "图中存在 2 跳路径支持该关系，与目标陈述无逻辑冲突。",
    }
    readable_triple = ("美国", "首都", "华盛顿哥伦比亚特区")

    result = agent.decide(
        allocator_weights,
        structure_result,
        fact_result,
        logic_result,
        readable_triple=readable_triple,
    )

    # 4) 打印结果
    print("\n" + "=" * 60)
    print("融合决策结果")
    print("=" * 60)
    print(f"  可读三元组: {readable_triple}")
    print(f"  分配权重   : 结构={allocator_weights[0]:.2f}, 事实={allocator_weights[1]:.2f}, 逻辑={allocator_weights[2]:.2f}")
    print(f"  加权分数   : {result['weighted_score']:.4f}")
    print(f"  最终置信度 : {result['final_score']:.4f}")
    #print(f"  是否用 LLM : {result['used_llm']}")
    print(f"  综合理由   : {result['explanation'][:200]}{'...' if len(result['explanation']) > 200 else ''}")
    print("=" * 60)
    if args.out:
        print(f"结果已追加写入: {args.out}")


if __name__ == "__main__":
    main()
