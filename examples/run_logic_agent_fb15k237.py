"""
逻辑验证智能体在 FB15k-237 上的运行示例（集成 RGCN 子图编码分数）。

流程：
  1. 加载 FB15K-237 训练集，构建图（用于路径挖掘）；
  2. 加载 TransE 与 GCN/RGCN checkpoint，从训练图构造邻接表；
  3. 对若干三元组：
     - 提取 k-hop 子图，经 RGCN 编码 + MLP 得到 S_logic_rgcn；
     - 利用多跳路径 + LLM 得到 S_logic_llm；
     - 依据路径丰富度与 LLM 可靠性自适应融合，得到最终 S_logic。

运行前请确保：
1. 数据与映射：CC_gra/data/FB15k 下存在 fb15k237-*.txt 及 entity2text.txt、relation2text.txt
2. 编码器：已训练并保存 outputs/gcn/fb15k237_gcn.pt 与 outputs/embeddings/fb15k237_transE.pt
3. 本地 Qwen（transformers）；若加载失败则用 MockLLMClient，LLM 分支恒为默认分数

在项目根目录 CC_gra 下执行：
  python examples/run_logic_agent_fb15k237.py
  python examples/run_logic_agent_fb15k237.py  # 默认跑 5 条，用于验证 LLM + RGCN 集成是否正常
"""

import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.path_mining import build_graph
from ccg_data.text_mapping import load_fb15k237_text_maps
from ccg_models.gcn_encoder import (
    build_adj_from_triples,
    extract_k_hop_subgraph_from_adj,
    AdjacencyBuilder,
    RelationalSubgraphEncoder,
    encode_triple_subgraph,
    SubgraphScorer,
)
from ccg_agents.llm_client import LocalQwenClient, MockLLMClient
from ccg_agents.logic_agent import LogicVerificationAgent, LogicAgentConfig
from ccg_pipeline.train_embeddings import load_config, load_model_config, load_transe_from_checkpoint


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


def main() -> None:
    dataset_name = "fb15k237"
    NUM_TEST_SAMPLES = 5

    cfg = load_config()
    model_cfg = load_model_config()
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
    train_triples = ds.train_triples
    graph = build_graph(train_triples)

    # 加载 TransE 与 GCN/RGCN checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dir = base / "embeddings"
    transe_ckpt = embed_dir / f"{dataset_name}_transE.pt"
    if not transe_ckpt.exists():
        print(f"未找到 TransE 权重: {transe_ckpt}，请先训练 TransE 和 GCN。")
        return
    transe_model = load_transe_from_checkpoint(str(transe_ckpt), device=device)
    transe_model.eval()
    entity_emb = transe_model.ent_emb.weight.detach()
    rel_emb = transe_model.rel_emb

    gcn_dir = base / "gcn"
    gcn_ckpt = gcn_dir / f"{dataset_name}_gcn.pt"
    if not gcn_ckpt.exists():
        print(f"未找到 GCN/RGCN checkpoint: {gcn_ckpt}，请先运行 train_gcn_encoder.py。")
        return
    state = torch.load(gcn_ckpt, map_location=device)
    ent_dim = int(state["ent_dim"])
    hidden_dim = int(state["hidden_dim"])
    num_relations = int(state["num_relations"])
    k_hop = int(state.get("k_hop", model_cfg.get("gcn", {}).get("num_layers", 2)))

    encoder = RelationalSubgraphEncoder(
        ent_dim=ent_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        dropout=float(model_cfg.get("gcn", {}).get("dropout", 0.3)),
    ).to(device)
    scorer = SubgraphScorer(triple_vec_dim=hidden_dim * 2 + ent_dim).to(device)
    encoder.load_state_dict(state["encoder_state"])
    scorer.load_state_dict(state["scorer_state"])
    encoder.eval()
    scorer.eval()

    # 预构建训练图邻接表，供子图提取使用
    adj = build_adj_from_triples(train_triples)
    builder = AdjacencyBuilder()

    data_root = _ROOT / "data" / "FB15k"
    try:
        ent2text, rel2text = load_fb15k237_text_maps(root=str(data_root))
        print(f"已加载实体映射 {len(ent2text)} 条、关系映射 {len(rel2text)} 条。")
    except FileNotFoundError as e:
        print(f"映射文件缺失: {e}，逻辑智能体将使用 ID 或路径形式。")
        ent2text, rel2text = {}, {}

    client, client_src = _create_llm_client()
    if client_src:
        print(f"使用 {client_src} 进行逻辑推理。")
    else:
        print("本地 Qwen 加载失败，使用 MockLLMClient。")

    agent = LogicVerificationAgent(
        llm_client=client,
        graph=graph,
        id2entity=ds.id2entity,
        id2relation=ds.id2relation,
        ent2text=ent2text,
        rel2text=rel2text,
        config=LogicAgentConfig(max_path_len=3, max_paths=10, max_tokens=350),
        save_result_path=None,
    )

    n_total = len(train_triples)
    n_run = min(NUM_TEST_SAMPLES, n_total) if NUM_TEST_SAMPLES else n_total
    triples_to_run = train_triples[:n_run]
    if NUM_TEST_SAMPLES:
        print(f"\n仅测试前 {n_run} 条三元组（共 {n_total} 条）。")

    print("\n逻辑验证智能体示例输出：")
    print("-" * 60)
    for triple in triples_to_run:
        # 先用 GCN/RGCN 对该三元组在 k-hop 子图上的逻辑合理性打分
        h, r, t = triple
        sub_nodes, sub_edges = extract_k_hop_subgraph_from_adj(
            adj, triple, k=k_hop, max_nodes=int(model_cfg.get("gcn", {}).get("max_subgraph_nodes", 2000))
        )
        s_logic_rgcn: float | None
        if len(sub_nodes) >= 2 and len(sub_edges) > 0:
            X, edge_index, edge_type, node2id = builder.build(sub_nodes, sub_edges, entity_emb.to(device))
            with torch.no_grad():
                triple_vec = encode_triple_subgraph(
                    encoder,
                    X,
                    edge_index,
                    edge_type,
                    node2id,
                    triple,
                    rel_emb,
                )
                logit = scorer(triple_vec)
                s_logic_rgcn = torch.sigmoid(logit).item()
        else:
            s_logic_rgcn = None

        # 再调用逻辑智能体（路径 + LLM），并与 RGCN 分数加权融合
        result = agent.score_triple(triple, s_logic_rgcn=s_logic_rgcn)
        h, r, t = triple
        head_str = ds.id2entity.get(h, f"e{h}")
        rel_str = ds.id2relation.get(r, f"r{r}")
        tail_str = ds.id2entity.get(t, f"e{t}")
        print(f"三元组(ID): ({head_str}, {rel_str}, {tail_str})")
        print(f"  路径数(2+跳): {result.get('path_count', 0)}")
        if result.get("path_descriptions"):
            for i, p in enumerate(result["path_descriptions"][:3]):
                print(f"    路径{i+1}: {p[:80]}{'...' if len(p) > 80 else ''}")
        print(f"  S_logic_rgcn = {result.get('S_logic_rgcn')}")
        print(f"  S_logic_llm  = {result.get('S_logic_llm'):.4f}")
        print(f"  beta         = {result.get('beta'):.3f}")
        print(f"  S_logic(fused) = {result['S_logic']:.4f}")
        if result.get("explanation"):
            expl = result["explanation"]
            print(f"  说明: {expl[:180]}{'...' if len(expl) > 180 else ''}")
        print()
    print("-" * 60)


if __name__ == "__main__":
    main()
