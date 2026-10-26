import sys
from pathlib import Path

# 将项目根加入路径，便于从 tests/ 运行时找到 ccg_models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ccg_models.gcn_encoder import (
    extract_k_hop_subgraph,
    AdjacencyBuilder,
    RelationalSubgraphEncoder,
    encode_triple_subgraph,
    SubgraphScorer
)

# 假设KG
triples = [
    (0,0,1),
    (1,1,2),
    (2,0,3),
    (3,2,4),
    (4,1,5),
]

target = (0,0,1)

num_entities = 10
num_relations = 3
ent_dim = 32

# 模拟embedding
entity_emb = torch.randn(num_entities, ent_dim)
rel_emb = torch.randn(num_relations, ent_dim)

# 1 提取子图
nodes, edges = extract_k_hop_subgraph(triples, target, k=2)

print("nodes:", nodes)
print("edges:", edges)

# 2 build graph
builder = AdjacencyBuilder()
X, edge_index, edge_type, node2id = builder.build(nodes, edges, entity_emb)

print("X shape:", X.shape)
print("edge_index:", edge_index.shape)

# 3 encoder
encoder = RelationalSubgraphEncoder(
    ent_dim=ent_dim,
    hidden_dim=64,
    num_relations=num_relations
)

H = encoder(X, edge_index, edge_type)

print("H:", H.shape)

# 4 triple vector
triple_vec = encode_triple_subgraph(
    encoder,
    X,
    edge_index,
    edge_type,
    node2id,
    target,
    rel_emb
)

print("triple_vec:", triple_vec.shape)

# 5 score
scorer = SubgraphScorer(triple_vec.shape[0])

score = scorer(triple_vec)

print("score:", score)