"""
子图编码层（论文级）：从目标三元组 (h,r,t) 提取 k-hop 子图，经 RGCN + 图注意力 + 残差得到节点表示，
再聚合成三元组向量 S_struct_gcn。
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 三元组：(头实体 id, 关系 id, 尾实体 id)
TripleId = Tuple[int, int, int]
# 子图中的边：(头节点 id, 关系 id, 尾节点 id)，节点 id 为全局实体 id
SubEdge = Tuple[int, int, int]


class SubgraphExtractor:
    """
    从全 KG 中提取目标三元组 (h,r,t) 的 k-hop 邻域子图。
    输出子图节点列表与边列表，供 AdjacencyBuilder 使用。
    """

    def __init__(self, k: int = 2) -> None:
        self.k = k

    def extract(
        self,
        triples: List[TripleId],
        target_triple: TripleId,
    ) -> Tuple[List[int], List[SubEdge]]:
        """
        以目标 (h, r, t) 的 h、t 为起点，在无向邻域上做 k 跳 BFS。
        返回 sub_nodes（子图内实体 id 列表）、sub_edges（(h_ent, r_ent, t_ent) 列表）。
        """
        h, r, t = target_triple
        adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for h_i, r_i, t_i in triples:
            adj[h_i].append((r_i, t_i))
            adj[t_i].append((r_i, h_i))

        visited = {h, t}
        frontier = {h, t}
        sub_edges: List[SubEdge] = []

        for _ in range(self.k):
            new_frontier = set()
            for node in frontier:
                for r_i, nbr in adj.get(node, []):
                    sub_edges.append((node, r_i, nbr))
                    if nbr not in visited:
                        visited.add(nbr)
                        new_frontier.add(nbr)
            frontier = new_frontier

        return list(visited), sub_edges


def extract_k_hop_subgraph(
    triples: List[TripleId],
    target_triple: TripleId,
    k: int = 2,
) -> Tuple[List[int], List[SubEdge]]:
    """便捷函数：等价于 SubgraphExtractor(k=k).extract(triples, target_triple)。"""
    return SubgraphExtractor(k=k).extract(triples, target_triple)


def build_adj_from_triples(triples: List[TripleId]) -> Dict[int, List[Tuple[int, int]]]:
    """从三元组列表构建邻接表（无向），供 extract_k_hop_subgraph_from_adj 复用，避免每条样本重复建图。"""
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for h_i, r_i, t_i in triples:
        adj[h_i].append((r_i, t_i))
        adj[t_i].append((r_i, h_i))
    return dict(adj)


def extract_k_hop_subgraph_from_adj(
    adj: Dict[int, List[Tuple[int, int]]],
    target_triple: TripleId,
    k: int = 2,
    max_nodes: int = 2000,
) -> Tuple[List[int], List[SubEdge]]:
    """在预构建的邻接表上做 k 跳 BFS；超过 max_nodes 后不再扩展，避免大子图导致 MemoryError。"""
    h, r, t = target_triple
    visited = {h, t}
    frontier = {h, t}
    sub_edges: List[SubEdge] = []

    for _ in range(k):
        if len(visited) >= max_nodes:
            break
        new_frontier = set()
        for node in frontier:
            for r_i, nbr in adj.get(node, []):
                sub_edges.append((node, r_i, nbr))
                if nbr not in visited:
                    if len(visited) < max_nodes:
                        visited.add(nbr)
                        new_frontier.add(nbr)
        frontier = new_frontier

    return list(visited), sub_edges

class AdjacencyBuilder:
    """
    将子图 (sub_nodes, sub_edges) 与实体嵌入转为 RGCN/GAT 所需格式：
    - X: 节点特征 (N, ent_dim)，来自 Entity Embedding
    - edge_index: (2, E)，COO 格式 [src, dst]，局部节点 id (0..N-1)
    - edge_type: (E,)，每条边的关系 id（全局）
    - node2id: 全局实体 id -> 局部 id
    """

    def build(
        self,
        sub_nodes: List[int],
        sub_edges: List[SubEdge],
        entity_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
        """
        返回 X (N, ent_dim), edge_index (2, E), edge_type (E,), node2id。
        """
        node2id: Dict[int, int] = {n: i for i, n in enumerate(sub_nodes)}
        n = len(sub_nodes)
        device = entity_emb.device
        dtype_emb = entity_emb.dtype

        # 只保留两端都在 sub_nodes 内的边，并转为局部 id
        src_list: List[int] = []
        dst_list: List[int] = []
        type_list: List[int] = []
        for h_ent, r_ent, t_ent in sub_edges:
            if h_ent in node2id and t_ent in node2id:
                src_list.append(node2id[h_ent])
                dst_list.append(node2id[t_ent])
                type_list.append(r_ent)

        edge_index = torch.tensor(
            [src_list, dst_list],
            dtype=torch.long,
            device=device,
        )
        edge_type = torch.tensor(type_list, dtype=torch.long, device=device)

        idx = torch.tensor(sub_nodes, dtype=torch.long, device=device)
        X = entity_emb[idx].to(dtype_emb)

        return X, edge_index, edge_type, node2id


class RelationalGraphConvLayer(nn.Module):
    """
    RGCN关系感知图卷积：每种关系使用独立权重 W_r，消息按关系类型与目标节点入度归一化 (1/c_{i,r})，
    自环用 W_0。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # 每种关系一个权重矩阵 W_r: (in_dim, out_dim)
        self.rel_weights = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        nn.init.xavier_uniform_(self.rel_weights)
        self.self_loop = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        X: (N, in_dim), edge_index: (2, E), edge_type: (E,)
        返回: (N, out_dim)
        """
        N = X.size(0)
        device = X.device
        out = torch.zeros(N, self.out_dim, device=device, dtype=X.dtype)

        src, dst = edge_index[0], edge_index[1]
        # 只遍历子图中出现的关系类型，避免 num_relations 很大时无效循环
        unique_rels = edge_type.unique()

        for r in unique_rels:
            r = r.item()
            mask = edge_type == r

            r_src = src[mask]
            r_dst = dst[mask]
            h = X[r_src]  # (E_r, in_dim)
            W = self.rel_weights[r]  # (in_dim, out_dim)
            msg = h @ W  # (E_r, out_dim)

            # 按目标节点归一化: 1/c_{i,r}，c_{i,r} = 目标节点 i 在关系 r 上的入度
            c = torch.zeros(N, device=device, dtype=torch.float32)
            c.scatter_add_(0, r_dst, torch.ones_like(r_dst, dtype=torch.float32))
            norm = c[r_dst].clamp(min=1e-6).unsqueeze(1)
            msg = msg / norm

            out.index_add_(0, r_dst, msg)

        out = out + self.self_loop(X)
        return F.relu(out)


class GraphAttentionLayer(nn.Module):
    """
    图注意力：对每条边 (i,j) 计算注意力 α_ij，按目标节点 j 做 softmax，再聚合邻居表示。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.randn(2 * out_dim) * 0.01)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        X: (N, in_dim), edge_index: (2, E)
        返回: (N, out_dim)，每个节点聚合邻居后的表示。
        """
        src, dst = edge_index[0], edge_index[1]
        h_src = self.W(X[src])  # (E, out_dim)
        h_dst = self.W(X[dst])  # (E, out_dim)
        cat = torch.cat([h_src, h_dst], dim=1)  # (E, 2*out_dim)
        e = self.leaky(cat @ self.att)  # (E,)

        # 按目标节点做 softmax：同一 dst 的边之间归一化
        alpha = _segment_softmax(e, dst)

        out = torch.zeros(X.size(0), h_src.size(1), device=X.device, dtype=X.dtype)
        out.index_add_(0, dst, alpha.unsqueeze(1) * h_src)
        return out


def _segment_softmax(logits: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """按 segment_ids 分组做 softmax，返回与 logits 同形的权重。"""
    if segment_ids.numel() == 0:
        return logits
    n_seg = int(segment_ids.max().item()) + 1
    max_per_seg = torch.full((n_seg,), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_per_seg.scatter_reduce_(0, segment_ids, logits, reduce="amax")
    logits_stable = logits - max_per_seg[segment_ids]
    exp_logits = torch.exp(logits_stable.clamp(max=20.0))
    sum_per_seg = torch.zeros(n_seg, device=logits.device, dtype=logits.dtype)
    sum_per_seg.index_add_(0, segment_ids, exp_logits)
    alpha = exp_logits / sum_per_seg[segment_ids].clamp(min=1e-6)
    return alpha


class RelationalSubgraphEncoder(nn.Module):
    """
    子图编码器：节点初始化(Entity Embedding) → RGCN → RGCN+残差 → Graph Attention → 节点表示。
    用于后续 Readout（取 h、t 的表示与 Relation Embedding 拼接成三元组表示）。
    """

    def __init__(
        self,
        ent_dim: int,
        hidden_dim: int,
        num_relations: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.rgcn1 = RelationalGraphConvLayer(ent_dim, hidden_dim, num_relations)
        self.rgcn2 = RelationalGraphConvLayer(hidden_dim, hidden_dim, num_relations)
        self.att = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        X: (N, ent_dim), edge_index: (2, E), edge_type: (E,)
        返回 H: (N, hidden_dim)。
        """
        H = self.rgcn1(X, edge_index, edge_type)
        H = self.dropout(H)
        H2 = self.rgcn2(H, edge_index, edge_type)
        H = H2 + H  # Residual
        H = self.att(H, edge_index)
        return H


def encode_triple_subgraph(
    encoder: RelationalSubgraphEncoder,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    node2id: Dict[int, int],
    triple: TripleId,
    rel_emb: torch.Tensor,
) -> torch.Tensor:
    """
    将子图编码为三元组表示向量：[h_vec; r_vec; t_vec]。

    - encoder 输出节点表示 H；
    - 取目标 h、t 在子图中的表示与关系 r 的嵌入，拼接为 triple_repr，供后续 MLP 得到 S_struct_gcn。
    """
    h, r, t = triple
    H = encoder(X, edge_index, edge_type)
    h_vec = H[node2id[h]]
    t_vec = H[node2id[t]]

    r_idx = torch.as_tensor(r, device=X.device, dtype=torch.long)
    if isinstance(rel_emb, nn.Module):
        r_vec = rel_emb(r_idx).squeeze(0)
    else:
        r_vec = rel_emb[r_idx].squeeze(0)

    triple_repr = torch.cat([h_vec, r_vec, t_vec], dim=0)
    return triple_repr


class SubgraphScorer(nn.Module):
    """将 triple_repr 经 MLP 映射为标量，sigmoid 后得 S_struct_gcn ∈ (0,1)。"""

    def __init__(self, triple_vec_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(triple_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, triple_vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(triple_vec).squeeze(-1)
