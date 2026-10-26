"""
TransE 结构表示模型。

评分函数：f(h, r, t) = ||h + r - t||_2
- 分数越低表示三元组越合理；分数越高表示越异常。
- 三元组置信度（结构嵌入部分）：S_struct_emb = sigmoid(-f(h,r,t))，分数越高越可信。
"""

import torch
import torch.nn as nn
from typing import Union, Tuple

# 支持 int / 1-D LongTensor / 2-D LongTensor
TripleBatch = Union[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class TransEModel(nn.Module):
    """
    TransE: 将实体和关系映射到同一向量空间，用 h + r ≈ t 的平移假设评分。
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.ent_emb = nn.Embedding(num_entities, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        # 初始化：Xavier
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def _get_embeddings(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """取 (h, r, t) 的嵌入向量，支持 0 维或 1 维 LongTensor。"""
        h_e = self.ent_emb(h)  # (*, dim)
        r_e = self.rel_emb(r)
        t_e = self.ent_emb(t)
        return h_e, r_e, t_e

    def score_triple(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算 TransE 评分：f(h,r,t) = ||h + r - t||_2（L2 范数）。
        """
        h_e, r_e, t_e = self._get_embeddings(h, r, t)
        diff = h_e + r_e - t_e
        return torch.norm(diff, p=2, dim=-1)

    def forward(
        self,
        pos_h: torch.Tensor,
        pos_r: torch.Tensor,
        pos_t: torch.Tensor,
        neg_h: torch.Tensor,
        neg_r: torch.Tensor,
        neg_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练用：同时计算正样本和负样本的分数，供 margin ranking loss 使用。
        """
        pos_score = self.score_triple(pos_h, pos_r, pos_t)
        neg_score = self.score_triple(neg_h, neg_r, neg_t)
        return pos_score, neg_score
