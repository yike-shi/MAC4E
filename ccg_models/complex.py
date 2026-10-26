"""
ComplEx 结构表示模型。

- 实体与关系用复向量表示：e = e_re + i * e_im，r = r_re + i * r_im。
- 评分函数：f(h, r, t) = Re(<h, r, t̄>)（复内积的实部），分数越高表示三元组越合理。
- 结构置信度：S_struct_emb = sigmoid(f(h,r,t))，与 TransE 的 sigmoid(-distance) 一起可融合为多视角结构置信度。
"""

import torch
import torch.nn as nn
from typing import Tuple, Union

TripleBatch = Union[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def complex_struct_confidence(score: torch.Tensor) -> torch.Tensor:
    """
    ComplEx 结构置信度：分数越高越可信，映射到 (0, 1)。
    与 TransE 的 sigmoid(-score) 对应，便于融合 S_struct = α * S_transe + (1-α) * S_complex。
    """
    return torch.sigmoid(score)


class ComplExModel(nn.Module):
    """
    ComplEx: 实体与关系在复向量空间中的双线性模型，Re(<h, r, t̄>)。
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.ent_re = nn.Embedding(num_entities, embedding_dim)
        self.ent_im = nn.Embedding(num_entities, embedding_dim)
        self.rel_re = nn.Embedding(num_relations, embedding_dim)
        self.rel_im = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.ent_re.weight)
        nn.init.xavier_uniform_(self.ent_im.weight)
        nn.init.xavier_uniform_(self.rel_re.weight)
        nn.init.xavier_uniform_(self.rel_im.weight)

    def score_triple(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算 ComplEx 评分：Re(<h, r, t̄>)。
        h, r, t: (*,) LongTensor，支持单条或 batch。
        返回: (*,) 实值分数，越高表示三元组越合理。
        """
        h_re = self.ent_re(h)
        h_im = self.ent_im(h)
        r_re = self.rel_re(r)
        r_im = self.rel_im(r)
        t_re = self.ent_re(t)
        t_im = self.ent_im(t)

        # Re(<h, r, t̄>) = h_re*r_re*t_re + h_im*r_re*t_im + h_re*r_im*t_im - h_im*r_im*t_re
        score = (
            h_re * r_re * t_re
            + h_im * r_re * t_im
            + h_re * r_im * t_im
            - h_im * r_im * t_re
        )
        return torch.sum(score, dim=-1)

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
        训练用：正样本与负样本的分数，供 margin ranking loss 使用。
        ComplEx 希望 pos_score 高、neg_score 低，loss 常取 margin - pos_score + neg_score。
        """
        pos_score = self.score_triple(pos_h, pos_r, pos_t)
        neg_score = self.score_triple(neg_h, neg_r, neg_t)
        return pos_score, neg_score
