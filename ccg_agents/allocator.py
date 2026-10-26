"""
任务分配智能体（Allocator）的 MLP 实现：
- 输入：为每个三元组构造的特征向量 x ∈ R^d（推理前须与训练时相同的 feat_mean/feat_std 标准化）
- 输出：长度为 3 的 log_softmax 或概率分布 [m_s, m_f, m_l]，三者和为 1
  分别表示：结构 Agent、事实 Agent、逻辑 Agent 的「调用倾向」。

训练时使用 log_softmax + KLDivLoss 与软标签 [m_s, m_f, m_l] 对齐（互斥分布），
避免 BCELoss 与 Softmax 混用导致的梯度冲突。
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class AllocatorMLP(nn.Module):
    """
    多层 MLP + BatchNorm + ReLU，最后一层 Linear 输出 logits，前向为 log_softmax(dim=-1)，
    供 KLDivLoss 训练；推理时概率为 exp(log_probs) 或 softmax(logits)。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[List[int], Tuple[int, ...]] = (64, 32),
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        dims = [in_dim] + list(hidden_dims) + [3]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 log_softmax(logits)，与 KLDivLoss 及互斥软标签一致。"""
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """推理用：与训练分布一致的概率 [m_s, m_f, m_l]。"""
        return self.forward(x).exp()

