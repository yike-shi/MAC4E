"""
端到端多智能体融合前：对各路智能体分数按「本 run 全部样本」做列归一化，使结构/事实/逻辑可比后再加权。

- none: 不归一化
- minmax: 每列单独 min-max 到 [0,1]（常数列退化为 0.5）
- zscore: 每列 z-score 后经标准正态 CDF 映到 [0,1]（示例端到端脚本默认此项）
"""

from __future__ import annotations

import math
from typing import List, Sequence


def normalize_score_channel(values: Sequence[float], method: str) -> List[float]:
    n = len(values)
    if n == 0 or method == "none":
        return [float(v) for v in values]
    vals = [float(v) for v in values]
    if method == "minmax":
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span < 1e-12:
            return [0.5] * n
        return [(v - lo) / span for v in vals]
    if method == "zscore":
        mu = sum(vals) / n
        if n == 1:
            return [0.5]
        var = sum((v - mu) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(max(var, 0.0))
        if std < 1e-12:
            return [0.5] * n
        out: List[float] = []
        for v in vals:
            z = (v - mu) / std
            p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            out.append(max(0.0, min(1.0, p)))
        return out
    raise ValueError(f"未知 agent_score_norm: {method}")
