"""
为 (h,r,t) 构造特征向量，供 Allocator MLP 使用。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple, List, Optional

import torch

from ccg_data.type_constraints import TypeConstraints
from ccg_data.path_mining import count_support_paths
from ccg_models.transE import TransEModel

TripleId = Tuple[int, int, int]

# ComplEx 与 complex_struct_confidence 按需导入，避免 train_allocator 仅用 transe 时强依赖
def _get_complex_module():
    from ccg_models.complex import ComplExModel, complex_struct_confidence
    return ComplExModel, complex_struct_confidence


def build_entity_frequency(triples: List[TripleId]) -> Dict[int, int]:
    """
    统计实体在三元组中出现的频率（头或尾都算），用于后续构造特征。
    返回：entity_id -> 计数。
    """
    freq: Dict[int, int] = {}
    for h, _, t in triples:
        freq[h] = freq.get(h, 0) + 1
        freq[t] = freq.get(t, 0) + 1
    return freq


def _normalize_log_freq(freq: int, max_freq: int) -> float:
    """
    对实体频率做 log 归一化
    """
    if max_freq <= 0:
        return 0.0
    return math.log1p(freq) / math.log1p(max_freq)


def build_feature_vector_for_triple(
    triple: TripleId,
    model: TransEModel,
    device: torch.device,
    type_constraints: TypeConstraints,
    graph,
    entity_freq: Dict[int, int],
    use_sigmoid_score: bool = False,
    max_path_len: int = 3,
    complex_model: Optional[Any] = None,
    emb_mode: str = "transe",
    fusion_alpha: float = 0.5,
) -> List[float]:
    """
    为单条三元组 (h,r,t) 构造特征向量 x ∈ R^d，作为 Allocator 的输入。

    emb_mode: "transe" 仅用 TransE；"complex" 仅用 ComplEx（需传入 complex_model）；"fusion" 用 alpha*S_transe + (1-alpha)*S_complex 作为 S1（需传入 complex_model）。
    fusion_alpha: 仅当 emb_mode=="fusion" 时生效，S1 = fusion_alpha*S_transe + (1-fusion_alpha)*S_complex。

    包含特征：
    - raw_score: TransE 距离或 ComplEx 的负分数（统一为越低越不合理）
    - s_struct_emb: 结构置信度 S1（use_sigmoid_score）
    - support_paths, is_func, freq_h, freq_t
    """
    h, r, t = triple
    h_t = torch.tensor([h], dtype=torch.long, device=device)
    r_t = torch.tensor([r], dtype=torch.long, device=device)
    t_t = torch.tensor([t], dtype=torch.long, device=device)
    mode = (emb_mode or "transe").lower()

    # 1) 嵌入分数与 S1（按 emb_mode 选择 transe / complex / fusion）
    with torch.no_grad():
        transe_raw = model.score_triple(h_t, r_t, t_t)  # TransE: 距离，越低越好
        s1_transe = float(torch.sigmoid(-transe_raw).item())
        raw_score: float = float(transe_raw.item())

    if mode == "complex":
        if complex_model is None:
            raise ValueError("emb_mode='complex' 需传入 complex_model")
        ComplExModel, complex_struct_confidence = _get_complex_module()
        comp_score = complex_model.score_triple(h_t, r_t, t_t)
        s1 = float(complex_struct_confidence(comp_score).item())
        raw_score = float(-comp_score.item())  # 统一为“越低越差”
    elif mode == "fusion":
        if complex_model is None:
            raise ValueError("emb_mode='fusion' 需传入 complex_model")
        ComplExModel, complex_struct_confidence = _get_complex_module()
        comp_score = complex_model.score_triple(h_t, r_t, t_t)
        s1_complex = float(complex_struct_confidence(comp_score).item())
        s1 = fusion_alpha * s1_transe + (1.0 - fusion_alpha) * s1_complex
        # raw_score 保持 TransE，特征维度不变
    else:
        s1 = s1_transe

    # 对 S1 做缩放：原始分布极度挤在 0 附近，使用 log1p(scale * S1) 拉开区分度
    # 例如 S1 ∈ [0, 0.01] 映射到大约 [0, ~4.6]，交给后续标准化处理
    s1 = math.log1p(s1 * 100.0)

    features: List[float] = [raw_score]
    if use_sigmoid_score:
        features.append(s1)

    # 2) 支持路径数量：原始计数可能极大，先 log1p 再交给标准化，避免单一特征主导
    raw_paths = float(count_support_paths(graph, h, r, t, max_len=max_path_len))
    support_paths = math.log1p(raw_paths)
    features.append(support_paths)

    # 3) 关系功能性
    is_func = 1.0 if type_constraints.is_functional(r) else 0.0
    features.append(is_func)

    # 4) 实体频率（低频实体通常更可疑）：log 归一化
    #    先找出频率最大值做归一化
    max_freq = max(entity_freq.values()) if entity_freq else 1
    freq_h = _normalize_log_freq(entity_freq.get(h, 0), max_freq)
    freq_t = _normalize_log_freq(entity_freq.get(t, 0), max_freq)
    features.append(freq_h)
    features.append(freq_t)

    return features

