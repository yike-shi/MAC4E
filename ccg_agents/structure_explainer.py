"""
结构验证智能体的可解释性模块：基于大语言模型对结构判断结果生成自然语言解释，
供融合决策智能体与错误分析使用。支持单条解释与批量生成。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

TripleId = Tuple[int, int, int]
ScoresDict = Dict[str, float]
Id2Str = Dict[int, str]


def build_structure_explanation_prompt(
    triple: TripleId,
    scores: ScoresDict,
    id2entity: Id2Str,
    id2relation: Id2Str,
    *,
    rank_in_batch: Optional[str] = None,
    embedding_similarity_context: Optional[str] = None,
) -> str:
    """
    构造面向 LLM 的提示：用实体/关系名称（非 ID）、S_struct 及可选上下文，
    请求一句可解释说明（高分解释为何合理，低分解释可能原因），便于错误分析。
    """
    h, r, t = triple
    head_str = id2entity.get(h, f"实体_{h}")
    rel_str = id2relation.get(r, f"关系_{r}")
    tail_str = id2entity.get(t, f"实体_{t}")

    s_struct = scores.get("S_struct", 0.0)

    rank_str = rank_in_batch or scores.get("rank_in_batch")
    sim_str = embedding_similarity_context or scores.get("embedding_similarity_context")

    prompt = f"""你是一个知识图谱专家，请对以下三元组给出可解释说明。

Triple: ({head_str}, {rel_str}, {tail_str})
结构置信度 S_struct={s_struct:.2f}（越接近 1 越合理，越接近 0 越可疑）
"""
    if rank_str:
        prompt += f"排序信息: {rank_str}\n"
    if sim_str:
        prompt += f"嵌入上下文: {sim_str}\n"

    prompt += """
请生成一句自然语言说明（英文）：
- 若分数高，简要解释为何该三元组在嵌入空间上合理；
- 若分数低，简要解释可能原因（如向量不匹配、不常见组合、可能错误）；
- 简洁、清楚，可用于错误分析。不要重复上述数值。"""

    return prompt


class LLMClientProtocol(Protocol):
    """大语言模型客户端协议：需实现 generate(prompt, **kwargs) 返回模型生成的文本。"""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


def make_llm_structure_explainer(
    client: LLMClientProtocol,
    *,
    max_tokens: int = 200,
) -> Callable[[TripleId, ScoresDict, Id2Str, Id2Str], str]:
    """
    根据具备 generate(prompt) 接口的 LLM 客户端，构造结构解释器 callable。
    返回的函数签名：(triple, scores, id2entity, id2relation) -> explanation_str。
    可用于 StructureConstraintAgent 的 llm_explainer 参数。
    """
    def explain(
        triple: TripleId,
        scores: ScoresDict,
        id2entity: Id2Str,
        id2relation: Id2Str,
    ) -> str:
        prompt = build_structure_explanation_prompt(
            triple, scores, id2entity, id2relation
        )
        out = client.generate(prompt, max_tokens=max_tokens)
        return (out or "").strip()
    return explain


def generate_explanations_batch(
    client: LLMClientProtocol,
    triples_with_scores: List[Tuple[TripleId, ScoresDict]],
    id2entity: Id2Str,
    id2relation: Id2Str,
    *,
    max_tokens: int = 200,
    rank_list: Optional[List[str]] = None,
    delay_seconds: float = 0.0,
) -> List[str]:
    """
    对多条 (triple, scores) 批量生成 LLM 解释。按顺序逐条调用 generate，
    避免单条调用开销过大时可在外部做异步或分片；可选 delay_seconds 缓解限流。

    - triples_with_scores: [(triple, scores_dict), ...]
    - rank_list: 可选，与 triples_with_scores 等长，如 ["在 batch 中排名 1/50", ...]，会写入 prompt。
    - delay_seconds: 每条调用后的休眠秒数，0 表示不等待。
    """
    n = len(triples_with_scores)
    rank_list = rank_list if rank_list is not None else [None] * n
    if len(rank_list) != n:
        rank_list = [None] * n

    out: List[str] = []
    for i, ((triple, scores), rank_str) in enumerate(zip(triples_with_scores, rank_list)):
        if rank_str is not None:
            scores = dict(scores)
            scores["rank_in_batch"] = rank_str
        prompt = build_structure_explanation_prompt(
            triple, scores, id2entity, id2relation
        )
        try:
            text = client.generate(prompt, max_tokens=max_tokens)
            out.append((text or "").strip())
        except Exception:
            out.append("")
        if delay_seconds > 0 and i < n - 1:
            time.sleep(delay_seconds)
    return out
