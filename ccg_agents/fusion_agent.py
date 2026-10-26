"""
融合决策智能体（Fusion Decision Agent）：

在任务分配智能体与结构、事实、逻辑三个子智能体之后，对本条三元组的检测结果进行综合决策。
输入：分配权重 [m_s, m_f, m_l]、结构/事实/逻辑的评分与理由；
输出：最终置信度分数 S_fusion ∈ [0,1]，以及由 LLM 综合生成的可解释理由。

设计原则：
- 综合各子智能体判断结果，避免过度依赖单一智能体结论；
- 以加权评分或规则引导为主，同时引入 LLM 对多源证据与理由进行综合，输出稳定且可解释的置信度。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

AllocatorWeights = Union[List[float], Tuple[float, float, float]]

# 结构智能体返回的字典至少包含 S_struct，可选 explanation
StructureResult = Dict[str, Any]
# 事实智能体返回的字典至少包含 S_fact、explanation
FactResult = Dict[str, Any]
# 逻辑智能体返回的字典至少包含 S_logic、explanation
LogicResult = Dict[str, Any]


def _build_fusion_prompt(
    allocator_weights: AllocatorWeights,
    structure_result: StructureResult,
    fact_result: FactResult,
    logic_result: LogicResult,
    *,
    readable_triple: Optional[Tuple[str, str, str]] = None,
) -> str:
    """
    构造面向 LLM 的融合决策提示：汇总任务分配权重、三个子智能体的评分与理由，
    要求 LLM 输出一个 [0,1] 的置信度分数及综合理由（不做正确/错误的二值判断）。

    参数
    ----
    allocator_weights : [m_s, m_f, m_l]
        任务分配智能体给出的权重，表示当前三元组更应依赖结构/事实/逻辑哪一类证据。
    structure_result : 至少含 "S_struct"，可选 "explanation"
    fact_result : 至少含 "S_fact", "explanation"
    logic_result : 至少含 "S_logic", "explanation"
    readable_triple : 可选 (头实体文本, 关系文本, 尾实体文本)，便于 LLM 理解上下文
    """
    m_s, m_f, m_l = _normalize_allocator_weights(allocator_weights)
    s_struct = float(structure_result.get("S_struct", 0.0))
    s_fact = float(fact_result.get("S_fact", 0.5))
    s_logic = float(logic_result.get("S_logic", 0.5))

    struct_expl = structure_result.get("explanation") or "（无说明）"
    fact_expl = fact_result.get("explanation") or "（无说明）"
    logic_expl = logic_result.get("explanation") or "（无说明）"

    lines: List[str] = [
        "你是一个知识图谱错误检测的融合决策专家。请根据以下多智能体的检测结果，给出该三元组为「正确」的置信度分数（0 表示几乎肯定错误，1 表示几乎肯定正确），并给出综合理由。",
        "",
        "【任务分配权重】（本三元组建议依赖各视角的程度）",
        f"  结构: {m_s:.2f},  事实: {m_f:.2f},  逻辑: {m_l:.2f}",
        "",
        "【结构验证智能体】",
        f"  评分 S_struct = {s_struct:.2f}（越高越合理）",
        f"  说明: {struct_expl[:300]}{'...' if len(str(struct_expl)) > 300 else ''}",
        "",
        "【事实验证智能体】",
        f"  评分 S_fact = {s_fact:.2f}（越高越合理）",
        f"  说明: {fact_expl[:300]}{'...' if len(str(fact_expl)) > 300 else ''}",
        "",
        "【逻辑验证智能体】",
        f"  评分 S_logic = {s_logic:.2f}（越高越合理）",
        f"  说明: {logic_expl[:300]}{'...' if len(str(logic_expl)) > 300 else ''}",
    ]
    if readable_triple is not None:
        h_t, r_t, t_t = readable_triple
        lines.insert(2, f"【当前三元组（可读形式）】头实体: {h_t}, 关系: {r_t}, 尾实体: {t_t}")
        lines.insert(3, "")

    lines.extend([
        "",
        "请综合考虑上述权重与各视角的评分及理由，输出一个 0 到 1 之间的置信度分数。",
        "严格按以下两行格式回复（不要省略）：",
        "第一行: Score: <0到1之间的数字>",
        "第二行: Reason: <一句话综合理由，说明置信度的依据，中文或英文均可>",
    ])
    return "\n".join(lines)


def _normalize_allocator_weights(w: AllocatorWeights) -> Tuple[float, float, float]:
    """将分配权重规范为 (m_s, m_f, m_l)，保证非负且和为 1。"""
    a, b, c = float(w[0]), float(w[1]), float(w[2])
    a, b, c = max(0.0, a), max(0.0, b), max(0.0, c)
    total = a + b + c
    if total < 1e-9:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (a / total, b / total, c / total)


def _parse_fusion_llm_output(response: str) -> Tuple[float, str]:
    """
    从 LLM 回复中解析置信度分数与理由。仅输出分数，不做正确/错误判定。

    返回
    ----
    (final_score, reason)
    - final_score: float in [0,1]，解析到 Score 则使用，否则 0.5；若仅有 Decision 则映射为 1.0/0.0
    - reason: 综合理由文本，解析失败时取回复前 500 字符
    """
    response = (response or "").strip()
    final_score = 0.5
    reason = response[:500] if response else "[融合决策] 无有效回复。"

    # 1) 优先解析 Score: 0-1（主输出为连续置信度）
    score_match = re.search(r"[Ss]core\s*:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if score_match:
        try:
            s = float(score_match.group(1))
            final_score = max(0.0, min(1.0, s))
        except ValueError:
            pass
    else:
        # 2) 若无 Score 仅有 Decision，则映射为 1.0/0.0 作为分数
        decision_match = re.search(
            r"[Dd]ecision\s*:\s*(correct|incorrect|正确|错误)",
            response,
            re.IGNORECASE,
        )
        if decision_match:
            raw = decision_match.group(1).lower()
            final_score = 1.0 if raw in ("correct", "正确") else 0.0

    # 3) 解析 Reason: ...
    reason_match = re.search(
        r"[Rr]eason\s*:\s*(.+?)(?:\n|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if reason_match:
        reason = reason_match.group(1).strip()[:500]
    elif score_match:
        idx = score_match.end()
        rest = response[idx:].strip()
        if rest:
            reason = rest.split("\n", 1)[0].strip()[:500]

    return final_score, reason


def _weighted_score(
    allocator_weights: AllocatorWeights,
    structure_result: StructureResult,
    fact_result: FactResult,
    logic_result: LogicResult,
) -> float:
    """
    不使用 LLM 时的后备融合方式：加权平均 S_final = m_s*S_struct + m_f*S_fact + m_l*S_logic。
    当 LLM 调用失败或未配置时，直接将该分数作为置信度输出，不做正确/错误判定。
    """
    m_s, m_f, m_l = _normalize_allocator_weights(allocator_weights)
    s_struct = float(structure_result.get("S_struct", 0.0))
    s_fact = float(fact_result.get("S_fact", 0.5))
    s_logic = float(logic_result.get("S_logic", 0.5))
    return m_s * s_struct + m_f * s_fact + m_l * s_logic


# -----------------------------------------------------------------------------
# 配置与主类
# -----------------------------------------------------------------------------


@dataclass
class FusionAgentConfig:
    """
    融合决策智能体的配置参数。

    - use_llm: 是否调用 LLM 进行综合决策与理由生成；为 False 时仅用加权平均作为置信度，理由为固定说明。
    - max_tokens: LLM 生成的最大 token 数。
    """

    use_llm: bool = True
    max_tokens: int = 350


class FusionDecisionAgent:
    """
    融合决策智能体：综合任务分配权重与结构、事实、逻辑三个子智能体的评分与理由，
    输出最终置信度分数 S_fusion ∈ [0,1] 及可解释的综合理由。

    流程简述：
    1）若配置 use_llm=True：构造包含权重与三路评分/理由的 prompt，调用 LLM；
       解析 LLM 返回的 Score 与 Reason，得到 final_score、explanation。
    2）若 use_llm=False 或 LLM 调用异常：采用加权平均作为 final_score，explanation 为固定说明。
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        config: Optional[FusionAgentConfig] = None,
        save_result_path: Optional[str] = None,
    ) -> None:
        """
        参数
        ----
        llm_client : 需实现 generate(prompt: str, max_tokens: int = ..., **kwargs) -> str
        config : 融合决策配置，None 则使用默认 FusionAgentConfig()
        save_result_path : 若提供，则将每条三元组的融合结果追加写入该 JSONL 文件
        """
        self.llm_client = llm_client
        self.config = config or FusionAgentConfig()
        self.save_result_path = save_result_path

    def decide(
        self,
        allocator_weights: AllocatorWeights,
        structure_result: StructureResult,
        fact_result: FactResult,
        logic_result: LogicResult,
        *,
        readable_triple: Optional[Tuple[str, str, str]] = None,
    ) -> Dict[str, Any]:
        """
        对单条三元组进行融合决策：综合分配权重与三路子智能体结果，输出最终置信度分数与理由。

        参数
        ----
        allocator_weights : [m_s, m_f, m_l]，来自任务分配智能体
        structure_result : 结构智能体 score_triple() 的返回值
        fact_result : 事实验证智能体 score_triple() 的返回值
        logic_result : 逻辑验证智能体 score_triple() 的返回值
        readable_triple : 可选，(头实体文本, 关系文本, 尾实体文本)，用于在 prompt 中展示

        返回
        ----
        dict，包含：
        - final_score: float in [0,1]，最终置信度（越高越可信为正例），由调用方自选阈值做二分类
        - explanation: str，综合理由
        - used_llm: bool，本次是否使用了 LLM（若 LLM 异常则会回退到加权平均）
        - weighted_score: float，加权平均分数（便于分析与对比）
        """
        m_s, m_f, m_l = _normalize_allocator_weights(allocator_weights)
        weighted = _weighted_score(
            allocator_weights,
            structure_result,
            fact_result,
            logic_result,
        )

        used_llm = False
        final_score = weighted
        explanation = (
            f"[融合决策] 未使用 LLM；采用加权平均 S_fusion = {weighted:.4f} "
            f"(m_s*S_struct + m_f*S_fact + m_l*S_logic)。"
        )

        if self.config.use_llm:
            prompt = _build_fusion_prompt(
                allocator_weights,
                structure_result,
                fact_result,
                logic_result,
                readable_triple=readable_triple,
            )
            try:
                raw = self.llm_client.generate(
                    prompt,
                    max_tokens=self.config.max_tokens,
                )
                score, reason = _parse_fusion_llm_output(raw)
                used_llm = True
                final_score = score
                explanation = reason
            except Exception as e:
                explanation = (
                    f"[融合决策] LLM 调用异常: {e!s}；回退到加权平均 S_fusion = {weighted:.4f}。"
                )

        result: Dict[str, Any] = {
            "final_score": final_score,
            "explanation": explanation,
            "used_llm": used_llm,
            "weighted_score": weighted,
            "allocator_weights": {"m_s": m_s, "m_f": m_f, "m_l": m_l},
        }

        if self.save_result_path:
            self._append_result_record(result)

        return result

    def _append_result_record(self, result: Dict[str, Any]) -> None:
        """将当前融合结果追加写入 JSONL（不包含三元组 ID，由调用方在 pipeline 中一并写入）。"""
        path = Path(self.save_result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "final_score": result.get("final_score"),
            "explanation": result.get("explanation"),
            "used_llm": result.get("used_llm"),
            "weighted_score": result.get("weighted_score"),
            "allocator_weights": result.get("allocator_weights"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
