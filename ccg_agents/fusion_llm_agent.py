"""
基于 LLM 的融合智能体（Reliability-Weighted Fusion：

由 LLM 根据三路分数与各智能体文本说明，输出可靠性权重 w_struct/w_fact/w_logic（和为 1），
再计算最终置信度：final_score = w_struct*S_struct + w_fact*S_fact + w_logic*S_logic。

与 FusionDecisionAgent 的区别：
- fusion_agent：Allocator 权重固定参与加权，或用另一套 prompt 让 LLM 直接给「整体置信度 Score」；
- 本模块：LLM 专门做「信任分配」，数值融合完全由 LLM 给出的权重决定（Allocator 仅可作为 prompt 中的软先验）。

用法：与 FusionDecisionAgent 相同传入 structure/fact/logic 的 dict 及可选 readable_triple；
额外可传 subgraph_info（邻接子图/邻居三元组等文本）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

AllocatorWeights = Union[List[float], Tuple[float, float, float]]
StructureResult = Dict[str, Any]
FactResult = Dict[str, Any]
LogicResult = Dict[str, Any]

_TRIPLE_ID = Tuple[int, int, int]


def _normalize_three_weights(a: float, b: float, c: float) -> Tuple[float, float, float]:
    a, b, c = max(0.0, float(a)), max(0.0, float(b)), max(0.0, float(c))
    s = a + b + c
    if s < 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (a / s, b / s, c / s)


def _normalize_allocator_weights(w: AllocatorWeights) -> Tuple[float, float, float]:
    return _normalize_three_weights(float(w[0]), float(w[1]), float(w[2]))


def _weighted_with(
    weights: Tuple[float, float, float],
    structure_result: StructureResult,
    fact_result: FactResult,
    logic_result: LogicResult,
) -> float:
    m_s, m_f, m_l = weights
    s_struct = float(structure_result.get("S_struct", 0.0))
    s_fact = float(fact_result.get("S_fact", 0.5))
    s_logic = float(logic_result.get("S_logic", 0.5))
    return m_s * s_struct + m_f * s_fact + m_l * s_logic


def _truncate(s: Any, max_len: int) -> str:
    t = "" if s is None else str(s).strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 回复中解析单个 JSON 对象（支持 ```json 代码块）。"""
    raw = (text or "").strip()
    if not raw:
        return None

    # 1) 整块即 JSON
    for candidate in (raw,):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 2) ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3) 第一个 { 到最后一个 }
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def build_llm_reliability_prompt(
    *,
    structure_result: StructureResult,
    fact_result: FactResult,
    logic_result: LogicResult,
    readable_triple: Optional[Tuple[str, str, str]] = None,
    triple_id: Optional[_TRIPLE_ID] = None,
    subgraph_info: Optional[str] = None,
    allocator_weights: Optional[AllocatorWeights] = None,
    expl_max_len: int = 600,
) -> str:
    """
    构造英文可靠性权重 prompt（与用户提供的专家系统说明一致），并填入分数与说明。

    allocator_weights
        任务分配智能体（Allocator）输出的 [m_s, m_f, m_l]，写入 prompt 的 Reference 节，供 LLM 作可靠性加权时的**参考**。
        可为 None（则 Reference 节说明未提供 Allocator 权重）。
    """
    s_struct = float(structure_result.get("S_struct", 0.0))
    s_fact = float(fact_result.get("S_fact", 0.5))
    s_logic = float(logic_result.get("S_logic", 0.5))

    if readable_triple is not None:
        h_t, r_t, t_t = readable_triple
        triple_line = f"({h_t}, {r_t}, {t_t})"
    elif triple_id is not None:
        h, r, t = triple_id
        triple_line = f"(head_id={h}, relation_id={r}, tail_id={t})"
    else:
        triple_line = "(unknown)"

    struct_expl = _truncate(structure_result.get("explanation"), expl_max_len) or "(none)"
    fact_expl = _truncate(fact_result.get("explanation"), expl_max_len) or "(none)"
    logic_expl = _truncate(logic_result.get("explanation"), expl_max_len) or "(none)"

    # 事实侧可读文本：优先显式字段
    fact_text_bits: List[str] = []
    rt = fact_result.get("readable_triple")
    if rt is not None and len(rt) == 3:
        fact_text_bits.append(f"Readable triple: ({rt[0]}, {rt[1]}, {rt[2]})")
    if fact_result.get("skipped"):
        fact_text_bits.append("Fact agent: skipped or low-resource.")
    subgraph_block = (subgraph_info or "").strip() or "(none)"

    # 任务分配智能体（Allocator）权重：单独成节，作为上游参考
    task_allocator_section = ""
    if allocator_weights is not None:
        m_s, m_f, m_l = _normalize_allocator_weights(allocator_weights)
        task_allocator_section = f"""
### Reference: Task Allocation Agent (upstream weights)

The **task allocation agent** (Allocator) has already produced weights for **this** triple: how much the pipeline initially trusts the **structure / fact / logic** branches based on learned features. These are denoted **m_struct, m_fact, m_logic** (non-negative, sum to 1).

Use them as **reference only**. Your output **w_struct, w_fact, w_logic** describe **reliability** of each agent given scores and text; you **may** diverge from m_* when evidence warrants.

- m_struct (structure): {m_s:.6f}
- m_fact (fact): {m_f:.6f}
- m_logic (logic): {m_l:.6f}
(Check: {m_s:.6f} + {m_f:.6f} + {m_l:.6f} = 1.0)

"""
    else:
        task_allocator_section = """
### Reference: Task Allocation Agent

(No task-allocation weights were provided for this call.)

"""

    # Optional: structure / logic extra keys commonly present
    extra_struct = ""
    for k in ("S_transe", "S_topo_paths", "support_paths"):
        if k in structure_result and structure_result[k] is not None:
            extra_struct += f"\n  - {k}: {_truncate(structure_result[k], 200)}"
    extra_logic = ""
    for k in ("S_logic_rgcn", "S_logic_llm", "beta", "path_count"):
        if k in logic_result and logic_result[k] is not None:
            extra_logic += f"\n  - {k}: {logic_result[k]}"

    fact_context = "\n".join(fact_text_bits) if fact_text_bits else "(no extra readable fields)"

    prompt = f"""You are an expert system for knowledge graph error detection.

Your task is NOT to directly judge whether a triple is correct.
Instead, you must determine how much we should TRUST each of the following three agents:

1. Structural Agent (TransE-based):
   - Captures embedding-level plausibility
   - Good at global statistical patterns
   - Weak at logical contradictions

2. Factual Agent (LLM-based):
   - Uses world knowledge and commonsense reasoning
   - Good at real-world facts
   - May hallucinate or be uncertain

3. Logical Agent (RGCN-based):
   - Uses subgraph structure and relational patterns
   - Good at detecting logical inconsistencies
   - May fail when subgraph is sparse or noisy

---

### Input:

Triple:
{triple_line}

Agent scores (higher = more likely TRUE):
- Structural score: {s_struct}
- Factual score: {s_fact}
- Logical score: {s_logic}

{task_allocator_section}
### Additional context

- Neighboring triples / subgraph (if any): {subgraph_block}
- Factual / readable context: {fact_context}

### Per-agent explanations / confidence (text):

Structural agent:
{struct_expl}{extra_struct}

Factual agent:
{fact_expl}

Logical agent:
{logic_expl}{extra_logic}

---

### Your Task:

Analyze the situation and assign RELIABILITY WEIGHTS to each agent:
- w_struct
- w_fact
- w_logic

Such that:
- All weights are between 0 and 1
- Sum of weights = 1

---

### Guidelines:

1. If all three agents agree → assign roughly equal weights.

2. If one agent strongly disagrees with others:
   - Analyze whether that agent is likely wrong or insightful.
   - Down-weight unreliable signals.

3. Prefer:
   - Logical agent when structural contradictions exist
   - Factual agent when commonsense knowledge is clear
   - Structural agent when patterns are consistent but others uncertain

4. If factual agent is uncertain or hallucination-prone → reduce its weight.

5. If subgraph is sparse → reduce logical agent weight.

6. Consider the **Task Allocation Agent** reference weights (m_struct, m_fact, m_logic) above: they summarize feature-based routing for this triple. Treat them as a sensible default or tie-breaker, but **override** when agent scores or explanations clearly indicate one branch is untrustworthy or especially informative.

---

### Output format (STRICT JSON only, no other text before or after):

{{
  "w_struct": <float>,
  "w_fact": <float>,
  "w_logic": <float>,
  "reason": "<brief explanation>"
}}
"""
    return prompt


def parse_reliability_json(response: str) -> Optional[Tuple[float, float, float, str]]:
    """解析 LLM 返回的 JSON，得到 (w_struct, w_fact, w_logic, reason)。"""
    obj = _extract_json_object(response)
    if obj is None:
        return None
    try:
        ws = float(obj["w_struct"])
        wf = float(obj["w_fact"])
        wl = float(obj["w_logic"])
        reason = str(obj.get("reason", "")).strip() or "(no reason)"
    except (KeyError, TypeError, ValueError):
        return None
    ws, wf, wl = _normalize_three_weights(ws, wf, wl)
    return (ws, wf, wl, reason)


@dataclass
class FusionLLMWeightAgentConfig:
    """LLM 可靠性权重融合配置。"""

    use_llm: bool = True
    max_tokens: int = 512
    # 是否在 prompt 中附带 Allocator 权重作为软先验
    include_allocator_hint: bool = True
    explanation_max_len: int = 600


class FusionLLMWeightAgent:
    """
    调用 LLM 输出三路可靠性权重，再线性融合分数。

    decide() 返回字段与 FusionDecisionAgent 对齐的主要键：
    - final_score, weighted_score, explanation, used_llm, allocator_weights
    并额外返回 reliability_weights、llm_reason、allocator_baseline_score。
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        config: Optional[FusionLLMWeightAgentConfig] = None,
        save_result_path: Optional[str] = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or FusionLLMWeightAgentConfig()
        self.save_result_path = save_result_path

    def decide(
        self,
        allocator_weights: AllocatorWeights,
        structure_result: StructureResult,
        fact_result: FactResult,
        logic_result: LogicResult,
        *,
        readable_triple: Optional[Tuple[str, str, str]] = None,
        triple_id: Optional[_TRIPLE_ID] = None,
        subgraph_info: Optional[str] = None,
        reference_allocator_weights: Optional[AllocatorWeights] = None,
    ) -> Dict[str, Any]:
        """
        参数
        ----
        allocator_weights
            任务分配智能体对当前三元组的分支权重 [m_s, m_f, m_l]；用于计算 allocator_baseline_score，并默认写入 LLM prompt 作参考。
        reference_allocator_weights
            若提供，则 **仅** 在 prompt 的「Task Allocation Agent」参考节中使用该组权重（例如与 allocator_weights 区分做消融时）；
            若不提供，则 prompt 中与 baseline 使用同一组 allocator_weights。
        """
        m_s, m_f, m_l = _normalize_allocator_weights(allocator_weights)
        allocator_baseline = _weighted_with(
            (m_s, m_f, m_l), structure_result, fact_result, logic_result
        )

        used_llm = False
        final_score = allocator_baseline
        weighted_score = allocator_baseline
        explanation = (
            f"[FusionLLMWeight] 未使用 LLM；回退为 Allocator 加权平均 = {allocator_baseline:.6f}。"
        )
        reliability_weights: Optional[Dict[str, float]] = None
        llm_reason = ""

        # prompt 中展示的「任务分配智能体参考权重」：默认同 allocator_weights，可被 reference_allocator_weights 覆盖
        if self.config.include_allocator_hint:
            hint_for_prompt: Optional[AllocatorWeights] = (
                reference_allocator_weights
                if reference_allocator_weights is not None
                else allocator_weights
            )
        else:
            hint_for_prompt = None

        if self.config.use_llm:
            prompt = build_llm_reliability_prompt(
                structure_result=structure_result,
                fact_result=fact_result,
                logic_result=logic_result,
                readable_triple=readable_triple,
                triple_id=triple_id,
                subgraph_info=subgraph_info,
                allocator_weights=hint_for_prompt,
                expl_max_len=self.config.explanation_max_len,
            )
            try:
                raw = self.llm_client.generate(
                    prompt,
                    max_tokens=self.config.max_tokens,
                )
                parsed = parse_reliability_json(raw)
                if parsed is None:
                    explanation = (
                        f"[FusionLLMWeight] LLM 返回无法解析为 JSON；"
                        f"回退 Allocator 加权 = {allocator_baseline:.6f}。原始片段: {_truncate(raw, 300)}"
                    )
                else:
                    ws, wf, wl, llm_reason = parsed
                    used_llm = True
                    reliability_weights = {"w_struct": ws, "w_fact": wf, "w_logic": wl}
                    fused = _weighted_with(
                        (ws, wf, wl), structure_result, fact_result, logic_result
                    )
                    final_score = max(0.0, min(1.0, fused))
                    weighted_score = final_score
                    explanation = llm_reason
            except Exception as e:
                explanation = (
                    f"[FusionLLMWeight] LLM 调用异常: {e!s}；"
                    f"回退 Allocator 加权 = {allocator_baseline:.6f}。"
                )
        else:
            explanation = (
                f"[FusionLLMWeight] use_llm=False；Allocator 加权 = {allocator_baseline:.6f}。"
            )

        result: Dict[str, Any] = {
            "final_score": float(final_score),
            "explanation": explanation,
            "used_llm": used_llm,
            "weighted_score": float(weighted_score),
            "allocator_weights": {"m_s": m_s, "m_f": m_f, "m_l": m_l},
            "allocator_baseline_score": float(allocator_baseline),
            "reliability_weights": reliability_weights,
            "llm_reason": llm_reason if used_llm else None,
        }

        if self.save_result_path:
            self._append_result_record(result)

        return result

    def _append_result_record(self, result: Dict[str, Any]) -> None:
        path = Path(self.save_result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            k: result.get(k)
            for k in (
                "final_score",
                "explanation",
                "used_llm",
                "weighted_score",
                "allocator_weights",
                "allocator_baseline_score",
                "reliability_weights",
                "llm_reason",
            )
        }
        record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
