"""
事实验证智能体（常识核查智能体）：
- 结合三元组内部语义与上下文信息，依靠大语言模型的先验知识进行开放域事实核查，
  评估该事实在现实世界中的合理性；针对具有强迷惑性的错误（如元素相关但不匹配）进行判断。
- 输入：目标三元组 (h, r, t)，以及实体/关系到可读文本的映射（见 ccg_data.text_mapping）。
- 输出：事实验证置信度 S_fact ∈ [0,1]、可选的可解释说明，供融合决策智能体使用。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

TripleId = Tuple[int, int, int]


@dataclass
class FactAgentConfig:
    """
    事实验证智能体的配置参数。

    - default_score_no_readable: 当无法得到可读三元组（缺映射）时的默认 S_fact，建议 0.5。
    - skip_fact_when_no_readable: 为 True 时，无可读表示则返回 default_score_no_readable 且 explanation 标记“跳过”；
      为 False 时仍尝试用关系路径等弱可读形式调用 LLM。
    - max_tokens: LLM 生成的最大 token 数。
    - output_format: 期望 LLM 输出格式，用于解析。"score_first" 表示先分数后理由。
    """

    default_score_no_readable: float = 0.5
    skip_fact_when_no_readable: bool = True
    max_tokens: int = 300
    output_format: str = "score_first"


def _build_fact_check_prompt(h_text: str, r_text: str, t_text: str) -> str:
    """构造面向 LLM 的事实验证提示：可读三元组 + 要求输出分数与理由。"""
    prompt = f"""你是一个知识图谱与事实核查专家。请判断以下陈述在现实世界中是否合理。

陈述（头实体, 关系, 尾实体）：
- 头实体：{h_text}
- 关系：{r_text}
- 尾实体：{t_text}

请从常识与事实角度评估：该关系是否可能真实地连接这两个实体？是否存在明显矛盾或强迷惑性错误（如实体相关但关系不匹配）？

请严格按以下两行格式回复（不要省略）：
第一行：Score: <0到1之间的数字>（1表示非常合理，0表示明显错误或不合理）
第二行：Reason: <一句话理由，中文或英文均可>"""
    return prompt


def _parse_llm_fact_output(response: str) -> Tuple[float, str]:
    """
    从 LLM 回复中解析 S_fact 与理由。容忍多种写法。
    返回 (score, reason)；若解析失败则返回 (0.5, response[:200])。
    """
    response = (response or "").strip()
    score = 0.5
    reason = response

    # 尝试匹配 "Score: 0.8" 或 "Score:0.8" 或 "score: 0.8"
    score_match = re.search(r"[Ss]core\s*:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if score_match:
        try:
            s = float(score_match.group(1))
            score = max(0.0, min(1.0, s))
        except ValueError:
            pass

    # 尝试匹配 "Reason: ..."
    reason_match = re.search(r"[Rr]eason\s*:\s*(.+?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    elif score_match:
        # 取 Score 行之后的内容作为理由
        idx = score_match.end()
        rest = response[idx:].strip()
        if rest:
            reason = rest.split("\n", 1)[0].strip()

    return score, reason[:500] if len(reason) > 500 else reason


class FactVerificationAgent:
    """
    事实验证智能体：基于可读三元组调用 LLM 进行开放域事实核查，输出 S_fact 与可解释说明。
    若无可读映射（或选择跳过），则返回配置的默认分数并标记跳过原因。
    """

    def __init__(
        self,
        llm_client: Any,
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        ent2text: Dict[str, str],
        rel2text: Dict[str, str],
        *,
        config: Optional[FactAgentConfig] = None,
        save_result_path: Optional[str] = None,
    ) -> None:
        """
        - llm_client: 需实现 generate(prompt: str, max_tokens: int = ..., **kwargs) -> str
        - id2entity / id2relation: 整数 ID -> 原始实体/关系字符串（如 Freebase mid、关系路径）
        - ent2text / rel2text: 原始字符串 -> 可读文本（如 entity2text.txt、relation2text.txt 的映射）
        """
        self.llm_client = llm_client
        self.id2entity = id2entity or {}
        self.id2relation = id2relation or {}
        self.ent2text = ent2text or {}
        self.rel2text = rel2text or {}
        self.config = config or FactAgentConfig()
        self.save_result_path = save_result_path

    def _triple_to_readable(self, triple: TripleId) -> Optional[Tuple[str, str, str]]:
        """
        将 ID 形式三元组转为 (h_text, r_text, t_text)。
        若缺少必要映射且为纯 ID 无语义，可返回 None 表示“无可读表示”。
        """
        from ccg_data.text_mapping import triple_id_to_text

        h_id, r_id, t_id = triple
        h_mid = self.id2entity.get(h_id)
        r_key = self.id2relation.get(r_id)
        t_mid = self.id2entity.get(t_id)

        if h_mid is None or r_key is None or t_mid is None:
            return None

        h_text, r_text, t_text = triple_id_to_text(
            triple,
            self.id2entity,
            self.id2relation,
            self.ent2text,
            self.rel2text,
        )
        # 若全部仍是原始 ID/路径，可视为“弱可读”；这里仍返回，由调用方决定是否降级
        return (h_text, r_text, t_text)

    def score_triple(self, triple: TripleId) -> Dict[str, Any]:
        """
        对单条三元组计算事实验证置信度 S_fact 及可解释说明。
        若无可读表示且配置为跳过，则返回 default_score 与 skip 标记。
        """
        h_id, r_id, t_id = triple
        readable = self._triple_to_readable(triple)

        if readable is None:
            s_fact = self.config.default_score_no_readable
            result: Dict[str, Any] = {
                "S_fact": s_fact,
                "explanation": "[事实验证] 无法解析实体/关系 ID，跳过事实核查；使用默认分数。",
                "readable_triple": None,
                "skipped": True,
            }
            if self.save_result_path:
                self._append_result_record(triple, result)
            return result

        h_text, r_text, t_text = readable

        # 可选：若配置为无可读则跳过，可在此处根据“是否全为原始 ID”再判断一次
        # 这里简单认为 triple_id_to_text 已做回退，只要有 readable 就调用 LLM
        if self.config.skip_fact_when_no_readable:
            # 检查头尾实体是否在 ent2text 中有可读映射（若均为原始 mid 则视为无可读）
            h_mid = self.id2entity.get(h_id)
            t_mid = self.id2entity.get(t_id)
            has_entity_readable = (h_mid in self.ent2text) or (t_mid in self.ent2text)
            if not has_entity_readable and h_mid is not None and t_mid is not None:
                s_fact = self.config.default_score_no_readable
                result = {
                    "S_fact": s_fact,
                    "explanation": "[事实验证] 实体无可读文本映射，跳过事实核查；使用默认分数。",
                    "readable_triple": (h_text, r_text, t_text),
                    "skipped": True,
                }
                if self.save_result_path:
                    self._append_result_record(triple, result)
                return result

        prompt = _build_fact_check_prompt(h_text, r_text, t_text)
        try:
            raw = self.llm_client.generate(
                prompt, max_tokens=self.config.max_tokens
            )
            s_fact, explanation = _parse_llm_fact_output(raw)
        except Exception as e:
            s_fact = self.config.default_score_no_readable
            explanation = f"[事实验证] LLM 调用异常: {e!s}；使用默认分数。"

        result = {
            "S_fact": s_fact,
            "explanation": explanation,
            "readable_triple": (h_text, r_text, t_text),
            "skipped": False,
        }
        if self.save_result_path:
            self._append_result_record(triple, result)
        return result

    def _append_result_record(self, triple: TripleId, result: Dict[str, Any]) -> None:
        """将当前三元组的事实验证结果追加写入 JSONL。"""
        path = Path(self.save_result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "triple": list(triple),
            "S_fact": result.get("S_fact"),
            "explanation": result.get("explanation"),
            "readable_triple": result.get("readable_triple"),
            "skipped": result.get("skipped", False),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
