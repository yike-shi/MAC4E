"""
逻辑验证智能体（逻辑推理智能体）：
- 提取目标三元组 (h, r, t) 在知识图谱中的多跳路径，结合路径一致性与关系模式，
  利用大语言模型进行链式逻辑推理，判断该三元组是否与已有知识或常识逻辑产生冲突。
- 输入：目标三元组、从图谱中提取的 h 到 t 的 k 跳路径（及可选的 RGCN 上下文分数）。
- 输出：逻辑置信度 S_logic ∈ [0,1]、由 LLM 生成的逻辑推理说明，供融合决策智能体使用。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TripleId = Tuple[int, int, int]
PathNodes = Tuple[int, ...]
PathRels = Tuple[int, ...]


def _entity_to_text(
    ent_id: int,
    id2entity: Dict[int, str],
    ent2text: Dict[str, str],
) -> str:
    """将实体 ID 转为可读文本。"""
    mid = id2entity.get(ent_id)
    if mid is None:
        return f"实体_{ent_id}"
    return ent2text.get(mid, mid)


def _relation_to_text(
    rel_id: int,
    id2relation: Dict[int, str],
    rel2text: Dict[str, str],
) -> str:
    """将关系 ID 转为可读文本。"""
    key = id2relation.get(rel_id)
    if key is None:
        return f"关系_{rel_id}"
    return rel2text.get(key, key.replace("/", " ").replace("_", " "))


def paths_to_readable(
    paths: List[Tuple[PathNodes, PathRels]],
    id2entity: Dict[int, str],
    id2relation: Dict[int, str],
    ent2text: Dict[str, str],
    rel2text: Dict[str, str],
) -> List[str]:
    """
    将路径列表 [(node_seq, rel_seq), ...] 转为可读自然语言描述列表。
    每条路径格式：h_text -- r1_text --> n1_text, n1_text -- r2_text --> t_text
    """
    out: List[str] = []
    for node_seq, rel_seq in paths:
        if len(node_seq) < 2 or len(rel_seq) != len(node_seq) - 1:
            continue
        parts = []
        for i in range(len(rel_seq)):
            h_n = node_seq[i]
            t_n = node_seq[i + 1]
            r_n = rel_seq[i]
            h_txt = _entity_to_text(h_n, id2entity, ent2text)
            t_txt = _entity_to_text(t_n, id2entity, ent2text)
            r_txt = _relation_to_text(r_n, id2relation, rel2text)
            parts.append(f"{h_txt} --[{r_txt}]--> {t_txt}")
        out.append("；".join(parts))
    return out


def _build_logic_prompt(
    h_text: str,
    r_text: str,
    t_text: str,
    path_descriptions: List[str],
) -> str:
    """构造面向 LLM 的逻辑推理提示：目标陈述 + 图谱路径证据，要求输出分数与理由。"""
    target = f"{h_text} -- {r_text} --> {t_text}"
    if path_descriptions:
        evidence = "图谱证据（多跳路径）：\n" + "\n".join(
            f"  路径{i+1}：{p}" for i, p in enumerate(path_descriptions[:10])
        )
    else:
        evidence = "图谱证据：未发现连接头实体与尾实体的多跳路径（2～3 跳内）。"

    prompt = f"""你是一个逻辑推理专家。请根据已知图谱路径判断目标陈述是否逻辑自洽。

目标陈述：{target}

{evidence}

请结合关系模式（如对称性、传递性、唯一性约束）与路径一致性，判断该目标陈述是否与上述证据存在逻辑冲突。
若存在冲突（例如头实体已通过其他路径指向不同对象、与目标尾实体矛盾），应给出较低分数；若路径支持或与目标不矛盾，则给出较高分数。

请严格按以下两行格式回复（不要省略）：
第一行：Score: <0到1之间的数字>（1表示逻辑自洽、无冲突，0表示存在明显逻辑冲突）
第二行：Reason: <简短理由，说明路径与目标的关系或冲突点，中文或英文均可>"""
    return prompt


def _parse_llm_logic_output(response: str) -> Tuple[float, str]:
    """从 LLM 回复中解析 S_logic 与理由。"""
    response = (response or "").strip()
    score = 0.5
    reason = response

    score_match = re.search(r"[Ss]core\s*:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if score_match:
        try:
            s = float(score_match.group(1))
            score = max(0.0, min(1.0, s))
        except ValueError:
            pass

    reason_match = re.search(r"[Rr]eason\s*:\s*(.+?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    elif score_match:
        idx = score_match.end()
        rest = response[idx:].strip()
        if rest:
            reason = rest.split("\n", 1)[0].strip()

    return score, reason[:500] if len(reason) > 500 else reason


@dataclass
class LogicAgentConfig:
    """
    逻辑验证智能体的配置参数。

    - max_path_len: 路径最大边数（2 或 3 对应 2-hop / 3-hop）。
    - max_paths: 最多保留多少条路径用于构造 LLM 输入。
    - default_score_no_paths: 当无路径且跳过 LLM 时的默认 S_logic。
    - max_tokens: LLM 生成的最大 token 数。
    """

    max_path_len: int = 3
    max_paths: int = 10
    default_score_no_paths: float = 0.5
    max_tokens: int = 350


class LogicVerificationAgent:
    """
    逻辑验证智能体：多跳路径提取 + 可读化 + LLM 逻辑推理，输出 S_logic 与可解释说明。
    可选接入 RGCN 子图编码分数作为上下文，与结构智能体分数对比以发现邻域逻辑异常。
    """

    def __init__(
        self,
        llm_client: Any,
        graph: Any,
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        ent2text: Dict[str, str],
        rel2text: Dict[str, str],
        *,
        config: Optional[LogicAgentConfig] = None,
        save_result_path: Optional[str] = None,
    ) -> None:
        """
        - llm_client: 需实现 generate(prompt: str, max_tokens: int = ..., **kwargs) -> str
        - graph: build_graph(triples) 返回的 nx.MultiDiGraph，用于 bfs_paths。
        - id2entity / id2relation: 整数 ID -> 原始实体/关系字符串。
        - ent2text / rel2text: 原始字符串 -> 可读文本（与事实验证智能体共用映射）。
        """
        self.llm_client = llm_client
        self.graph = graph
        self.id2entity = id2entity or {}
        self.id2relation = id2relation or {}
        self.ent2text = ent2text or {}
        self.rel2text = rel2text or {}
        self.config = config or LogicAgentConfig()
        self.save_result_path = save_result_path

    def _triple_to_readable(self, triple: TripleId) -> Tuple[str, str, str]:
        """将目标三元组转为可读 (h_text, r_text, t_text)。"""
        h, r, t = triple
        h_text = _entity_to_text(h, self.id2entity, self.ent2text)
        r_text = _relation_to_text(r, self.id2relation, self.rel2text)
        t_text = _entity_to_text(t, self.id2entity, self.ent2text)
        return h_text, r_text, t_text

    def _extract_paths(self, h: int, t: int) -> List[Tuple[PathNodes, PathRels]]:
        """从图中提取 h 到 t 的多跳路径（不含直接边，仅 2 跳及以上）。"""
        from ccg_data.path_mining import bfs_paths

        paths = bfs_paths(
            self.graph,
            h,
            t,
            max_len=self.config.max_path_len,
            max_paths=self.config.max_paths,
            return_relation_sequence=True,
        )
        # 仅保留 2 跳及以上路径，排除直接边作为“证据”
        return [(ns, rs) for ns, rs in paths if len(rs) >= 2]

    def score_triple(
        self,
        triple: TripleId,
        s_logic_rgcn: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        对单条三元组计算逻辑置信度 S_logic 及可解释说明。

        流程：
        1) 提取 h->t 多跳路径 → 转为可读描述 → 构造 LLM 提示 → 解析 Score 与 Reason 得到 S_logic_llm；
        2) 若提供了 RGCN 子图编码分数 s_logic_rgcn，则按权重
           S_logic = beta * S_logic_llm + (1 - beta) * s_logic_rgcn
           进行加权，其中 beta 根据路径丰富度与 LLM 可靠性自适应调整；
        3) 若未提供 s_logic_rgcn，则直接返回 S_logic_llm。
        """
        h, r, t = triple
        h_text, r_text, t_text = self._triple_to_readable(triple)

        paths = self._extract_paths(h, t)
        path_descriptions = paths_to_readable(
            paths,
            self.id2entity,
            self.id2relation,
            self.ent2text,
            self.rel2text,
        )

        prompt = _build_logic_prompt(h_text, r_text, t_text, path_descriptions)
        try:
            raw = self.llm_client.generate(
                prompt, max_tokens=self.config.max_tokens
            )
            s_logic_llm, explanation = _parse_llm_logic_output(raw)
            llm_failed = False
        except Exception as e:
            s_logic_llm = self.config.default_score_no_paths
            explanation = f"[逻辑验证] LLM 调用异常: {e!s}；使用默认分数。"
            llm_failed = True

        path_count = len(paths)

        # 若未提供 RGCN 分数，则直接使用 LLM 分数
        if s_logic_rgcn is None:
            s_logic_final = s_logic_llm
            beta = 1.0
        else:
            # 检测是否使用 Mock 客户端
            used_mock = type(self.llm_client).__name__ == "MockLLMClient"

            # 自适应 beta：
            # - LLM 失败或 Mock：完全信任 RGCN（beta=0）
            # - 无 2+ 跳路径：LLM 证据贫乏，beta 较小（0.2）
            # - 正常情况：根据路径数量在线性调整 beta ∈ [0.5, 0.7]
            if llm_failed or used_mock:
                beta = 0.0
            elif path_count == 0:
                beta = 0.2
            else:
                richness = min(path_count / 5.0, 1.0)
                beta = 0.5 + 0.2 * richness

            s_logic_final = beta * s_logic_llm + (1.0 - beta) * float(s_logic_rgcn)

        result: Dict[str, Any] = {
            "S_logic": s_logic_final,
            "S_logic_llm": s_logic_llm,
            "S_logic_rgcn": s_logic_rgcn,
            "beta": beta,
            "explanation": explanation,
            "path_count": path_count,
            "path_descriptions": path_descriptions[:5],
        }

        if self.save_result_path:
            self._append_result_record(triple, result)

        return result

    def _append_result_record(self, triple: TripleId, result: Dict[str, Any]) -> None:
        """将当前三元组的逻辑验证结果追加写入 JSONL。"""
        path = Path(self.save_result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "triple": list(triple),
            "S_logic": result.get("S_logic"),
            "explanation": result.get("explanation"),
            "path_count": result.get("path_count"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
