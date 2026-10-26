"""
StructureConstraintAgent：
- 对单条三元组 (h,r,t) 给出结构置信度 S_struct ∈ [0,1]。
- 结构分数由 TransE 嵌入置信度与 h->t 拓扑路径支撑（support_paths/has_path）按权重融合得到。
- 可选：接入大语言模型对判断结果生成可解释说明，并持久化供融合决策智能体使用。
"""

from __future__ import annotations

import math
from collections import deque
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ccg_models.transE import TransEModel
from ccg_models.complex import ComplExModel, complex_struct_confidence
from ccg_data.path_mining import count_support_paths

TripleId = Tuple[int, int, int]
# 结构解释器：(triple, scores_dict, id2entity, id2relation) -> 说明文本
StructureExplainer = Callable[
    [TripleId, Dict[str, float], Dict[int, str], Dict[int, str]], str
]


@dataclass
class StructureAgentConfig:
    """
    结构约束智能体的配置参数。

    - emb_mode: 嵌入分数 S_emb 的来源，三选一：
        "transe"  -> 仅 TransE 分数（默认）
        "complex" -> 仅 ComplEx 分数（需传入 complex_model）
        "fusion"  -> 融合 S_emb = alpha*S_transe + (1-alpha)*S_complex（需传入 complex_model）
    - alpha: 仅当 emb_mode=="fusion" 时生效，融合系数
    - transe_confidence_center: 将 TransE 距离 f 转为“概率式”置信度：S = sigmoid((center - f) / scale)。
      距离 f 越小（三元组越合理）S 越高。center 越大整体分数越高，越小整体分数越低。典型 4~6。
    - transe_confidence_scale: 温度，越大曲线越缓，越小曲线越陡、正负分离越明显。典型 1.0~2.0。
    """

    emb_mode: str = "transe"  # "transe" | "complex" | "fusion"
    alpha: float = 0.5
    transe_confidence_center: float = 5.0  # 越大 S_struct 整体越高；sigmoid((center - f) / scale)
    transe_confidence_scale: float = 2.0

    # --- 拓扑路径统计（支持路径数量） ---
    # 用于计算结构中的“拓扑支撑度”，以轻量 BFS 可达性门控后再调用 count_support_paths。
    use_topo_path_support: bool = True
    topo_max_path_len: int = 3
    # count_support_paths 内部会枚举最多 max_paths 条路径；该值越大越慢但更精细
    topo_count_max_paths: int = 200
    # 如果不可达，则跳过 count_support_paths（降低复杂度）
    topo_use_reachable_gate: bool = True
    # 融合权重：S_struct = w_transe * S_transe + (1-w_transe) * S_topo_paths
    w_transe_for_struct: float = 0.7
    # 当 topo_count_max_paths 太小时，归一化更平滑可用对数映射
    use_log_norm_for_topo_paths: bool = True


class StructureConstraintAgent:
    """
    结构智能体：结构置信度由嵌入分数决定，S_struct = S_emb。
    - 嵌入评分：TransEModel / ComplExModel.score_triple
    """

    def __init__(
        self,
        transe_model: TransEModel,
        graph,
        *,
        complex_model: Optional[ComplExModel] = None,
        config: Optional[StructureAgentConfig] = None,
        device: Optional[torch.device] = None,
        llm_explainer: Optional[StructureExplainer] = None,
        id2entity: Optional[Dict[int, str]] = None,
        id2relation: Optional[Dict[int, str]] = None,
        save_explanation_path: Optional[str] = None,
    ) -> None:
        self.transe_model = transe_model
        self.complex_model = complex_model
        self.graph = graph
        self.device = device or next(transe_model.parameters()).device

        self.config = config or StructureAgentConfig()

        self.llm_explainer = llm_explainer
        self.id2entity = id2entity or {}
        self.id2relation = id2relation or {}
        self.save_explanation_path = save_explanation_path

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------

    def score_triple(self, triple: TripleId) -> Dict[str, Any]:
        """
        对单条三元组 (h,r,t) 计算结构置信度及子分数；若配置了 LLM 解释器则生成
        可解释说明并写入 result["explanation"]，可选持久化到 JSONL 供融合智能体使用。
        """
        h, r, t = triple

        # 1) 结构嵌入分（此处“结构智能体”最终融合时以 TransE 为主，符合你的要求）
        s_transe = float(self._score_transe_confidence(h, r, t))

        # 2) 拓扑路径支撑分（support_paths / has_path）
        s_topo = 0.0
        support_paths = 0
        has_path = 0.0
        if self.config.use_topo_path_support:
            s_topo, support_paths, has_path = self._score_topo_path_support(
                h,
                r,
                t,
                max_len=int(self.config.topo_max_path_len),
                max_paths=int(self.config.topo_count_max_paths),
                use_reachable_gate=bool(self.config.topo_use_reachable_gate),
                use_log_norm=bool(self.config.use_log_norm_for_topo_paths),
            )

        # 3) 融合：与 TransE 的分数加权融合为最终结构置信度
        w_transe = float(self.config.w_transe_for_struct)
        w_transe = max(0.0, min(1.0, w_transe))
        s_struct = w_transe * s_transe + (1.0 - w_transe) * float(s_topo)

        result: Dict[str, Any] = {
            "S_struct": float(s_struct),
            "S_transe": float(s_transe),
            "S_topo_paths": float(s_topo),
            "support_paths": float(support_paths),
            "has_path": float(has_path),
        }

        if self.llm_explainer is not None:
            try:
                ctx = self._embedding_similarity_context(h, r, t)
                if ctx:
                    result["embedding_similarity_context"] = ctx
                explanation = self.llm_explainer(
                    triple,
                    dict(result),
                    self.id2entity,
                    self.id2relation,
                )
                result["explanation"] = explanation
            except Exception:
                result["explanation"] = None

        if self.save_explanation_path:
            self._append_explanation_record(triple, result)

        return result

    def _score_transe_confidence(self, h: int, r: int, t: int) -> float:
        """TransE 结构置信度：sigmoid((center - f) / scale)。"""
        device = self.device
        h_t = torch.tensor([h], dtype=torch.long, device=device)
        r_t = torch.tensor([r], dtype=torch.long, device=device)
        t_t = torch.tensor([t], dtype=torch.long, device=device)

        center = float(self.config.transe_confidence_center)
        scale = max(float(self.config.transe_confidence_scale), 1e-6)
        with torch.no_grad():
            f = self.transe_model.score_triple(h_t, r_t, t_t)
            return float(torch.sigmoid((center - f) / scale).item())

    def _has_reachable_path(self, h: int, t: int, max_len: int) -> bool:
        """
        轻量可达性检查：判断 h 是否能在 <= max_len 跳内到达 t。
        不枚举路径，只做 BFS 早停，降低 count_support_paths 的触发频率。
        """
        if h == t:
            return True
        if self.graph is None:
            return False

        visited = {h}
        q = deque([(h, 0)])
        while q:
            node, d = q.popleft()
            if d >= max_len:
                continue
            # node 可能不在图中（例如噪声注入替换出的冷启动实体），此时视为不可扩展
            if node not in self.graph:
                continue
            # MultiDiGraph: out_edges(..., keys=True) 会返回 (u, v, key)
            for _, nxt, _ in self.graph.out_edges(node, keys=True):
                if nxt == t:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))
        return False

    def _score_topo_path_support(
        self,
        h: int,
        r: int,
        t: int,
        *,
        max_len: int,
        max_paths: int,
        use_reachable_gate: bool,
        use_log_norm: bool,
    ) -> Tuple[float, int, float]:
        """
        拓扑路径支撑分：
        - 若不可达（可选 reachable gate），support_paths=0
        - 否则用 count_support_paths 得到 support_paths（最多枚举 max_paths 条）
        - 将 support_paths 归一化到 [0,1]，并可返回 has_path（support_paths>0）
        """
        if self.graph is None:
            return 0.0, 0, 0.0

        reachable = True
        if use_reachable_gate:
            reachable = self._has_reachable_path(h, t, max_len=max_len)

        if not reachable:
            return 0.0, 0, 0.0

        support_paths = int(
            count_support_paths(
                self.graph,
                h,
                r,
                t,
                max_len=max_len,
                max_paths=max_paths,
            )
        )
        has_path = 1.0 if support_paths > 0 else 0.0

        max_paths_f = max(float(max_paths), 1.0)
        if use_log_norm:
            # 使用 log1p 做平滑，避免大计数把分数压扁
            s_topo = math.log1p(float(support_paths)) / math.log1p(max_paths_f)
        else:
            s_topo = min(float(support_paths), max_paths_f) / max_paths_f

        s_topo = max(0.0, min(1.0, float(s_topo)))
        return float(s_topo), support_paths, float(has_path)

    def _append_explanation_record(
        self, triple: TripleId, result: Dict[str, Any]
    ) -> None:
        """将当前三元组的结构评分与可解释说明追加写入 JSONL，供融合智能体读取。"""
        path = Path(self.save_explanation_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "triple": list(triple),
            "S_struct": result.get("S_struct"),
            "explanation": result.get("explanation"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 嵌入相似度上下文（供 LLM 解释器可选使用）
    # ------------------------------------------------------------------

    def _embedding_similarity_context(self, h: int, r: int, t: int) -> str:
        """
        计算 head+rel 与 tail 的余弦相似度，返回简短描述供 prompt 使用。
        若无法计算则返回空字符串。
        """
        try:
            device = self.device
            h_t = torch.tensor([h], dtype=torch.long, device=device)
            r_t = torch.tensor([r], dtype=torch.long, device=device)
            t_t = torch.tensor([t], dtype=torch.long, device=device)
            with torch.no_grad():
                h_e, r_e, t_e = self.transe_model._get_embeddings(h_t, r_t, t_t)
                hr = (h_e + r_e).squeeze(0)
                tt = t_e.squeeze(0)
                cos = torch.nn.functional.cosine_similarity(
                    hr.unsqueeze(0), tt.unsqueeze(0), dim=1
                ).item()
            return f"head+rel 与 tail 的余弦相似度: {cos:.3f}"
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 各子分数计算
    # ------------------------------------------------------------------

    def _score_embedding(self, h: int, r: int, t: int) -> float:
        device = self.device
        h_t = torch.tensor([h], dtype=torch.long, device=device)
        r_t = torch.tensor([r], dtype=torch.long, device=device)
        t_t = torch.tensor([t], dtype=torch.long, device=device)
        mode = (self.config.emb_mode or "transe").lower()

        center = float(self.config.transe_confidence_center)
        scale = max(float(self.config.transe_confidence_scale), 1e-6)
        with torch.no_grad():
            if mode == "complex":
                if self.complex_model is None:
                    f = self.transe_model.score_triple(h_t, r_t, t_t)
                    return torch.sigmoid((center - f) / scale).item()
                comp = self.complex_model.score_triple(h_t, r_t, t_t)
                return complex_struct_confidence(comp).item()

            f = self.transe_model.score_triple(h_t, r_t, t_t)
            s_transe = torch.sigmoid((center - f) / scale).item()

            if mode == "transe":
                return s_transe

            # mode == "fusion"
            if self.complex_model is None:
                return s_transe
            complex_score = self.complex_model.score_triple(h_t, r_t, t_t)
            s_complex = complex_struct_confidence(complex_score).item()
            alpha = float(self.config.alpha)
            return alpha * s_transe + (1.0 - alpha) * s_complex
