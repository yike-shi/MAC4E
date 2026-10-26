"""
端到端多智能体错误检测（WN18RR）：
- 结构 / 事实 / 逻辑 / 融合与 FB15K-237 版一致：逻辑仅 RGCN，融合可选 LLM；默认三路 zscore。
- 数据与权重路径由 --dataset（默认 wn18rr）与 ccg_config/datasets.yaml 决定。

Checkpoint 约定（outputs 根目录下）：
- embeddings/{dataset}_transE.pt
- gcn/{dataset}_gcn.pt（可用 --gcn_ckpt 覆盖）
- allocator/{dataset}_allocator.pt（可用 --allocator_ckpt 覆盖）
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_data.kg_dataset import KGDataset
from ccg_data.noise_injector import inject_noise
from ccg_data.path_mining import build_graph
from ccg_data.text_mapping import load_wn18rr_text_maps
from ccg_utils.agent_score_norm import normalize_score_channel
from ccg_utils.kg_utils import build_entity_frequency
from ccg_models.gcn_encoder import build_adj_from_triples
from ccg_pipeline.train_embeddings import (
    load_complex_from_checkpoint,
    load_config,
    load_model_config,
    load_transe_from_checkpoint,
)
from ccg_agents.fact_agent import FactAgentConfig, FactVerificationAgent
from ccg_agents.fusion_agent import FusionAgentConfig, FusionDecisionAgent
from ccg_agents.structure_agent import StructureAgentConfig, StructureConstraintAgent
from ccg_data.type_constraints import TypeConstraints

from run_end2end_multi_agent_error_detection_fb15k237_rgcn_logic_only import (
    _allocator_weights_for_triple,
    _compute_s_logic_rgcn,
    _create_llm_client,
    _format_prk_table,
    _load_allocator,
    _load_gcn_if_available,
    _setup_logger,
    precision_recall_at_percents_for_errors,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it


TripleId = Tuple[int, int, int]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WN18RR 端到端错误检测（逻辑仅RGCN + 融合LLM + 三路 zscore）"
    )
    parser.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr"])
    parser.add_argument("--noise_ratio", type=float, default=0.05, choices=[0.05, 0.10, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_triples", type=int, default=None)
    parser.add_argument("--out_metrics", type=str, default=None)
    parser.add_argument("--out_jsonl", type=str, default=None)
    parser.add_argument(
        "--explain_bottom_percent",
        type=float,
        default=100.0,
        help="仅在 --out_jsonl 时生效：按 final_score 升序仅导出最低分底部 x%%（默认 100%% 即全导出）",
    )
    parser.add_argument("--percents", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--no_gcn", action="store_true", help="禁用逻辑RGCN（此时逻辑分固定0.5）")
    parser.add_argument("--no_llm", action="store_true", help="禁用事实LLM（事实分固定0.5）")
    parser.add_argument("--no_fusion_llm", action="store_true", help="禁用融合LLM，回退为纯加权平均")
    parser.add_argument(
        "--allocator_ckpt",
        type=str,
        default=None,
        help="默认 outputs/allocator/{dataset}_allocator.pt",
    )
    parser.add_argument(
        "--transe_ckpt",
        type=str,
        default=None,
        help="默认 outputs/embeddings/{dataset}_transE.pt",
    )
    parser.add_argument(
        "--gcn_ckpt",
        type=str,
        default=None,
        help="默认 outputs/gcn/{dataset}_gcn.pt",
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument(
        "--logic_temperature",
        type=float,
        default=8.0,
        help="逻辑分温度 T：s=sigmoid(logit/T)，T>1 可减少分数挤压到 0/1",
    )
    parser.add_argument(
        "--score_eps",
        type=float,
        default=1e-6,
        help="逻辑分截断下界/上界：s=clip(s, eps, 1-eps)，避免精确 0/1",
    )
    parser.add_argument(
        "--agent_score_norm",
        type=str,
        default="zscore",
        choices=["none", "minmax", "zscore"],
        help="融合前对本 run 全部样本三路分别列归一化；默认 zscore（Φ(z)）。",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    model_cfg = load_model_config()
    base = _ROOT / cfg.get("output_dir", "./outputs").lstrip("./")
    ds_key = args.dataset.lower()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        Path(args.log_file)
        if args.log_file
        else base / "logs" / f"end2end_{ds_key}_rgcn_logic_fusion_llm_{ts}.log"
    )
    log = _setup_logger(log_path)
    log.info("=" * 72)
    log.info("端到端运行（WN18RR，逻辑仅RGCN + 融合LLM）启动")
    log.info("log_file=%s", log_path.resolve())
    log.info("CLI=%s", json.dumps(vars(args), ensure_ascii=False, default=str))
    log.info("logic_temperature=%.4f score_eps=%.8f", float(args.logic_temperature), float(args.score_eps))
    if args.agent_score_norm != "none":
        log.info("agent_score_norm=%s（三路同一规则，融合用）", args.agent_score_norm)

    ds_cfg = cfg.get(ds_key)
    if not isinstance(ds_cfg, dict):
        raise KeyError(f"未找到 {ds_key} 配置（ccg_config/datasets.yaml）")
    ds = KGDataset.load_from_files(
        str(_ROOT / ds_cfg["train_path"].lstrip("./")),
        str(_ROOT / ds_cfg["valid_path"].lstrip("./")),
        str(_ROOT / ds_cfg["test_path"].lstrip("./")),
    )

    clean_all: List[TripleId] = list(ds.train_triples) + list(ds.valid_triples) + list(ds.test_triples)
    if args.max_triples is not None and args.max_triples > 0 and len(clean_all) > args.max_triples:
        clean_all = random.sample(clean_all, args.max_triples)
    clean_triples, noisy_triples, labeled = inject_noise(
        clean_all,
        num_entities=ds.num_entities,
        noise_ratio=float(args.noise_ratio),
        mode="head_tail",
    )
    graph = build_graph(list(clean_triples) + list(noisy_triples))
    log.info("N=%d clean=%d noisy=%d", len(labeled), len(clean_triples), len(noisy_triples))

    if args.transe_ckpt:
        tp = Path(args.transe_ckpt)
        transe_ckpt = tp if tp.is_absolute() else (_ROOT / str(tp).lstrip("./"))
    else:
        transe_ckpt = base / "embeddings" / f"{ds_key}_transE.pt"
    if not transe_ckpt.exists():
        raise FileNotFoundError(f"未找到 TransE checkpoint: {transe_ckpt}")
    transe = load_transe_from_checkpoint(str(transe_ckpt), device=device)
    transe.eval()
    entity_emb = transe.ent_emb.weight.detach()
    rel_emb = transe.rel_emb

    complex_model = None
    complex_ckpt = base / "embeddings" / f"{ds_key}_complEx.pt"
    if complex_ckpt.exists():
        complex_model = load_complex_from_checkpoint(str(complex_ckpt), device=device)
        complex_model.eval()

    gcn_ckpt_resolved: Optional[Path] = None
    gcn_ctx = None
    adj_for_gcn = None
    if not args.no_gcn:
        if args.gcn_ckpt:
            gp = Path(args.gcn_ckpt)
            gcn_ckpt_resolved = gp if gp.is_absolute() else (_ROOT / str(gp).lstrip("./"))
        else:
            gcn_ckpt_resolved = base / "gcn" / f"{ds_key}_gcn.pt"
        gcn_ctx = _load_gcn_if_available(gcn_ckpt_resolved, device, model_cfg)
    if gcn_ctx is not None:
        adj_for_gcn = build_adj_from_triples(ds.train_triples)
        _enc, _sc, _st, k_hop, _bd = gcn_ctx
        log.info("逻辑RGCN已启用: k_hop=%d ckpt=%s", k_hop, gcn_ckpt_resolved.resolve() if gcn_ckpt_resolved else "")
    else:
        log.info("逻辑RGCN未启用（--no_gcn 或 checkpoint缺失/加载失败），逻辑分将使用0.5默认值。")

    if args.allocator_ckpt:
        p = Path(args.allocator_ckpt)
        allocator_ckpt = p if p.is_absolute() else (_ROOT / str(p).lstrip("./"))
    else:
        allocator_ckpt = base / "allocator" / f"{ds_key}_allocator.pt"
    if not allocator_ckpt.exists():
        raise FileNotFoundError(f"未找到 Allocator checkpoint: {allocator_ckpt}")
    allocator, feat_mean, feat_std, alloc_meta = _load_allocator(allocator_ckpt, device=device)
    emb_mode = str(alloc_meta.get("emb_model", "transe")).lower()
    fusion_alpha = float(alloc_meta.get("fusion_alpha", 0.5))
    log.info("Allocator=%s emb_mode=%s", allocator_ckpt.resolve(), emb_mode)

    type_constraints = TypeConstraints()
    type_constraints.build_from_triples(clean_triples)
    entity_freq = build_entity_frequency(clean_triples)

    llm_client, llm_src = _create_llm_client(prefer_local_qwen=not args.no_llm)
    log.info("事实LLM=%s", "禁用(--no_llm)" if args.no_llm else (llm_src or "Mock (fallback)"))
    log.info("融合LLM=%s", "禁用(--no_fusion_llm)" if args.no_fusion_llm else "启用")

    struct_agent = StructureConstraintAgent(
        transe_model=transe,
        graph=graph,
        complex_model=complex_model,
        config=StructureAgentConfig(emb_mode="transe"),
        llm_explainer=None,
        id2entity=ds.id2entity,
        id2relation=ds.id2relation,
    )
    data_root = _ROOT / "data" / "WN18RR"
    try:
        ent2text, rel2text = load_wn18rr_text_maps(root=str(data_root))
    except Exception as e:  # pragma: no cover
        log.warning("WN18RR 文本映射加载失败，退回空映射: %s", e)
        ent2text, rel2text = {}, {}
    if not ent2text:
        log.info("事实分支：无 entity2text 或合成表为空，FactAgent 将主要依赖 id2entity/id2relation。")
    fact_agent = FactVerificationAgent(
        llm_client=llm_client,
        id2entity=ds.id2entity,
        id2relation=ds.id2relation,
        ent2text=ent2text,
        rel2text=rel2text,
        config=FactAgentConfig(default_score_no_readable=0.5, skip_fact_when_no_readable=True, max_tokens=300),
        save_result_path=None,
    )
    fusion_agent = FusionDecisionAgent(
        llm_client=llm_client,
        config=FusionAgentConfig(use_llm=(not args.no_fusion_llm), max_tokens=350),
        save_result_path=None,
    )

    triples_all: List[TripleId] = [(h, r, t) for (h, r, t, _y) in labeled]
    labels_all: List[int] = [int(_y) for (_h, _r, _t, _y) in labeled]
    packed: List[Tuple[List[float], Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []

    t0 = time.perf_counter()
    for idx, triple in enumerate(tqdm(triples_all, desc="Scoring (fact LLM + logic RGCN + fusion LLM)", unit="triple"), start=1):
        w = _allocator_weights_for_triple(
            triple,
            allocator=allocator,
            feat_mean=feat_mean,
            feat_std=feat_std,
            transe_model=transe,
            device=device,
            graph=graph,
            entity_freq=entity_freq,
            type_constraints=type_constraints,
            emb_mode=emb_mode,
            complex_model=complex_model,
            fusion_alpha=fusion_alpha,
            max_path_len=3,
        )
        s_res = struct_agent.score_triple(triple)
        if args.no_llm:
            f_res = {"S_fact": 0.5, "explanation": None, "skipped": True, "readable_triple": None}
        else:
            f_res = fact_agent.score_triple(triple)

        s_rgcn = _compute_s_logic_rgcn(
            triple,
            gcn_ctx=gcn_ctx,
            adj=adj_for_gcn,
            entity_emb=entity_emb,
            rel_emb=rel_emb,
            device=device,
            model_cfg=model_cfg,
            temperature=float(args.logic_temperature),
            score_eps=float(args.score_eps),
        )
        s_logic = float(s_rgcn) if s_rgcn is not None else 0.5
        l_res = {
            "S_logic": s_logic,
            "S_logic_llm": None,
            "S_logic_rgcn": s_rgcn,
            "beta": 0.0 if s_rgcn is not None else None,
            "explanation": "[logic-rgcn-only] 使用RGCN分数；未调用逻辑LLM。",
            "path_count": None,
        }
        packed.append((w, s_res, f_res, l_res))

        if args.log_every > 0 and idx % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            log.info(
                "[收集进度] %d/%d (%.1f%%) elapsed=%.1fs speed=%.2f triple/s",
                idx,
                len(triples_all),
                100.0 * idx / len(triples_all),
                elapsed,
                idx / elapsed if elapsed > 0 else 0.0,
            )

    norm_mode = str(args.agent_score_norm)
    struct_raw = [float(p[1].get("S_struct", 0.0)) for p in packed]
    fact_raw = [float(p[2].get("S_fact", 0.5)) for p in packed]
    logic_raw = [float(p[3].get("S_logic", 0.5)) for p in packed]
    if norm_mode != "none":
        struct_n = normalize_score_channel(struct_raw, norm_mode)
        fact_n = normalize_score_channel(fact_raw, norm_mode)
        logic_n = normalize_score_channel(logic_raw, norm_mode)
    else:
        struct_n = fact_n = logic_n = []

    final_scores: List[float] = []
    records: List[Dict[str, Any]] = []
    t1 = time.perf_counter()
    for idx, (w, s_res, f_res, l_res) in enumerate(packed, start=1):
        triple = triples_all[idx - 1]
        if norm_mode in ("minmax", "zscore"):
            s_use = {**s_res, "S_struct": struct_n[idx - 1]}
            f_use = {**f_res, "S_fact": fact_n[idx - 1]}
            l_use = {**l_res, "S_logic": logic_n[idx - 1]}
        else:
            s_use, f_use, l_use = s_res, f_res, l_res
        readable = f_use.get("readable_triple")
        fused = fusion_agent.decide(w, s_use, f_use, l_use, readable_triple=readable)
        final_scores.append(float(fused["final_score"]))

        if args.out_jsonl:
            rec: Dict[str, Any] = {
                "triple": list(triple),
                "label": int(labels_all[idx - 1]),
                "final_score": float(fused["final_score"]),
                "weighted_score": float(fused["weighted_score"]),
                "fusion_used_llm": bool(fused.get("used_llm")),
                "fusion_explanation": fused.get("explanation"),
                "allocator_weights": fused.get("allocator_weights"),
                "S_struct": s_use.get("S_struct"),
                "S_fact": f_use.get("S_fact"),
                "S_logic": l_use.get("S_logic"),
                "S_logic_rgcn": l_res.get("S_logic_rgcn"),
                "agent_score_norm": norm_mode,
            }
            if norm_mode != "none":
                rec["S_struct_raw"] = struct_raw[idx - 1]
                rec["S_fact_raw"] = fact_raw[idx - 1]
                rec["S_logic_raw"] = logic_raw[idx - 1]
            records.append(rec)

        if args.log_every > 0 and idx % args.log_every == 0:
            elapsed = time.perf_counter() - t1
            log.info(
                "[融合进度] %d/%d (%.1f%%) elapsed=%.1fs speed=%.2f triple/s last_final=%.4f",
                idx,
                len(packed),
                100.0 * idx / len(packed),
                elapsed,
                idx / elapsed if elapsed > 0 else 0.0,
                final_scores[-1],
            )

    percents = [int(x) for x in args.percents]
    metrics = precision_recall_at_percents_for_errors(labels_all, final_scores, percents=percents)
    table = _format_prk_table(metrics, percents)
    print("\n" + "=" * 70)
    print("End-to-end Multi-Agent Error Detection (RGCN-logic + Fusion-LLM)")
    print("=" * 70)
    print(
        f"dataset={ds_key}  noise_ratio={args.noise_ratio}  N={metrics['n_total']}  F(errors)={metrics['n_errors']}"
    )
    print(table)
    print("=" * 70)
    log.info("评估结果:\n%s", table)

    if args.out_jsonl:
        bottom_p = max(0.0, min(100.0, float(args.explain_bottom_percent)))
        n_total = len(records)
        n_bottom = int((n_total * bottom_p + 99.0) // 100.0) if bottom_p > 0 else 0
        n_bottom = max(0, min(n_bottom, n_total))
        if n_bottom < n_total:
            order = sorted(range(n_total), key=lambda i: float(records[i]["final_score"]))
            keep = set(order[:n_bottom])
            records = [rec for i, rec in enumerate(records) if i in keep]

        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info(
            "逐条打分已写入: %s（bottom %.2f%%, n=%d）",
            out_path.resolve(),
            bottom_p,
            len(records),
        )

    if args.out_metrics:
        mpath = Path(args.out_metrics)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        gcn_ckpt_str = str(gcn_ckpt_resolved.resolve()) if gcn_ckpt_resolved is not None else ""
        payload = {
            "dataset": ds_key,
            "noise_ratio": float(args.noise_ratio),
            "seed": int(args.seed),
            "n_clean": len(clean_triples),
            "n_noisy": len(noisy_triples),
            "n_total": metrics["n_total"],
            "n_errors": metrics["n_errors"],
            "percents": percents,
            "metrics": {k: v for k, v in metrics.items() if k not in ("percents",)},
            "logic_mode": "rgcn_only",
            "fusion_mode": "llm_or_weighted_fallback",
            "allocator": {"checkpoint": str(allocator_ckpt.resolve()), **alloc_meta},
            "structure_agent_config": asdict(struct_agent.config),
            "llm": {
                "fact_disabled": bool(args.no_llm),
                "fusion_disabled": bool(args.no_fusion_llm),
                "source": llm_src,
            },
            "logic_gcn": {
                "enabled": gcn_ctx is not None and not args.no_gcn,
                "checkpoint": gcn_ckpt_str,
                "logic_temperature": float(args.logic_temperature),
                "score_eps": float(args.score_eps),
            },
            "agent_score_norm": str(args.agent_score_norm),
            "run_log": str(log_path.resolve()),
            "transe_checkpoint": str(transe_ckpt.resolve()),
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("指标已写入: %s", mpath.resolve())

    log.info("全部完成。")


if __name__ == "__main__":
    main()
