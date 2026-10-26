# MAC4E: Multi-Agent Collaboration for Error Knowledge Detection

MAC4E detects erroneous triples in knowledge graphs. Three specialist agents score each triple from a structural, a factual and a logical perspective, and a learned allocator decides, per triple, how much to trust each of them.

This repository contains the code of my undergraduate thesis at the School of Artificial Intelligence, Beihang University (2026), supervised by Prof. Guanglin Niu.

## Overview

Knowledge graphs built by automatic extraction contain wrong facts. Most existing detectors rely on a single signal (embedding geometry, text semantics or graph structure) and combine evidence with fixed weights. MAC4E uses several signals and adapts their weights to each triple:

1. **Pretraining** – trains TransE embeddings and an RGCN subgraph encoder on the target graph. They provide shared features (embedding scores, path support, entity frequency) for the agents.
2. **Task allocation** – a small MLP predicts, for each triple, a weight for each of the three perspectives. It is trained with weak supervision: rule-based pseudo-labels and a KL-divergence loss.
3. **Multi-perspective verification**
   - **Structural agent** – combines the TransE distance with topological path support between head and tail.
   - **Factual agent** – turns the triple into text and asks an LLM (Qwen2.5-7B-Instruct by default) whether it is plausible. A default score is used if the answer cannot be parsed.
   - **Logical agent** – extracts the k-hop subgraph around the triple and scores its relational consistency with the RGCN scorer.
4. **Fusion decision** – starts from the allocator-weighted sum of the three scores. An LLM then re-weights the agents when they disagree, using their scores and explanations. If the LLM call fails, the weighted sum is kept.

## Results

Setting: FB15K-237 and WN18RR with 5% injected noise. Triples are ranked by final score (lowest = most suspicious), and Precision@K is the share of real errors among the K% lowest-scored triples. Baseline numbers are taken from the thesis.

**FB15K-237 – Precision@K**

| Method | 1% | 2% | 3% | 4% | 5% |
|---|---|---|---|---|---|
| TransE | 0.756 | 0.674 | 0.605 | 0.546 | 0.488 |
| CAGED | 0.852 | 0.796 | 0.735 | 0.665 | 0.595 |
| DKRL | 0.823 | 0.752 | 0.677 | 0.613 | 0.561 |
| JointSTC | 0.870 | 0.815 | 0.753 | 0.689 | 0.623 |
| **MAC4E** | **0.952** | **0.910** | **0.858** | **0.749** | **0.655** |

**WN18RR – Precision@K**

| Method | 1% | 2% | 3% | 4% | 5% |
|---|---|---|---|---|---|
| TransE | 0.581 | 0.488 | 0.371 | 0.345 | 0.331 |
| CAGED | 0.847 | 0.702 | 0.608 | 0.537 | 0.482 |
| ADKGD | 0.943 | 0.845 | 0.724 | 0.635 | 0.560 |
| SeSICL | 0.953 | 0.867 | 0.756 | 0.681 | 0.612 |
| **MAC4E** | **0.960** | **0.872** | **0.803** | **0.777** | **0.669** |

The thesis reports more baselines (DistMult, ComplEx, CKRL, KGTtm, KGIst, and others). In the ablation study, removing the factual agent causes the largest precision drop at mid-range K, and replacing the learned allocation with fixed equal weights lowers precision at small K.

## Repository structure

```
.
├── ccg_agents/      # allocator, structural / factual / logical agents, fusion agents, LLM client
├── ccg_data/        # dataset loading, noise injection, path mining, ID-to-text mapping
├── ccg_models/      # TransE, ComplEx, RGCN subgraph encoder
├── ccg_pipeline/    # training: embeddings, RGCN encoder, allocator
├── ccg_utils/       # feature construction, score normalization
├── ccg_config/      # dataset paths and hyperparameters (datasets.yaml, model.yaml)
├── examples/        # end-to-end runs, ablations, single-agent experiments, baselines
├── scripts/         # Top-K tables, plots, data/checkpoint consistency check
└── tests/           # sanity checks and link-prediction evaluation
```

Main entry points:

| Dataset | Script |
|---|---|
| FB15K-237 | `examples/run_end2end_multi_agent_error_detection_fb15k237_rgcn_logic_only.py` |
| WN18RR | `examples/run_end2end_multi_agent_error_detection_wn18rr_rgcn_logic_fusion_llm.py` |

Ablations: `*_wn18rr_fact_only.py`, `*_wn18rr_logic_only.py`, `*_wn18rr_transe_only.py`, `*_fb15k237_struct_fact_only.py`, `*_fb15k237_rgcn_logic_only_no_allocator.py`.

## Setup

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/yike-shi/MAC4E.git
cd MAC4E
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The factual and fusion agents run a local LLM through Hugging Face `transformers` (default `Qwen/Qwen2.5-7B-Instruct`, about 15 GB of GPU memory). To load the model from a local folder instead of downloading it:

```bash
export LOCAL_LLM_PATH=/path/to/Qwen2.5-7B-Instruct
```

Without a GPU, use `--no_llm` (and `--no_fusion_llm` for WN18RR) to run the pipeline with the LLM parts switched off. The resulting numbers are then not comparable to the table above.

### Data

Place the datasets as configured in `ccg_config/datasets.yaml`:

```
data/FB15k/fb15k237-{train,valid,test}.txt   + entity2text.txt, relation2text.txt
data/WN18RR/{train,valid,test}.txt           + entity2text.txt, relation2text.txt
```

Entity and relation IDs are assigned in order of first appearance, so pretrained checkpoints only work with the exact same files. Check a dataset copy against a checkpoint with:

```bash
python scripts/check_data_alignment.py --ckpt outputs/embeddings/fb15k237_transE.pt \
  --train data/FB15k/fb15k237-train.txt --valid data/FB15k/fb15k237-valid.txt --test data/FB15k/fb15k237-test.txt
```

## Usage

```bash
# 1. Pretrain embeddings and the RGCN encoder
python ccg_pipeline/train_embeddings.py --train fb15k237 --model transe
python ccg_pipeline/train_gcn_encoder.py --dataset fb15k237

# 2. Train the task allocator (WN18RR has its own pseudo-label rules)
python ccg_pipeline/train_allocator.py --dataset fb15k237
python examples/train_allocator_wn18rr.py

# 3. End-to-end error detection
python examples/run_end2end_multi_agent_error_detection_fb15k237_rgcn_logic_only.py \
  --gcn_ckpt outputs/gcn/fb15k237_gcn_new_full.pt \
  --allocator_ckpt outputs/allocator/fb15k237_allocator_154.pt \
  --fusion_llm_reliability --agent_score_norm zscore

python examples/run_end2end_multi_agent_error_detection_wn18rr_rgcn_logic_fusion_llm.py \
  --allocator_ckpt outputs/allocator/wn18rr_allocator_351.pt
```

Add `--max_triples N` for a quick run on a subset. Each run logs a Precision@K / Recall@K table to `outputs/logs/`.

Code comments and log messages are mostly in Chinese.

## Citation

```bibtex
@thesis{shi2026mac4e,
  title  = {Research on Erroneous Knowledge Detection Based on Multi-Agent Cooperation},
  author = {Shi, Yike},
  school = {Beihang University},
  type   = {Bachelor's thesis},
  year   = {2026}
}
```

## Acknowledgements

Thanks to Prof. Guanglin Niu for supervising this work.

## License

[MIT](LICENSE)
