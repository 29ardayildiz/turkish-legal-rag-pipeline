# Ablation Study Notebooks

8 ablation systems comparing incremental optimizations of the RAG pipeline.  
All notebooks run on **Kaggle (T4 x2 GPU)**.

## Prerequisites

Before running any ablation notebook, ensure these Kaggle datasets are attached:
1. `ardayildiz29/legalo` — corpus, gold benchmark, training data
2. `ardayildiz29/legal-models-tr-finetuned` — fine-tuned model weights (from training notebooks)

## Systems Overview

| Notebook | Pipeline | What changes vs previous |
|----------|----------|--------------------------|
| S1_Baseline | Base BGE-M3 → Dense → Base Qwen | — (starting point) |
| S2_EmbedFT | **FT** BGE-M3 → Dense → Base Qwen | Embedding: base → FT |
| S3_SparseOnly | BM25 → Base Qwen | Retrieval: dense → sparse only |
| S4_Hybrid | FT BGE-M3 + BM25 → **RRF** → Base Qwen | Adds BM25 + fusion to S2 |
| S5_BaseReranker | Hybrid → **Base CE** → Base Qwen | Adds base cross-encoder |
| S6_FTReranker | Hybrid → **FT CE** → Base Qwen | Reranker: base → FT |
| S7_LLMFT | Hybrid → **QLoRA Qwen** | LLM: base → QLoRA (no reranker) |
| S8_Full | Hybrid → FT CE → **QLoRA Qwen** | Everything FT |

## Shared Config Parameters

All systems use:
- `BENCH_SIZE = 240` (gold benchmark questions)
- `TOP_K_FINAL = 5` (retrieved documents passed to LLM)
- `TOP_K_DENSE/SPARSE = 20` (candidates before reranking/fusion)
- `RRF_K = 60` (Reciprocal Rank Fusion constant)
- `MAX_NEW_TOKENS = 128` (S1–S6, S8) / `256` (S7)
- `RANDOM_SEED = 42`

## Results

All outputs are saved to `/kaggle/working/legal_rag/results/` as JSON files:
- `{SYSTEM_NAME}_results.json` — per-question results
- `{SYSTEM_NAME}_metrics.json` — aggregated evaluation metrics
- `{SYSTEM_NAME}_config.json` — hyperparameter snapshot

## Expected Metric Trend

```
S1 (Baseline) < S2 (EmbedFT) < S3 (SparseOnly) [?] < S4 (Hybrid)
             < S5 (BaseReranker) < S6 (FTReranker) < S7 (LLMFT) < S8 (Full)
```
S3 is a diagnostic baseline — BM25 alone is not expected to beat dense retrieval.
