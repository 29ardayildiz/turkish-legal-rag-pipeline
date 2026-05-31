# System Architecture

## Pipeline Overview

```
Turkish Legal Question
        │
        ▼
┌───────────────┐
│  Query Encode │  BGE-M3 (base or FT)
└───────┬───────┘
        │
   ┌────┴─────┐
   │          │
   ▼          ▼
┌──────┐  ┌───────┐
│FAISS │  │ BM25  │   Dense + Sparse retrieval
│Dense │  │Sparse │   Top-20 each
└──┬───┘  └───┬───┘
   │           │
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │ RRF Fusion│   Reciprocal Rank Fusion (k=60)
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │  Reranker │   Cross-Encoder (base or FT)   [optional]
   │    CE     │   Top-20 → Top-5
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │    LLM    │   Qwen2.5-7B-Instruct (base or QLoRA)
   │  Qwen2.5  │   Max 128/256 new tokens
   └─────┬─────┘
         │
         ▼
    Legal Answer
  (with citations)
```

## Component Details

### Embedding Layer
- **Model:** `BAAI/bge-m3`
- **Dimension:** 1024
- **Index:** FAISS `IndexFlatIP` (inner product / cosine)
- **FT Loss:** `MultipleNegativesRankingLoss`
- **FT Data:** ~1949 (query, positive_passage) pairs

### Sparse Layer
- **Library:** `rank_bm25`
- **Tokenization:** Whitespace + punctuation strip
- **Language:** Turkish (no special stemmer, morphology not applied)

### Fusion
- **Method:** Reciprocal Rank Fusion (RRF)
- **Formula:** `score(d) = Σ 1/(k + rank(d))` where k=60
- **Input:** Top-20 dense + Top-20 sparse
- **Output:** Top-20 fused candidates

### Reranker
- **Base:** `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Task:** Binary relevance classification
- **FT:** BCE loss with `pos_weight=1.69`
- **Input:** Top-20 → **Output:** Top-5

### LLM
- **Base:** `Qwen/Qwen2.5-7B-Instruct`
- **Quantization:** 4-bit (bitsandbytes NF4)
- **FT:** QLoRA via PEFT + TRL SFT
- **LoRA rank:** 16, alpha: 32, target: q/v projections
- **Prompt format:** System + context passages + question

## Evaluation Framework

### Retrieval Metrics
- **Recall@5, @10** — is the relevant doc in top-k?
- **MRR** — Mean Reciprocal Rank
- **nDCG** — Normalized Discounted Cumulative Gain

### QA Metrics
- **EM** — Exact Match (normalized)
- **F1** — Token-level F1
- **ROUGE-L** — Longest Common Subsequence
- **BLEU** — N-gram precision
- **Faithfulness** — Answer grounded in context?
- **Citation Accuracy** — Correct source cited?

## Hardware

All experiments run on **Kaggle Free Tier: T4 x2 (2× 16GB VRAM)**

| Stage | Notebook | Est. Time |
|-------|----------|-----------|
| Embedding FT | P1 | ~45 min |
| Reranker FT | P2 (part 1) | ~30 min |
| LLM QLoRA | P2 (part 2) | ~90 min |
| Model merge | P3 | ~10 min |
| Each ablation (S1–S8) | S* | ~20–40 min |
| **Total** | — | **~5–6 hours** |
