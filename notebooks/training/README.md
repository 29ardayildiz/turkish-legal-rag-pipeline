# Training Notebooks

These notebooks are designed to run on **Kaggle (T4 x2 GPU)**.  
Run them in order: P1 → P2 → P3.

## Execution Order

### Step 1 — P1_EmbeddingFT_v2.ipynb
**BGE-M3 Embedding Domain Adaptation**

- Input: `ardayildiz29/legalo` → `embedding.jsonl`
- Output: Fine-tuned BGE-M3 weights → Kaggle dataset
- Loss: `MultipleNegativesRankingLoss`
- Duration: ~45 min on T4

**Key changes from v1:**
- `TripletLoss` → `MultipleNegativesRankingLoss` (matches BGE-M3 pre-training)
- Max seq len: 128 → 512
- Full dataset: 1949 examples (was 2000 sample)

---

### Step 2 — p2-trainingmodels-v2.ipynb
**Reranker Fine-Tuning + Qwen QLoRA SFT** (two steps in one notebook)

- Input: `ardayildiz29/legalo` → `reranker.jsonl` + `llm.jsonl`
- Output: `reranker_ft/` + `qwen_qlora_ft/` → Kaggle dataset
- Duration: ~30 min (reranker) + ~90 min (QLoRA)

**Reranker key fixes:**
- `pos_weight=1.69` for class imbalance (pos:neg = 1:1.7)
- Minimum query length filter: ≥3 words

**LLM key fixes:**
- `max_seq_length=1024` (prevents OOM)
- `max_grad_norm=1.0` (prevents loss spikes)
- 5000 training samples (was 2000)

---

### Step 3 — P3_ConsolidateModels_v3.ipynb
**Model Consolidation**

Merges all fine-tuned model versions into a single structured Kaggle dataset:

```
legalo_models_merged/
├── bgem3/           ← from dataset version 1
├── reranker_ft/     ← from dataset version 2
└── qwen_qlora_ft/   ← from dataset version 2
```

Run this if model versions are scattered across multiple Kaggle dataset uploads.

## Kaggle Dataset Dependencies

| Notebook | Input Dataset | Output Dataset |
|----------|--------------|----------------|
| P1 | `ardayildiz29/legalo` | `ardayildiz29/legal-models-tr-finetuned` (v1) |
| P2 | `ardayildiz29/legalo` | `ardayildiz29/legal-models-tr-finetuned` (v2) |
| P3 | `ardayildiz29/legal-models-tr-finetuned` | Merged version |
