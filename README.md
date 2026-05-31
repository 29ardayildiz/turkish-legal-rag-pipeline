# Turkish Legal RAG Pipeline — CENG493

> Domain-adapted Retrieval-Augmented Generation system for Turkish legal question answering.  
> Minimizes hallucination through optimized embedding, reranking, and LLM components.

## Team Members

| Name | GitHub |
|------|--------|
| **Arda Yıldız** | [@29ardayildiz](https://github.com/29ardayildiz) |
| **Bengisu Yılmaz** | [@Bngsu](https://github.com/Bngsuyy) | 

## Project Overview

| Component | Detail |
|-----------|--------|
| **Task** | Turkish Legal Question Answering |
| **Approach** | Retrieval-Augmented Generation (RAG) |
| **Embedding** | BGE-M3 (base + domain fine-tuned) |
| **Retrieval** | Dense (FAISS) + Sparse (BM25) + Hybrid (RRF) |
| **Reranker** | Cross-Encoder ms-marco-MiniLM-L-12-v2 (base + FT) |
| **LLM** | Qwen2.5-7B-Instruct (base + QLoRA) |
| **Platform** | Kaggle (T4 x2 GPU) |

## System Architecture

```
Query
  │
  ├─► Dense Retrieval (FAISS + BGE-M3)  ─┐
  │                                       ├─► RRF Fusion ─► Reranker ─► Top-k ─► LLM ─► Answer
  └─► Sparse Retrieval (BM25)            ─┘
```

## Ablation Study — 8 Systems

| # | System | Embedding | Retrieval | Reranker | LLM |
|---|--------|-----------|-----------|----------|-----|
| S1 | Baseline | BGE-M3 | Dense | ✗ | Base Qwen |
| S2 | EmbedFT | **FT** BGE-M3 | Dense | ✗ | Base Qwen |
| S3 | SparseOnly | — | **BM25** | ✗ | Base Qwen |
| S4 | Hybrid | FT BGE-M3 | **Dense+BM25+RRF** | ✗ | Base Qwen |
| S5 | BaseReranker | FT BGE-M3 | Hybrid | **Base CE** | Base Qwen |
| S6 | FTReranker | FT BGE-M3 | Hybrid | **FT CE** | Base Qwen |
| S7 | LLMFT | FT BGE-M3 | Hybrid | ✗ | **QLoRA Qwen** |
| S8 | **Full** | FT BGE-M3 | Hybrid | FT CE | **QLoRA Qwen** |

## Repository Structure

```
notebooks/
├── training/          # Model fine-tuning notebooks (Kaggle)
│   ├── P1_EmbeddingFT_v2.ipynb       # BGE-M3 domain adaptation
│   ├── p2-trainingmodels-v2.ipynb    # Reranker FT + Qwen QLoRA SFT
│   └── P3_ConsolidateModels_v3.ipynb # Model consolidation & upload
└── ablation/          # 8 ablation study pipelines (Kaggle)
    ├── S1_Baseline.ipynb
    ├── S2_EmbedFT.ipynb
    ├── S3_SparseOnly.ipynb
    ├── S4_Hybrid.ipynb
    ├── S5_BaseReranker.ipynb
    ├── S6_FTReranker.ipynb
    ├── S7_LLMFT.ipynb
    └── S8_Full.ipynb
```

## Dataset

The training and evaluation data (`ardayildiz29/legalo`) was **custom-built** by combining and preprocessing 8 public Turkish legal sources:

[Turkish Law QA](https://huggingface.co/datasets/OrionCAF/turkish_law_qa_dataset) · [Turkish Law LLM FT](https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning) · [Turkish LawChatbot](https://huggingface.co/datasets/Renicames/turkish-lawchatbot) · [Constitutional Court](https://huggingface.co/datasets/KocLab-Bilkent/turkish-constitutional-court) · [Yargıtay Retrieval](https://github.com/koc-lab/yargitay_retrieval_dataset) · [Yargıtay 2025](https://huggingface.co/datasets/samet713/9_yargitay_kararlari_2025) · [Caselaw Retrieval](https://huggingface.co/datasets/newmindai/caselaw-retrieval) · [Turkish Law 700k](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered)

Sources were merged, deduplicated, and split into task-specific files: `corpus.jsonl`, `embedding.jsonl`, `reranker.jsonl`, `llm.jsonl`, and `gold_benchmark.json` (240 verified QA pairs). See [data/README.md](data/README.md) for full details.

## Evaluation Metrics

**Retrieval:** Recall@5, Recall@10, MRR, nDCG  
**QA:** Exact Match, F1, BLEU, ROUGE, Faithfulness, Citation Accuracy

## Models (Kaggle Dataset)

Fine-tuned models are stored at `ardayildiz29/legal-models-tr-finetuned`:

| Folder | Description |
|--------|-------------|
| `bgem3/` | Domain-adapted BGE-M3 (MNRL, 2 epochs) |
| `reranker_ft/` | Fine-tuned Cross-Encoder |
| `qwen_qlora_ft/` | QLoRA Qwen2.5-7B adapter weights |

## Key Hyperparameters

### Embedding FT (P1)
- Base: `BAAI/bge-m3`
- Loss: `MultipleNegativesRankingLoss`
- Batch: 4, Epochs: 2, Max seq len: 512

### Reranker FT (P2)
- Base: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- `pos_weight=1.69` (class imbalance compensation)
- Min query len: 3 words

### LLM QLoRA (P2)
- Base: `Qwen/Qwen2.5-7B-Instruct`
- Max seq len: 1024, Max grad norm: 1.0
- Train samples: 5000

## Course
CENG493 — Term Project  
Platform: Kaggle (T4 x2 GPU)
