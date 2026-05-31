# Data

All datasets are hosted on Kaggle. **No data files are committed to this repo.**

## Kaggle Dataset: `ardayildiz29/legalo`

The datasets used in this project were **custom-built** by combining and preprocessing multiple public Turkish legal sources (see [Source Datasets](#source-datasets) below). Raw sources were merged, deduplicated, and reformatted into task-specific training splits.

| File | Description | Used by |
|------|-------------|---------|
| `corpus.jsonl` | Turkish legal document corpus | S1–S8 |
| `gold_benchmark.json` | 240 QA pairs with verified answers | S1–S8 (evaluation) |
| `embedding.jsonl` | (query, positive) pairs for embedding FT | P1 |
| `reranker.jsonl` | (query, passage, label) pairs for reranker FT | P2 |
| `llm.jsonl` | Instruction-tuning samples for QLoRA SFT | P2 |

## Source Datasets

The custom dataset was constructed by combining and preprocessing the following public sources:

| Dataset | Source | Content |
|---------|--------|---------|
| [Turkish Law QA Dataset](https://huggingface.co/datasets/OrionCAF/turkish_law_qa_dataset) | HuggingFace | Turkish legal question-answer pairs |
| [Turkish Law Dataset for LLM Fine-Tuning](https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning) | Kaggle | LLM instruction-tuning corpus |
| [Turkish LawChatbot](https://huggingface.co/datasets/Renicames/turkish-lawchatbot) | HuggingFace | Legal chatbot QA pairs |
| [Turkish Constitutional Court](https://huggingface.co/datasets/KocLab-Bilkent/turkish-constitutional-court) | HuggingFace (KocLab-Bilkent) | Constitutional court decisions |
| [Yargıtay Retrieval Dataset](https://github.com/koc-lab/yargitay_retrieval_dataset) | GitHub (KocLab) | Supreme court retrieval benchmark |
| [9 Yargıtay Kararları 2025](https://huggingface.co/datasets/samet713/9_yargitay_kararlari_2025) | HuggingFace | 2025 Court of Cassation rulings |
| [Caselaw Retrieval](https://huggingface.co/datasets/newmindai/caselaw-retrieval) | HuggingFace | Caselaw retrieval pairs |
| [Turkish Law Documents 700k Clustered](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) | HuggingFace | Large-scale clustered legal documents |

## Dataset Construction

The custom splits were built from the sources above through the following steps:

1. **Merge & deduplication** — all source documents concatenated and near-duplicate passages removed
2. **Corpus construction** — documents chunked into passages for `corpus.jsonl`
3. **Embedding pairs** — (query, positive_passage) pairs mined from QA sources for contrastive training
4. **Reranker pairs** — positive passages paired with hard negatives to form binary labels for `reranker.jsonl`
5. **LLM SFT pairs** — QA sources reformatted into instruction-tuning format for `llm.jsonl`
6. **Gold benchmark** — 240 QA pairs manually verified for evaluation in `gold_benchmark.json`

## Data Schema

### corpus.jsonl
```json
{"id": "doc_001", "text": "Türk Medeni Kanunu madde 1...", "source": "TMK", "article": "1"}
```

### gold_benchmark.json
```json
[{"question": "Kira sözleşmesi kaç yıl geçerlidir?", "answer": "...", "doc_ids": ["doc_042"]}]
```

## Setup on Kaggle

In each notebook, attach the following datasets under **Input**:
- `ardayildiz29/legalo` → mounts at `/kaggle/input/datasets/ardayildiz29/legalo/`
- `ardayildiz29/legal-models-tr-finetuned` → mounts at `/kaggle/input/datasets/ardayildiz29/legal-models-tr-finetuned/`
