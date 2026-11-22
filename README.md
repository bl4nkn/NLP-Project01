# NLP Project I: Structured Text Classification & Information Extraction

| Field            | Information                             |
|:---------------- |:--------------------------------------- |
| **Student Name** | `Muhammet Fatih Bektaş`                 |
| **Student ID**   | `2021556012`                            |
| **Course**       | Natural Language Processing — Project I |

---

## 1. Objective

The goal of this project is to move beyond free-form conversational LLM outputs by forcing a local open-source language model to produce strictly structured JSON that software systems can parse reliably. Using movie reviews, the model jointly performs binary sentiment classification (`positive` vs. `negative`), predicts a confidence-style sentiment score in `[0.0, 1.0]`, and extracts named person entities.

---

## 2. Dataset, Gold Annotations, and Matching Rules

We used the Stanford IMDB dataset (`stanfordnlp/imdb`, 25,000 test reviews stored locally in `data/stanfordnlp_imdb/imdb_test.parquet`). Because IMDB provides only sentiment labels and no entity labels, we sampled a balanced evaluation subset of **100 short, clean reviews** (`50` positive, `50` negative; `120–380` characters, HTML tags removed) saved in `data/dataset_100.csv`. All 100 reviews were manually annotated with gold person entities, yielding **127 total entity mentions** across 69 reviews and 31 zero-entity reviews (`[]`) to test false-positive hallucination.

- **Annotation Rules:**
  1. **Include** named people explicitly mentioned in the review body: actors, directors, writers, public figures, and named fictional characters (e.g., *Shawn Phelan*, *Aunty Lou*).
  2. **Exact surface form & deduplication:** Copy the substring as written in the review, keeping the fullest name form when a person is mentioned multiple times (e.g., *Bruce Willis* rather than both *Bruce Willis* and *Willis*).
  3. **Exclude** movie/TV titles (e.g., *The Buddy Holly Story*), studios/networks (*BBC*, *Disney channel*), bands (*Pink Floyd*), locations, nationalities, and reviewer sign-offs.
- **Entity Matching Rule:** Evaluation uses **case-insensitive normalized exact matching**: both gold and predicted entity strings are stripped of leading/trailing whitespace and punctuation, lowercased, and compared as unique sets per review to compute micro Precision, Recall, and F1. *(Note: Because the IMDB dataset is in English, standard Python `str.lower()` is safe here; for Turkish text, `'İ'` and `'I'` would first need replacement with `'i'` and `'ı'` before calling `lower()` to avoid combining-dot mismatches.)*

---

## 3. Experimental Setup & Reproducibility

All experiments ran completely offline using Ollama without any cloud APIs.

| Component                 | Exact Value / Setting                                                                             |
|:------------------------- |:------------------------------------------------------------------------------------------------- |
| **Model & Tag**           | `qwen2.5:3b` (Architecture: `qwen2`, Parameters: `3.1B`, Size: `1.9 GB`)                          |
| **Quantization Level**    | `Q4_K_M` (verified via `ollama show qwen2.5:3b`)                                                  |
| **Runtime & Client**      | Ollama `v0.40.17` (local server) \| Python `ollama` package `v0.6.3`                              |
| **Hardware**              | Apple M2 (16 GB unified RAM), macOS                                                               |
| **Generation Parameters** | `temperature = 0.0`, `seed = 42`, `num_ctx = 2048`                                                |
| **Required JSON Schema**  | `{"sentiment": "positive" \| "negative", "sentiment_score": number [0, 1], "entities": string[]}` |

---

## 4. Methodology: Two Experimental Settings

Both experimental settings evaluate the exact same 100 reviews using the identical system prompt and user prompt. The only variable changed is how the JSON format is enforced:

- **Setting 1 — Prompt-Only:** JSON structure is requested purely via natural-language instructions. Ollama's `format` parameter is **not** passed.
- **Setting 2 — Schema-Constrained:** The same prompt is paired with Ollama's `format=RESPONSE_SCHEMA` parameter, which applies grammar-based token masking during decoding so generated tokens conform to the JSON Schema.

**Cleaning & Exception Handling:** First, `json.loads()` is applied directly to the raw model string (**Raw JSON Parse Rate**). Next, `clean_json_text()` strips whitespace, removes Markdown fences (` ```json ... ``` `), and slices from the first `{` to the last `}`, followed by a `try-except json.JSONDecodeError` parser (**Cleaned JSON Parse Rate**). Finally, `is_valid_schema()` checks all required keys and value types (**Schema Validity Rate**). Samples failing validation are labeled `"invalid"` and penalized in accuracy and macro F1.

---

## 5. Evaluation Results (100 Samples)

| Metric                                                  | Prompt-Only     | Schema-Constrained |
|:------------------------------------------------------- |:---------------:|:------------------:|
| **Raw JSON Parse Rate** (before cleaning)               | 99.00% (99/100) | 100.00% (100/100)  |
| **Cleaned JSON Parse Rate** (after cleaning)            | 99.00% (99/100) | 100.00% (100/100)  |
| **Schema Validity Rate**                                | 99.00% (99/100) | 100.00% (100/100)  |
| **Sentiment Accuracy**                                  | 93.00%          | 94.00%             |
| **Sentiment Macro F1**                                  | 93.45%          | 94.00%             |
| **Entity Precision** (TP/FP/FN: 110/53/17 vs 110/54/17) | 67.48%          | 67.07%             |
| **Entity Recall**                                       | 86.61%          | 86.61%             |
| **Entity Micro F1**                                     | 75.86%          | 75.60%             |
| **Average Inference Latency**                           | 1.250 s/sample  | 1.662 s/sample     |

**Key Observations:** Modern instruction-tuned models like `qwen2.5:3b` follow JSON prompts well (99/100 valid in Prompt-Only), yet still exhibit rare structural slips. Specifically, on **Sample ID 58**, Prompt-Only emitted a duplicate opening brace (`'{\n\n{\n  "sentiment": ...'`), failing both raw `json.loads()` and fence/brace cleaning (handled safely via `try-except`). Schema-Constrained mode achieved 100% raw JSON and schema validity, at a modest runtime overhead (+0.41 s/sample on average due to grammar constraint checking at each decoding step). Across entity extraction, recall was high (86.61%) while precision was moderate (~67%), mainly because the 3B model occasionally extracted generic phrases (e.g., *"American soldier"*, *"The little girl"*) or movie titles (e.g., *"Nattens engel"*).

---

## 6. Discussion: Does Schema-Constrained Decoding Change Task Accuracy?

**Short Answer:** Primarily through *format failure recovery* rather than improved semantic understanding. When comparing the 99 reviews where Prompt-Only produced valid JSON, Prompt-Only and Schema-Constrained generated **identical** sentiment labels and entity lists on every single one of those 99 samples.

**Why:** Schema-constrained decoding operates at the token-sampling layer by masking out tokens that would violate the JSON grammar (for instance, blocking a second `{` right after the opening `{`). It does not add world knowledge or alter the Transformer's internal semantic representations. Consequently, on Sample ID 58, Schema-Constrained prevented the duplicate-brace syntax crash and recovered the correct `positive` sentiment, raising overall Sentiment Accuracy from **93.00% to 94.00%** (and Macro F1 from **93.45% to 94.00%**). Interestingly, however, on that same recovered sample, the model truncated *"Jonathan Silverman"* to *"Jonathan Silver"*, adding one false positive and nudging Entity F1 slightly from **75.86% to 75.60%**.

**Conclusion:** For production pipelines, schema-constrained decoding is highly valuable because it eliminates syntax crashes (100% schema validity) and removes downstream parsing failures that otherwise drag down end-to-end accuracy—though improving actual semantic accuracy (such as handling sarcasm or avoiding non-person nouns) still depends on prompt clarity, few-shot examples, or model scale.

---

## 7. Repository Structure & How to Run

```text
.
├── README.md                     # Project report & documentation
├── nlp_project1.ipynb            # Complete notebook (experiment, live tests, diagnostics)
├── data/
│   ├── dataset_100.csv           # 100 annotated evaluation samples
│   └── stanfordnlp_imdb/         # Local Stanford IMDB train/test parquet copy
└── results/
    ├── metrics_summary.csv       # Summary comparison table
    ├── predictions_prompt_only.json
    └── predictions_schema.json
```

To reproduce locally:

```bash
ollama serve
ollama pull qwen2.5:3b
jupyter notebook nlp_project1.ipynb
```
