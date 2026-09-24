# SECURE-RAG reproducible baseline

This repository implements a pilot pipeline for **SECURE-RAG: Evidence-
Grounded Cybersecurity Reasoning with Hybrid Retrieval, Reranking, and
Uncertainty-Aware Abstention**.

The current scope is deliberately narrow:

- validate the official KCV and VOOD TSV files without editing ground truth;
- build a deterministic pilot grouped by CVE source URL;
- run the official prompts unchanged against an OpenAI-compatible endpoint;
- cache every raw response, model identifier, latency, and token-usage record;
- parse labels strictly and report accuracy, macro-F1, coverage, and abstention
  precision/recall/F1.

The project includes E0 baseline inference, E1 dense retrieval, E2 hybrid
retrieval, E3 model reranking, and E4 evidence verification with an abstention
gate. E2–E4 require local inference to produce measured results. See
`reports/Submission_Report.md` for the measured pilot and its limitations.

## Why KCV and VOOD are paired

The official repository contains 466 KCV rows and 466 VOOD rows. Corresponding
rows use the same CVE URL and question. KCV supplies the CVE JSON as evidence;
VOOD omits that evidence and expects `X` (abstain). Pilot selection therefore
groups by source URL and always keeps both tasks together.

## Setup

Python 3.11 or later is required. Runtime code uses only the standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
git clone --depth 1 https://github.com/aiforsec/SECURE.git ../SECURE
```

Record the exact upstream revision:

```bash
git -C ../SECURE rev-parse HEAD
```

## 1. Validate the official data

```bash
secure-rag validate --data-dir ../SECURE/Dataset
```

Expected official summary at revision
`a2412e8ab4b6051ba7381d4f590cb48d9743c7c8`:

| Task | Rows | Sources | Labels |
|---|---:|---:|---|
| KCV | 466 | 124 | F=282, T=183, X=1 |
| VOOD | 466 | 124 | X=466 |

The single official KCV `X` label is preserved exactly.

## 2. Create the fixed paired pilot

```bash
secure-rag prepare-pilot \
  --data-dir ../SECURE/Dataset \
  --output-dir data/pilot \
  --sources 20 \
  --seed 20260920 \
  --upstream-commit a2412e8ab4b6051ba7381d4f590cb48d9743c7c8
```

The command writes `data/pilot/examples.jsonl` and `data/pilot/manifest.json`.
The manifest records the selected URLs, seed, upstream commit, label counts, and
SHA-256 hashes of the two official files.

## 3. Run a model baseline

The runner accepts an OpenAI-compatible `/chat/completions` endpoint. Use the
exact model identifier exposed by your endpoint. If authentication is required,
set the key only in the environment:

```bash
export SECURE_RAG_API_KEY='your-key'
secure-rag run-baseline \
  --examples data/pilot/examples.jsonl \
  --output outputs/e0_predictions.jsonl \
  --base-url http://localhost:1234/v1 \
  --model your-model-id \
  --seed 20260920
```

Use `--limit 2` for a smoke test. The output cache is rewritten atomically after
each successful response. It resumes only if the prompts, model, and settings
match. The legacy E0/E1 outputs remain evaluable; use a fresh output path for
new inference.

Do not place API keys in notebooks, command arguments, output files, or Git.

## 4. Evaluate predictions

```bash
secure-rag evaluate \
  --examples data/pilot/examples.jsonl \
  --predictions outputs/e0_predictions.jsonl \
  --output outputs/e0_metrics.json
```

Parsing is intentionally strict: after whitespace removal, the complete output
must be exactly `T`, `F`, or `X`. Explanations such as `False because ...` are
counted as invalid rather than silently repaired.

For a no-model pipeline sanity check, generate an all-`X` prediction file:

```bash
secure-rag make-template \
  --examples data/pilot/examples.jsonl \
  --output outputs/x_only_predictions.jsonl
```

This is a sanity baseline, not an LLM result. It should score perfectly on VOOD
and poorly on supported KCV questions, demonstrating why overall accuracy alone
is misleading.

## 5. Prepare E1 dense-only prompts

E1 retrieves only from evidence supplied inside each benchmark example. It
does not fetch withheld CVE records for VOOD, because doing so would invalidate
the official `X` labels. The embedding endpoint must expose the OpenAI-compatible
`/embeddings` route.

```bash
secure-rag prepare-dense \
  --examples data/pilot/examples.jsonl \
  --output data/pilot/e1_dense_examples.jsonl \
  --retrieval-output outputs/e1_retrieval.jsonl \
  --embedding-cache outputs/e1_embedding_cache.jsonl \
  --base-url http://localhost:11434/v1 \
  --model nomic-embed-text \
  --top-k 5
```

This writes path-aware evidence chunks, cosine scores, and an embedding cache.
Then run the same generator and evaluator used for E0:

```bash
secure-rag run-baseline \
  --examples data/pilot/e1_dense_examples.jsonl \
  --output outputs/e1_predictions.jsonl \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:3b-instruct \
  --seed 20260920

secure-rag evaluate \
  --examples data/pilot/e1_dense_examples.jsonl \
  --predictions outputs/e1_predictions.jsonl \
  --output outputs/e1_metrics.json
```

Use exactly the same generator model for E0 and E1. The embedding model is a
separate retrieval component and must be recorded with the results.

## 6. Compare E0 and E1

Generate paired correctness transitions and retrieval-score summaries before
adding another component:

```bash
secure-rag compare-runs \
  --examples data/pilot/examples.jsonl \
  --baseline outputs/e0_predictions.jsonl \
  --candidate outputs/e1_predictions.jsonl \
  --retrieval outputs/e1_retrieval.jsonl \
  --output outputs/e0_e1_comparison.json
```

## 7. Prepare E2 BM25 + dense prompts

E2 adds standard-library Okapi BM25 and combines its ranking with the E1 dense
ranking using reciprocal rank fusion. It reuses the E1 embedding cache and does
not add a package dependency.

```bash
secure-rag prepare-hybrid \
  --examples data/pilot/examples.jsonl \
  --output data/pilot/e2_hybrid_examples.jsonl \
  --retrieval-output outputs/e2_retrieval.jsonl \
  --embedding-cache outputs/e1_embedding_cache.jsonl \
  --base-url http://localhost:11434/v1 \
  --model embeddinggemma:latest \
  --top-k 5 \
  --rrf-k 60
```

The retrieval log records dense, BM25, and fused ranks and scores, plus a
ten-passage candidate pool. The E2 generator receives only the top five. Run
the generator and evaluator exactly as in E1, changing
only the example and output filenames. Detailed Mac commands are in
`RUN_ON_MAC.md`. Treat VOOD results cautiously: an empty evidence section
explicitly reveals missing context to the generator, so VOOD gains are not
attributable to semantic ranking alone.

## 8. Run E3 reranking and E4 verification

E3 asks a local model once per KCV example to select the five most relevant
hybrid passages, then sends those to the generator. E4 separately asks the local model
whether the evidence supports, contradicts, or cannot establish the claim. A
T/F answer is retained only when generator and verifier agree; all other
outputs become X. E4 is an additional model check, not independent evidence
or calibrated confidence. `RUN_ON_MAC.md` has the complete command sequence.
Ground-truth labels are never included in the model prompts.
If E3 returns duplicate or invalid passage numbers, the ranking log marks
`ranking_valid: false` and fills the missing slots in E2 order. Report how many
rankings needed this repair before interpreting an E3 gain.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## Research integrity

- Keep the official TSV files unchanged.
- Use the same pilot manifest for every E0-E4 comparison.
- Never compare pilot scores directly with full-dataset paper scores.
- Save raw outputs and failed/invalid parses.
- Report the exact model ID, endpoint implementation, seed support, and hardware.
- Treat generated cybersecurity advice as requiring human review.

## Source

- Official benchmark: https://github.com/aiforsec/SECURE
- Paper: https://arxiv.org/abs/2405.20441
