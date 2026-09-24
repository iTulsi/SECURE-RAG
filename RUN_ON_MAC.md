# Run the complete E2–E4 pipeline on your Mac

Run from the project directory. The E0 and E1 pilot artifacts are included in
`data/pilot/` and `outputs/`. The original E1 embedding cache was not in the
results archive, so E2 will request and cache the same embeddings again. No
official ground-truth TSV is modified.

1. Confirm `ollama list` shows `qwen2.5:3b-instruct` and `embeddinggemma:latest`.
   If the server is not running, use `ollama serve` in another terminal.
2. Install this project in a Python 3.11+ virtual environment using
   `python -m pip install -e .`.
3. Run the following commands:

```bash
secure-rag prepare-hybrid \
  --examples data/pilot/examples.jsonl \
  --output data/pilot/e2_hybrid_examples.jsonl \
  --retrieval-output outputs/e2_retrieval.jsonl \
  --embedding-cache outputs/e1_embedding_cache.jsonl \
  --base-url http://localhost:11434/v1 \
  --model embeddinggemma:latest \
  --top-k 5 --rrf-k 60

secure-rag run-baseline \
  --examples data/pilot/e2_hybrid_examples.jsonl \
  --output outputs/e2_predictions.jsonl \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:3b-instruct \
  --seed 20260920

secure-rag evaluate \
  --examples data/pilot/e2_hybrid_examples.jsonl \
  --predictions outputs/e2_predictions.jsonl \
  --output outputs/e2_metrics.json

secure-rag compare-runs \
  --examples data/pilot/examples.jsonl \
  --baseline outputs/e1_predictions.jsonl \
  --candidate outputs/e2_predictions.jsonl \
  --retrieval outputs/e2_retrieval.jsonl \
  --output outputs/e1_e2_comparison.json
```

Check that the model identifiers match the saved E1 outputs before interpreting
the E1/E2 comparison. The prompt format and top-five evidence budget match;
only the ranking changes. The first command may take time to rebuild the
missing embedding cache. Subsequent calls reuse it.

E2 retrieval also stores ten candidate passages in `candidate_pool`; the E2
generator sees only the selected five. After the commands above, run E3:

```bash
secure-rag prepare-reranked \
  --examples data/pilot/examples.jsonl --retrieval outputs/e2_retrieval.jsonl \
  --output data/pilot/e3_reranked_examples.jsonl \
  --reranking-output outputs/e3_reranking.jsonl \
  --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct \
  --seed 20260920 --top-k 5

secure-rag run-baseline \
  --examples data/pilot/e3_reranked_examples.jsonl \
  --output outputs/e3_predictions.jsonl \
  --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct \
  --seed 20260920

secure-rag evaluate --examples data/pilot/e3_reranked_examples.jsonl \
  --predictions outputs/e3_predictions.jsonl --output outputs/e3_metrics.json

secure-rag compare-runs --examples data/pilot/examples.jsonl \
  --baseline outputs/e2_predictions.jsonl --candidate outputs/e3_predictions.jsonl \
  --output outputs/e2_e3_comparison.json
```

The reranker makes one ranking call for each KCV example and caches the
selected passage indices for resuming. E4 makes a separate local model call for each example
to check whether the selected evidence directly supports or contradicts the
claim. It keeps an answer only if the generator and verifier agree on T or F:

```bash
secure-rag prepare-verification \
  --examples data/pilot/e3_reranked_examples.jsonl \
  --output data/pilot/e4_verification_examples.jsonl

secure-rag run-baseline \
  --examples data/pilot/e4_verification_examples.jsonl \
  --output outputs/e4_verifier_predictions.jsonl \
  --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct \
  --seed 20260920

secure-rag apply-verification --examples data/pilot/examples.jsonl \
  --generator outputs/e3_predictions.jsonl \
  --verifier outputs/e4_verifier_predictions.jsonl \
  --output outputs/e4_predictions.jsonl

secure-rag evaluate --examples data/pilot/examples.jsonl \
  --predictions outputs/e4_predictions.jsonl --output outputs/e4_metrics.json

secure-rag compare-runs --examples data/pilot/examples.jsonl \
  --baseline outputs/e3_predictions.jsonl --candidate outputs/e4_predictions.jsonl \
  --output outputs/e3_e4_comparison.json

python -m unittest discover -s tests -v
```

Every metrics file must report 72 KCV and 72 VOOD examples. Report KCV
accuracy and coverage, VOOD abstention, and token use; E4 can increase
abstention while lowering KCV coverage. The same Qwen model serves the
reranker, generator, and verifier, so agreement is not an independent safety
guarantee. The pilot has already been inspected: improvements on it require
a separate source-grouped test set before general claims. Do not claim E2–E4
scores until these commands have finished.
Count `ranking_valid: false` records in `outputs/e3_reranking.jsonl` and
report them alongside E3 metrics; those records use E2 order to fill any
missing passage slots.
