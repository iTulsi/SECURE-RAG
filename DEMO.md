# Milestone 2 lab demonstration (5-7 minutes)

For the latest demonstration, first follow the root [README](README.md) to
prepare the pinned pilot. Show [latest results](reports/Latest_Results.md),
then recompute `reports/evidence/decision_review_reconstructed_predictions.jsonl`
with the evaluator. Explain the reconstruction's provenance. The older E0/E1
commands below remain useful for showing the baseline but are superseded by
the latest result report.

Run from this project directory. Python 3.11+ is sufficient to demonstrate
the saved experiments; the runtime has no package dependencies.

## Show the implementation

Open `src/secure_rag/pilot.py` (paired source selection), `retrieval.py`
(context chunks and ranking), `runner.py` (local inference and token logging),
and `evaluation.py` (strict T/F/X scoring). Show `data/pilot/manifest.json`
for the seed, official revision, source URLs, and file hashes.

## Recompute the measured output

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v

PYTHONPATH=src python3 -m secure_rag evaluate \
  --examples data/pilot/examples.jsonl \
  --predictions outputs/e0_predictions.jsonl \
  --output outputs/e0_demo_metrics.json

PYTHONPATH=src python3 -m secure_rag evaluate \
  --examples data/pilot/e1_dense_examples.jsonl \
  --predictions outputs/e1_predictions.jsonl \
  --output outputs/e1_demo_metrics.json

PYTHONPATH=src python3 -m secure_rag compare-runs \
  --examples data/pilot/examples.jsonl \
  --baseline outputs/e0_predictions.jsonl \
  --candidate outputs/e1_predictions.jsonl \
  --retrieval outputs/e1_retrieval.jsonl \
  --output outputs/e0_e1_demo_comparison.json
```

Explain KCV accuracy (45.83% E0, 40.28% E1), VOOD accuracy (61.11% E0,
100.00% E1), and seven KCV regressions. VOOD's E1 prompt announces missing
evidence, which limits the interpretation. Stored response usage totals are
110,718 E0 and 121,759 E1 generator tokens.

## Optional live two-question inference on the original Mac

Start Ollama, with `qwen2.5:3b-instruct` installed:

```bash
PYTHONPATH=src python3 -m secure_rag run-baseline \
  --examples data/pilot/e1_dense_examples.jsonl \
  --output outputs/live_demo_predictions.jsonl \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:3b-instruct \
  --seed 20260920 --limit 2
```

This cache has only two rows. Do not report its score as a 144-row result.
`RUN_ON_MAC.md` describes the full E2 run.

## Show the complete code path

`src/secure_rag/retrieval.py` implements E2 BM25 plus dense rank fusion and
preserves a ten-passage pool. `src/secure_rag/stages.py` implements E3
model-based passage reranking, E4 verification prompt preparation, and the
abstention gate. The Mac commands in `RUN_ON_MAC.md` produce every E2–E4
prediction, metric, and paired comparison file. `reports/Final_Results_Status.md`
marks which scores are actually measured. Run all stages locally before
presenting any E2–E4 score in the lab.
