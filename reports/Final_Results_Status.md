# SECURE-RAG experiment status and final result sheet

**Historical checkpoint.** The current best measured development-pilot
result is 87.50% KCV and 93.75% overall. The multi-source run achieved
30.56% KCV and 65.28% overall. See [Latest_Results.md](Latest_Results.md)
and [published evidence](evidence/README.md). Historical pending cells below
refer to the older archive, not the current implementation status.

The packaged pilot has 72 KCV and 72 VOOD examples from 20 source URLs.
E0 and E1 below are **measured from saved predictions**. E2–E4 contain
complete executable code but **no measured scores in this package**: the
local Ollama models and embedding cache are on the original Mac. Run every
command in `RUN_ON_MAC.md` before replacing the pending cells.

| Measure | E0 original | E1 dense | E2 hybrid | E3 reranked | E4 verified |
|---|---:|---:|---:|---:|---:|
| KCV accuracy | 45.83% | 40.28% | pending | pending | pending |
| KCV coverage | 55.56% | 48.61% | pending | pending | pending |
| VOOD accuracy | 61.11% | 100.00% | pending | pending | pending |
| Overall accuracy | 53.47% | 70.14% | pending | pending | pending |
| Abstention F1 | 59.46% | 79.56% | pending | pending | pending |

Read exact values from `outputs/eN_metrics.json`: task accuracy and coverage
are in `kcv` and `vood`; overall accuracy is in `overall`; abstention F1 is in
`abstention`. For E2–E4 use these files:

- `outputs/e2_predictions.jsonl`, `outputs/e2_metrics.json`,
  `outputs/e2_retrieval.jsonl`, `outputs/e1_e2_comparison.json`;
- `outputs/e3_reranking.jsonl`, `outputs/e3_predictions.jsonl`,
  `outputs/e3_metrics.json`, `outputs/e2_e3_comparison.json`;
- `outputs/e4_verifier_predictions.jsonl`, `outputs/e4_predictions.jsonl`,
  `outputs/e4_metrics.json`, `outputs/e3_e4_comparison.json`.

The PDF in this archive records the **earlier E0/E1 checkpoint**. Its section
on future E3/E4 work predates the additional code in this archive. Do not
submit it as a final E0–E4 result report without updating the methods and
table using the measured E2–E4 files.

**Interpretation limits:** The same Qwen model acts as generator, reranker,
and verifier, so E4 agreement is not independent validation. E1's explicit
empty-evidence cue can explain its VOOD gain. The pilot was inspected while
designing later stages; use another source-grouped set to estimate general
performance. Report KCV accuracy and coverage alongside VOOD abstention.
