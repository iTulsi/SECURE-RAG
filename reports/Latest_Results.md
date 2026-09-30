# SECURE RAG: measured pilot improvement

Published label-level evidence and recomputation are documented in
[evidence/README.md](evidence/README.md). The complete terminal reconstruction
reproduces the reported counts with the existing evaluator; it is not the
original model-response cache. Pilot/holdout source manifests and original
corpus snapshot hashes are in [data/manifests](../data/manifests).

The decision-review pipeline completed on the user's Mac with **87.50% KCV accuracy and 93.75% overall accuracy**. This is the latest measured development-pilot result, replacing the earlier 51.39% / 75.00% result as the best observed run.

## Experiment comparison

| Experiment | KCV accuracy | VOOD accuracy | Overall accuracy |
|---|---:|---:|---:|
| Original cached baseline | 33/72 = 45.83% | 44/72 = 61.11% | 77/144 = 53.47% |
| Multi-source retrieval | 22/72 = 30.56% | 72/72 = 100.00% | 94/144 = 65.28% |
| Evidence-first with repaired parser | 37/72 = 51.39% | 71/72 = 98.61% | 108/144 = 75.00% |
| Structured decision review with evidence rules | **63/72 = 87.50%** | **72/72 = 100.00%** | **135/144 = 93.75%** |

Compared with the repaired evidence-first run, this adds **26 correct KCV answers** and **27 correct answers overall**. The gains are **36.11 percentage points in KCV** and **18.75 percentage points overall**. Nine KCV answers remain incorrect.

VOOD 72/72 is derived from the reported overall count minus the KCV count: 135 − 63 = 72. Missing supplied context is routed to X by the pipeline; this score does not measure the model's independent ability to identify missing evidence.

## What changed

The improved pipeline reuses the original compact CVE evidence and saved first-pass predictions:

1. Missing supplied context produces X without a model request.
2. Narrowly supported CVSS claims use direct field comparisons, including user interaction in the CVSS vector. HIGH and CRITICAL are treated as distinct values.
3. Other existing valid T/F decisions are retained.
4. Prior X or invalid answers receive a fresh model review with explicit instructions to distinguish contradiction from genuine missing information.
5. The model returns structured evidence, comparison, and T/F/X fields. The runner validates responses and checkpoints each successful answer.
6. Complete merged predictions are scored using the existing evaluator. The run writes a paired comparison, input/output hashes, and a report.

The completed pilot logged eight CVSS checks, 72 missing-context decisions, 34 retained binary answers, and **30 fresh model reviews**. The new model calls used the existing Qwen 2.5 3B Instruct endpoint, temperature 0, seed 20260920, and a 512-token output budget. No model training was performed.

The latest result is a combined pipeline result, not a fresh independent 144-question model run. Task names and gold labels are not used to prepare decisions or inference prompts; they are used for scoring. Selection of this method was informed by development-pilot errors, so independent validation is still needed.

## Dataset integration and attribution

CVE, NVD, CISA KEV, CWE, and MITRE ATT&CK evidence ingestion remains in the project. The external retrieval experiment achieved 65.28% overall and is reported separately. The improved 93.75% result uses original benchmark CVE context and must not be attributed to the external datasets, additional training data, or fine-tuning.

## Reproduction and artifacts

Source implementation commit: `caa0a1d` — `Review uncertain CVE claims with structured evidence decisions`.

The original run artifacts are on the author's Mac under the project directory.

The completed run's original artifacts are:

- `outputs/kcv_review/RESULTS.md`
- `outputs/kcv_review/metrics.json`
- `outputs/kcv_review/predictions.jsonl`
- `outputs/kcv_review/comparison.json`
- `outputs/kcv_review/run_manifest.json`
- `outputs/kcv_review/review_examples.jsonl`
- `outputs/kcv_review/review_predictions.jsonl`
- `outputs/kcv_review/fixed_predictions.jsonl`

Run again to resume the same cached experiment:

```bash
cd "/Users/mac/Desktop/SECURE_RAG_Submission 2"
bash RUN_KCV_IMPROVEMENT.command
```

Validation before delivery included 48 passing tests, Ruff lint, launcher syntax and diff checks, a clean patch-application check, and a local HTTP integration check with synthetic responses. Synthetic endpoint responses were used only to verify software behavior, not to claim model accuracy.

The measured accuracy here is based on the user's completed-run terminal output. The final prediction and metrics files remain on the Mac and have not been uploaded for independent hash verification. Use those original artifacts for the submission's full metrics, rather than inventing unreported F1 values or confidence measures.

## Independent holdout check

The packaged holdout contains 146 examples from 20 CVE sources that do not overlap the pilot: 73 KCV and 73 VOOD. Keep the method fixed and run:

```bash
cd "/Users/mac/Desktop/SECURE_RAG_Submission 2"
export PYTHONPATH="$PWD/src"
python3.11 -m secure_rag prepare-claim-evidence --examples data/holdout/examples.jsonl --output data/holdout/claim_evidence_examples.jsonl
python3.11 -m secure_rag run-reasoned --examples data/holdout/claim_evidence_examples.jsonl --output outputs/kcv_review_holdout/first_pass_predictions.jsonl --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct --seed 20260920 --progress
python3.11 -m secure_rag run-kcv-review --examples data/holdout/examples.jsonl --predictions outputs/kcv_review_holdout/first_pass_predictions.jsonl --output-dir outputs/kcv_review_holdout
```

This reproduces both stages: a fresh evidence-first run, followed by the same selective structured review, retention of binary decisions, and evidence rules. Both inference stages resume saved responses. The holdout's accuracy is currently unmeasured. Do not tune against holdout labels or present the pilot score as a full-benchmark or held-out score. Omitting `--predictions` is supported for a fresh single-stage decision review, but that is a different evaluation from this two-stage pilot pipeline.

## Concise explanation for the teacher

We found that the small language model often confused a contradicted claim with insufficient information, and some explanations failed our verdict parser. We repaired explicit-verdict parsing, then added a structured second review for uncertain responses and direct checks for unambiguous CVSS facts. On the fixed 144-example development pilot, KCV accuracy increased from 45.83% in the original baseline to 87.50%, and overall accuracy increased from 53.47% to 93.75%. These are measured combined-pipeline results; independent holdout performance remains to be established.
