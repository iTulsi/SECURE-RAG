# SECURE-RAG: code and verified submission results

**Current submission entry point:** [README.md](README.md) and
[reports/Latest_Results.md](reports/Latest_Results.md). The best measured
development-pilot result is 87.50% KCV and 93.75% overall. The older verified
baseline and multi-source instructions below remain as experiment history.
Published terminal reconstructions are described in
[reports/evidence/README.md](reports/evidence/README.md).

Prepared for Tulsi Tomar, Bennett University, on 1 October 2026.

The submission ZIP includes executable code, tests, the normalized evidence
corpus, benchmark prompts, retrieval audits, and saved predictions. Native
source snapshots are supplied separately as `SECURE_RAG_Source_Snapshots.zip`
to keep the main code upload below 10 MB. They are needed only to rebuild
the corpus: extract that companion ZIP beside the main ZIP so both populate
the same `SECURE_RAG_Submission/data/` folder. New inference uses the already
included normalized corpus and does not need the companion ZIP.

## Results available now

The following scores were recomputed from the recovered saved predictions.
They cover the same 144-example development pilot: 72 KCV and 72 VOOD
questions from 20 CVE sources. No mock response is included in these scores.

| Experiment | KCV accuracy | KCV coverage | VOOD accuracy | Overall accuracy |
|---|---:|---:|---:|---:|
| E0 original Qwen prompts | 45.83% (33/72) | 55.56% | 61.11% (44/72) | 53.47% (77/144) |
| E1 dense retrieval | 40.28% (29/72) | 48.61% | 100.00% (72/72) | 70.14% (101/144) |
| E5 context routing, exploratory | 45.83% (33/72) | 55.56% | 100.00% (72/72) | 72.92% (105/144) |
| E6 routing + CVSS rules, exploratory | 48.61% (35/72) | 55.56% | 100.00% (72/72) | 74.31% (107/144) |
| New multi-source Qwen predictions | Awaiting local model run | Awaiting run | Awaiting run | Awaiting run |

E5 combines the previously saved E0/E1 outputs using context availability.
E6 applies limited structured CVSS rules to E5. Both were designed after
examining this pilot, so their improvements are exploratory. E6 improves
overall accuracy by 20.83 percentage points over E0, but KCV improves by
only 2.78 points. Most of the gain comes from VOOD abstention. These results
do not support a 90% KCV claim or a claim of established generalization.

Exact scores and prediction hashes: `reports/Verified_Pilot_Results.json`.
Raw predictions: `outputs/e0_predictions.jsonl`, `e1_predictions.jsonl`,
`e5_predictions.jsonl`, and `e6_predictions.jsonl`.
Paired changes: `outputs/e0_e6_verified_comparison.json`.

## What the complete code implements

- SECURE KCV/VOOD dataset validation, paired source-grouped sampling, and strict evaluation.
- Original prompts, dense retrieval, BM25+dense fusion, model reranking, verifier gating,
  context routing, claim-evidence prompts, and limited CVSS rules.
- A normalized external evidence corpus from CVE List, NVD, CISA KEV, CWE, and MITRE ATT&CK.
- Entity filtering before ranking, source URLs, deterministic passage IDs, and hashes.
- A context-available evaluation protocol: withheld-context examples retain empty evidence.
- Cached inference that resumes and rejects predictions made with different prompts or settings.
- `run-review`: one command for BM25 retrieval, Qwen prediction, evaluation, comparison,
  provenance manifest, and a new Markdown results report.

The corpus has 6,812 passages. Saved retrieval audits cover the pilot (144
questions), a separate source-disjoint set (146), and the full benchmark
(932). Pilot and source-disjoint CVE source sets have zero overlap. Full-set
retrieval selected 2,327 passages; all 466 examples without supplied context
retained empty evidence. Passage counts measure retrieval execution, not
answer accuracy. The benchmark labels were not expanded or used to train Qwen.

## Run new results tonight on the Mac

Extract this ZIP into a new folder and open Terminal in that folder. Start
the Ollama application. The already installed `qwen2.5:3b-instruct` model
and Python 3.11 or newer are required. If the model is missing, run
`ollama pull qwen2.5:3b-instruct` first. If Ollama is not running, use
`ollama serve` in another terminal.

```bash
bash RUN_RESULTS_ON_MAC.command
```

This needs no pip install and no new runtime dependencies. It runs all 144
pilot prompts, displays progress, and saves each prediction. Repeat the
same command after interruption to resume. It reuses E0 as the comparison
baseline; it does not rerun E0. Runtime depends on the Mac and context sizes.

After completion, submit the updated code and these new files:

- `outputs/submission/RESULTS.md`
- `outputs/submission/predictions.jsonl`
- `outputs/submission/metrics.json`
- `outputs/submission/comparison.json`
- `outputs/submission/run_manifest.json`
- `outputs/submission/retrieval.jsonl`

No new metrics/report is created when inference stops before completion.
The recovered pilot predictions are preserved. To change the model, seed,
or input set, choose another `--output-dir` through `run-review`.

## Source-disjoint evaluation after the pilot

To check performance on the separate source set, first run its original
baseline and then compare the multi-source method on that same set:

```bash
export PYTHONPATH="$PWD/src"
python3.11 -m secure_rag run-baseline \
  --examples data/holdout/examples.jsonl \
  --output outputs/holdout_original_predictions.jsonl \
  --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct \
  --seed 20260920 --progress

python3.11 -m secure_rag run-review \
  --examples data/holdout/examples.jsonl \
  --baseline-predictions outputs/holdout_original_predictions.jsonl \
  --output-dir outputs/holdout_submission
```

Use the pilot first for the urgent submission. Do not present its accuracy
as the source-disjoint score. Source snapshots are newer than the original
benchmark and can conflict with its historical labels. More evidence may
improve or reduce accuracy; the new report records what the model actually does.

## Verified software checks

39 unit tests pass, including interruption/resume, no incomplete report,
invalid comparison IDs, input overwrite prevention, evidence routing,
source provenance, and prior stages. Ruff correctness lint and shell syntax
checks pass. The launcher was also exercised against an unavailable endpoint:
it reports incomplete inference and creates no accuracy result. Model response
mocks in tests validate software plumbing only.

```bash
PYTHONPATH=src python3.11 -m unittest discover -s tests -v
```

Older PDFs and reports in `reports/` record earlier checkpoints. Use this
file for the current submission status and the generated
`outputs/submission/RESULTS.md` after a completed new model run.
