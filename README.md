# SECURE-RAG: evidence-based cybersecurity claim verification

Research implementation by **Tulsi Tomar, B.Tech CSE (Cybersecurity), Bennett University**, based on [SECURE: Benchmarking Large Language Models for Cybersecurity](https://github.com/aiforsec/SECURE) (ACSAC 2024).

The project checks whether a small local model can distinguish a supported cybersecurity claim, a contradicted claim, and a claim that cannot be decided from CVE evidence. It implements retrieval experiments and a focused decision-review pipeline with **Qwen 2.5 3B Instruct through Ollama**. Runtime code uses Python's standard library.

## Latest measured results

The development pilot contains **144 examples: 72 KCV and 72 VOOD**, grouped by 20 CVE sources. These are pilot results, not full-benchmark scores.

| Experiment | KCV accuracy | VOOD accuracy | Overall accuracy |
|---|---:|---:|---:|
| Original benchmark prompts | 45.83% | 61.11% | 53.47% |
| Multi-source retrieval | 30.56% | 100.00% | 65.28% |
| Evidence-first, repaired parser | 51.39% | 98.61% | 75.00% |
| **Structured decision review + evidence rules** | **87.50% (63/72)** | **100.00% (72/72)** | **93.75% (135/144)** |

Decision review adds **26 correct KCV answers** over the repaired evidence-first run: **36.11 percentage points**. Nine KCV errors remain. This is a combined pipeline result, including retained model decisions and deterministic evidence checks. VOOD is routed to X when context is absent.

Start with [the latest report](reports/Latest_Results.md), [run evidence and provenance](reports/evidence/README.md), and [the code](src/secure_rag). The latest run completed on the author's Mac. Terminal-reconstructed predictions are identified as such; they are not the original response cache. Full raw responses from the latest run remain on the Mac. Holdout performance is pending.

## KCV and VOOD

- **KCV:** verify a claim using supplied CVE JSON. T = supported, F = contradicted, X = undetermined.
- **VOOD:** the corresponding claim has no supplied evidence. This protocol preserves missing evidence and expects abstention rather than outside knowledge.

The official benchmark has 466 KCV and 466 VOOD examples. The source-grouped pilot uses a paired subset. A separate holdout contains **146 examples from 20 different CVE sources**, with zero source overlap with the pilot.

## Implemented method

1. Pin and validate the benchmark without changing its labels.
2. Compact supplied CVE JSON while preserving descriptions, versions, solutions, and metrics.
3. Run the evidence-first model; save full responses, prompt hashes, settings, latency, and usage.
4. Route absent context to X and directly check narrowly supported CVSS facts.
5. Retain other valid T/F decisions and review X/invalid decisions with a focused comparison prompt.
6. Require structured evidence, comparison, and verdict fields; validate and checkpoint responses.
7. Score complete merged predictions and save paired changes and input/output hashes.

The completed improvement run used **eight CVSS checks, 72 missing-context decisions, 34 retained binary decisions, and 30 new model reviews**. The review distinguishes contradiction from missing information and checks negation, version boundaries, multi-part claims, HIGH versus CRITICAL severity, and CVSS user interaction. Gold labels and task names do not select decisions or enter inference prompts. Structured formatting does not itself guarantee factual correctness.

## Multi-dataset retrieval

| Source | Implemented role |
|---|---|
| [CVE List](https://github.com/CVEProject/cvelistV5) | Descriptions, affected products, versions, metrics, remediation |
| [NVD](https://nvd.nist.gov/developers/vulnerabilities) | CVE enrichment and weakness/metric information |
| [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) | Positive evidence of known exploitation |
| [CWE](https://cwe.mitre.org/data/downloads.html) | Definitions of linked weaknesses |
| [MITRE ATT&CK](https://github.com/mitre-attack/attack-stix-data) | Techniques when an explicit technique ID is present |

The prepared corpus has **6,812 passages** with source URLs, entity IDs, record hashes, and passage IDs. Retrieval routes by entity before BM25 or dense+BM25 reciprocal rank fusion; optional reranking retains provenance. See [MULTISOURCE_REVIEW.md](MULTISOURCE_REVIEW.md) and [validation](reports/Multisource_Validation.json). Source snapshot hashes are in [data/manifests/corpus.json](data/manifests/corpus.json). Third-party source snapshots are not relabeled or included under the project's MIT license.

External retrieval scored 65.28% overall. The 93.75% result uses original benchmark CVE context and decision review, not external-data training or fine-tuning. KEV and ATT&CK were imported but supplied no selected top-five passages in the recorded full-set retrieval audit.

## Setup and prepare the benchmark

Requirements: **Python 3.11+**, Git, and Ollama with `qwen2.5:3b-instruct`. Ollama must support [JSON-schema output](https://docs.ollama.com/capabilities/structured-outputs).

```bash
git clone https://github.com/iTulsi/SECURE-RAG.git
cd SECURE-RAG
export PYTHONPATH="$PWD/src"
git clone https://github.com/aiforsec/SECURE.git ../SECURE
git -C ../SECURE checkout a2412e8ab4b6051ba7381d4f590cb48d9743c7c8
python3.11 -m secure_rag validate --data-dir ../SECURE/Dataset
python3.11 -m secure_rag prepare-pilot --data-dir ../SECURE/Dataset --output-dir data/pilot --sources 20 --seed 20260920 --upstream-commit a2412e8ab4b6051ba7381d4f590cb48d9743c7c8
python3.11 -m secure_rag prepare-pilot --data-dir ../SECURE/Dataset --output-dir data/holdout --sources 20 --seed 20261001 --exclude-manifest data/pilot/manifest.json --upstream-commit a2412e8ab4b6051ba7381d4f590cb48d9743c7c8
```

The benchmark is fetched from its original repository. Reference split manifests are in [data/manifests](data/manifests). No pip install is needed with PYTHONPATH set. Model weights and third-party datasets retain their original licenses.

## Reproduce the two-stage pilot pipeline

Start the Ollama application, or run `ollama serve` in another terminal. Pull the model if needed:

```bash
ollama pull qwen2.5:3b-instruct
python3.11 -m secure_rag prepare-claim-evidence --examples data/pilot/examples.jsonl --output data/pilot/claim_evidence_examples.jsonl
python3.11 -m secure_rag run-reasoned --examples data/pilot/claim_evidence_examples.jsonl --output outputs/submission/evidence_first_predictions.jsonl --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct --seed 20260920 --progress
python3.11 -m secure_rag repair-reasoned --examples data/pilot/examples.jsonl --predictions outputs/submission/evidence_first_predictions.jsonl --output outputs/submission/evidence_first_repaired_predictions.jsonl
bash RUN_KCV_IMPROVEMENT.command
```

Repeating a command resumes matching caches. Different models/prompts/settings require fresh output paths. First pass uses a 256-token budget; review uses temperature 0, seed 20260920, JSON schema, and a 512-token budget. Results appear in `outputs/kcv_review/RESULTS.md`, `metrics.json`, `predictions.jsonl`, `comparison.json`, and `run_manifest.json`, alongside prompts and saved responses. Different model builds/Ollama versions can produce different fresh scores.

## Verify published labels without model inference

After preparing the pilot above:

```bash
python3.11 -m secure_rag evaluate --examples data/pilot/examples.jsonl --predictions reports/evidence/decision_review_reconstructed_predictions.jsonl --output outputs/published_labels_metrics.json
```

This scores reconstructed terminal labels and deterministic decisions. It verifies counts without claiming to recreate the original model-response cache. See [provenance](reports/evidence/README.md).

## Independent holdout evaluation

Keep the method fixed and run both stages on new CVE sources:

```bash
python3.11 -m secure_rag prepare-claim-evidence --examples data/holdout/examples.jsonl --output data/holdout/claim_evidence_examples.jsonl
python3.11 -m secure_rag run-reasoned --examples data/holdout/claim_evidence_examples.jsonl --output outputs/kcv_review_holdout/first_pass_predictions.jsonl --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct --seed 20260920 --progress
python3.11 -m secure_rag run-kcv-review --examples data/holdout/examples.jsonl --predictions outputs/kcv_review_holdout/first_pass_predictions.jsonl --output-dir outputs/kcv_review_holdout
```

Keep pilot and holdout metrics separate. Do not tune against holdout labels. Holdout accuracy is currently unmeasured in the published report.

## Code map and validation

| Module | Responsibility |
|---|---|
| dataset.py, pilot.py | Benchmark validation, source-grouped splits, hashes |
| corpus.py, retrieval.py | Source normalization, entity routing, dense/BM25 retrieval |
| stages.py | Context compaction, review selection/merge, reranking, verification |
| kcv_facts.py | Narrow evidence-based CVSS checks |
| runner.py | Endpoint calls, verdict validation, checkpoint/resume |
| evaluation.py, comparison.py | Metrics and paired changes |
| cli.py | Experiment commands |

```bash
python3.11 -m unittest discover -s tests -v
python3.11 -m secure_rag --help
```

**48 tests passed** before publication. They cover success/failure paths, wrong IDs, input preservation, cache isolation, and independence from task names/answer labels. Ruff E4/E7/E9/F passed. GitHub Actions runs tests and lint on Python 3.11 and 3.13. Legacy E0–E4 commands remain in [RUN_ON_MAC.md](RUN_ON_MAC.md); this README and the latest report supersede historical statuses.

## Scope and attribution

The method was developed after inspecting pilot errors. The 87.50% KCV score is a development-pilot result, not a full-benchmark score or proof of outperforming the base paper. Current external CVE snapshots can differ from historical benchmark context. Source-disjoint validation is the next check.

Original implementation code is MIT licensed. Upstream benchmark/data/model licenses remain with their owners. Base paper: [ACSAC 2024 DOI](https://doi.org/10.1109/ACSAC63791.2024.00019). [Official benchmark](https://github.com/aiforsec/SECURE).
