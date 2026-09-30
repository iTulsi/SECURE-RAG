# SECURE-RAG v0.6 - project review and execution guide

**Historical extension guide.** Its pending-inference status predates the
completed Mac experiments. The multi-source run achieved 30.56% KCV and
65.28% overall. Later decision review achieved 87.50% KCV and 93.75% overall
using original context. See [the current README](README.md) and
[latest results](reports/Latest_Results.md) for the current status.

Prepared for Tulsi Tomar, B.Tech CSE (Cybersecurity), Bennett University.

## Problem and implemented contribution

Cybersecurity questions need precise evidence about the correct vulnerability.
A small language model can answer unsupported claims, confuse similar CVEs,
or abstain even when evidence is available. Adding more documents alone does
not establish better accuracy.

The implemented extension imports multiple official evidence datasets,
records their provenance, selects documents by entity identity, ranks only
eligible evidence, and produces auditable prompts for the existing model
runner. SECURE remains the evaluation benchmark; its answers are unchanged.
Runtime code uses the Python standard library; Ruff is a development tool.

## Dataset roles

| Dataset | Role | Native format | Selection |
|---|---|---|---|
| SECURE KCV + VOOD | Evaluation only | Two official TSVs | 466 paired questions per task |
| CVE List | Descriptions, affected products, CVSS and remediation | CVE JSON 5 | Exact CVE identity |
| NVD | Independent enrichment of CVE descriptions, metrics and weaknesses | API 2.0 JSON | Exact CVE identity |
| CISA KEV | Positive evidence of known exploitation | Catalog JSON | Exact CVE identity |
| CWE | Definitions of linked weakness types | XML or zipped XML | CWE explicitly present in available context/question |
| MITRE ATT&CK | Technique descriptions | STIX bundle JSON | Explicit technique ID in the question |

The saved snapshot counts and hashes are in `reports/Multisource_Validation.json`.
Each normalized passage has a source, URL, entity ID, chunk index, record
hash, and deterministic passage ID. Duplicate passages within a source are
removed; different sources retain separate provenance.

These datasets have different purposes. CWE and ATT&CK do not supply new
KCV/VOOD labels, and this work does not fine-tune model weights. ATT&CK is
supported and tested, but SECURE KCV questions need not mention a technique
ID, so it may contribute zero passages to this benchmark.

The completed corpus contains **6,812 passages**: 124 CVE records, 124 NVD
records, 1,729 KEV entries, 969 CWE weaknesses, and 697 active ATT&CK
techniques. The full SECURE BM25 run selected CVE, NVD, CWE, and original
context passages; KEV and ATT&CK supplied no top-five passages in this run.
Importing a source does not prove it improved the benchmark.

## Execution flow

1. Validate official KCV and VOOD data and pin the upstream commit.
2. Import local official snapshots into one evidence JSONL and hash manifest.
3. Inspect whether the input prompt actually supplies context. An absent
   context produces an empty evidence pool, even if its metadata contains a CVE.
4. When context is available, use its CVE and source URL to route CVE/NVD/KEV
   records; include linked CWE entries and explicitly requested ATT&CK entries.
5. Rank eligible passages with BM25, or reuse dense retrieval + BM25 with RRF.
6. Send the top five cited passages to the existing T/F/X runner. Optional
   reranking and verification use the existing E3/E4 commands.
7. Evaluate fresh predictions and compare KCV accuracy/coverage and VOOD
   abstention separately. Inspect disagreements before changing the pipeline.

The protocol is **context-available augmentation**, not unrestricted external
search. Context presence decides evidence access; task names and gold labels
are never used by the retrieval decision. In a deployment that retrieves
records for context-free questions, original VOOD labels would no longer
represent the same experiment. Such a deployment needs a separately labeled
evaluation set.

## Results and their limits

The following values were recomputed from the prediction files in the saved
tested v0.5 archive; they are historical pilot results, not new multi-source
scores. The pilot has 72 KCV and 72 VOOD examples from 20 CVE sources.

| Experiment | KCV accuracy | KCV coverage | VOOD accuracy | Overall accuracy |
|---|---:|---:|---:|---:|
| E0 original prompts | 45.83% | 55.56% | 61.11% | 53.47% |
| E1 dense retrieval | 40.28% | 48.61% | 100.00% | 70.14% |
| E5 context routing, post hoc | 45.83% | 55.56% | 100.00% | 72.92% |
| E6 CVSS rules, post hoc | 48.61% | 55.56% | 100.00% | 74.31% |
| v0.6 multi-source model inference | Pending | Pending | Pending | Pending |

Accuracy counts all examples, including wrong abstentions. KCV coverage is
the fraction answered with T/F. VOOD measures whether the system returns X
when evidence is withheld. Overall accuracy can rise through VOOD alone;
it cannot prove improved factual KCV reasoning.

E2-E4 raw results were not in the archive inspected here; consult the original
Mac outputs before citing their scores. E5/E6 were designed after pilot
inspection and are exploratory. Their stored scores do not establish general
performance. The optional CVSS rules cover limited phrasing and should be
evaluated separately from the model.

## Completed validation

- 36 automated tests pass, covering native source formats, deterministic
  provenance, unsafe XML rejection, invalid/duplicate records, exact entity
  filtering, label-independent routing, empty-evidence preservation, hybrid
  integration, path protection, source-disjoint selection, and prior stages.
- Ruff correctness checks and `git diff --check` pass.
- Real BM25 retrieval was executed on the pilot, source-disjoint set, and
  all 932 benchmark examples. Retrieval counts are in the validation JSON;
  they are passage counts, not answer accuracy.
- Pilot retrieval selected 358 passages, the source-disjoint set selected
  365, and the full benchmark selected 2,327. All 466 context-free full-set
  examples retained empty evidence pools.
- The source-disjoint set contains 73 KCV + 73 VOOD questions from 20 CVE
  source URLs excluded from the original pilot, using seed `20261001`.
- Fresh full-benchmark inference and fresh embedding-model inference are
  pending. Tests use mock model responses to test plumbing; none are reported
  as measured model performance.

## Quick review demo

Unzip the accompanying review package into a new folder; keep your old Mac
experiment folder. The package contains current code, source snapshots,
normalized corpus, all benchmark prompts, retrieval audits, and old pilot
predictions. The prepared Git change contains code and reports; generated datasets
remain outside Git history.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

secure-rag build-corpus \
  --cve-dir data/raw/cve --nvd data/raw/nvd.json \
  --kev data/raw/kev.json --cwe data/raw/cwe.xml.zip \
  --attack data/raw/attack.json --output data/corpus/evidence.jsonl

secure-rag prepare-multisource \
  --examples data/pilot/examples.jsonl --corpus data/corpus/evidence.jsonl \
  --output data/pilot/multisource_bm25_examples.jsonl \
  --retrieval-output outputs/multisource_bm25_retrieval.jsonl
```

Open `outputs/multisource_bm25_retrieval.jsonl` to show exact CVE routing,
source citations, and the empty VOOD pools. Open the generated examples to
show the actual prompts sent to the model. `reports/Multisource_Validation.json`
summarizes the real retrieval run.

## Fresh model run on the Mac

Use the same Qwen model and seed for a comparable baseline. Ollama must be
running (`ollama serve` in a separate terminal if needed).

```bash
ollama pull qwen2.5:3b-instruct
ollama pull embeddinggemma
curl http://localhost:11434/api/tags

secure-rag run-baseline \
  --examples data/pilot/multisource_bm25_examples.jsonl \
  --output outputs/multisource_bm25_predictions.jsonl \
  --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct \
  --seed 20260920

secure-rag evaluate --examples data/pilot/examples.jsonl \
  --predictions outputs/multisource_bm25_predictions.jsonl \
  --output outputs/multisource_bm25_metrics.json

secure-rag compare-runs --examples data/pilot/examples.jsonl \
  --baseline outputs/e0_predictions.jsonl \
  --candidate outputs/multisource_bm25_predictions.jsonl \
  --retrieval outputs/multisource_bm25_retrieval.jsonl \
  --output outputs/e0_multisource_comparison.json
```

The model runner checkpoints after every prediction. Repeat the same command
to resume; use a fresh output name when prompts, seed, or model change.

For the dense + BM25 experiment:

```bash
secure-rag prepare-multisource --strategy hybrid \
  --examples data/pilot/examples.jsonl --corpus data/corpus/evidence.jsonl \
  --output data/pilot/multisource_hybrid_examples.jsonl \
  --retrieval-output outputs/multisource_hybrid_retrieval.jsonl \
  --embedding-cache outputs/multisource_embeddings.jsonl \
  --base-url http://localhost:11434/v1 --model embeddinggemma

secure-rag run-baseline \
  --examples data/pilot/multisource_hybrid_examples.jsonl \
  --output outputs/multisource_hybrid_predictions.jsonl \
  --base-url http://localhost:11434/v1 --model qwen2.5:3b-instruct \
  --seed 20260920
```

Use `evaluate` on the new predictions and original pilot labels. For the
holdout, use `data/holdout/examples.jsonl` and
`data/holdout/multisource_bm25_examples.jsonl`; first run an original-prompt
baseline on that same holdout. For the full benchmark, use the matching
`data/full/` files and fresh `outputs/full_*` names. Do not compare a full-set
candidate against the 144-row pilot baseline.

For a one-example live demonstration, add `--limit 1` to `run-baseline`.
This produces one real prediction; strict evaluation needs the completed set.

## What to explain to the reviewer

"We expanded the evidence corpus with CVE, NVD, KEV, CWE and ATT&CK, while
keeping SECURE as the unchanged evaluation benchmark. We route by entity
before retrieval and retain source hashes and citations. We also prepared
a separate CVE-disjoint evaluation set. The code and real retrieval runs
are complete; new accuracy requires the local Qwen inference run. We have
not established a 90% KCV score."

Main limitations: the current external snapshots are newer than SECURE's
embedded CVE records and can disagree with historical gold answers; chunking
can separate important facts; the 3B model may lack reasoning capacity; an
empty-evidence cue can explain high VOOD scores; generator/verifier agreement
is not independent validation when they use the same model. The full set
overlaps the development pilot and is a descriptive evaluation, not untouched
test evidence. The separate source-disjoint set is the primary next check.

Next controlled experiments: original prompts vs entity-routed BM25 vs
entity-routed hybrid, then reranking on the same source-disjoint set. Report
source usage and unsupported answers alongside accuracy. Freeze snapshots
and settings before inspecting holdout outcomes. No calibration threshold
was tuned in this extension.

## Source provenance and terms

Source snapshots were obtained during this implementation session. The CISA
website feed returned HTTP 403, so the KEV file came from CISA's official
`cisagov/kev-data` repository. MITRE ATT&CK imports active attack-patterns
only, excluding revoked/deprecated entries.

- SECURE: https://github.com/aiforsec/SECURE at `a2412e8ab4b6051ba7381d4f590cb48d9743c7c8`
- CVE: https://github.com/CVEProject/cvelistV5
- NVD: https://nvd.nist.gov/developers/vulnerabilities
- KEV: https://github.com/cisagov/kev-data and https://www.cisa.gov/known-exploited-vulnerabilities-catalog
- CWE: https://cwe.mitre.org/data/downloads.html
- ATT&CK: https://github.com/mitre-attack/attack-stix-data

Original data retains its source terms; the project's MIT license applies to
project code. NVD notice: this product uses data from the NVD API but is not
endorsed or certified by the NVD.
