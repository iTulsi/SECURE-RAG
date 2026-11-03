# SECURE model-only evaluation protocol

Status: dataset audited and splits prepared; new model performance is **unmeasured**.
The old 93.75% development-pilot score is not an outcome of this experiment.

## Research question and possible contribution

When does evidence-first answering and one model self-review help or hurt a
small language model across the six SECURE tasks, relative to original prompts
on exactly the same questions? Does any gain survive source grouping, duplicate
sensitivity, another model, another seed, invalid-output accounting and cost?

This is an empirical study of an existing family of methods, not a claim to
invent self-refinement. Self-Refine already studied same-model refinement;
subsequent research found intrinsic correction can also degrade reasoning:

- Madaan et al., Self-Refine, NeurIPS 2023: https://arxiv.org/abs/2303.17651
- Huang et al., Large Language Models Cannot Self-Correct Reasoning Yet,
  ICLR 2024: https://arxiv.org/abs/2310.01798
- Bhusal et al., SECURE, ACSAC 2024:
  https://doi.org/10.1109/ACSAC63791.2024.00019

A useful contribution requires reproducible, task-specific findings that add
to this literature. Larger sample size alone does not establish novelty or
publication readiness. Negative results and measured tradeoffs are valid outcomes.

## Data and separation

Official repository: https://github.com/aiforsec/SECURE

Pinned revision: `a2412e8ab4b6051ba7381d4f590cb48d9743c7c8`.
Every TSV is checked against its exact SHA-256 before preparation. Original
labels and prompt text are retained. One completely blank CWET row (zero-based
index 572) is excluded and recorded. Other incomplete rows cause failure.

| Task | Scorable official rows | Development | Validation | Test |
|---|---:|---:|---:|---:|
| MAET | 1,072 | 192 | 278 | 602 |
| CWET | 964 | 223 | 169 | 572 |
| KCV | 466 | 131 | 53 | 282 |
| VOOD | 466 | 131 | 53 | 282 |
| RERT | 1,000 | 203 | 228 | 569 |
| CPST | 100 | 17 | 23 | 60 |
| Total | 4,068 | 897 | 804 | 2,367 |

This is 28.25 times the 144-question pilot in available scorable questions;
the new test split alone is approximately 16.44 times that pilot. These are
question counts, not independent-source counts or evidence of improved quality.

The dataset contains 1,490 connected groups. Connected source URLs, CVE mentions
across KCV/VOOD/RERT, exact duplicate prompts and repeated CVSS vectors stay in
one split. CWE concepts shared across different records are not treated as the
same record. Hash allocation uses seed 20261031 and approximate 20/20/60 group
fractions, rather than equal question counts. Prior pilot sources and everything
connected to them are forced into development. Prior holdout source groups are
forced into test unless linked to the pilot; the audit found zero such conflicts.
No analyst utilities or pretrained-model contamination guarantees follow from
source grouping. Cross-record paraphrases are not automatically deduplicated.

CWET contains 26 exact repeated-prompt excess rows; RERT contains three. They
remain in official full scoring, with unique-prompt sensitivity scores reported
separately. KCV includes one original X label, retained unchanged.

There is no weight training in this experiment. Development is for prompt
design, validation for method selection, and test for the frozen comparison.
Calling development a model-training split would misrepresent what happened.
`inputs.jsonl` contains no gold labels; `labels.jsonl` is joined only for scoring.
Third-party data is fetched from its original repository; project MIT licensing
does not relicense those sources.

## Prespecified arms and controls

1. **official:** the unmodified official prompt, with the model's textual answer.
2. **evidence_first:** the same complete problem with a short explicit check of
   evidence/clauses and an analysis/answer JSON schema. One generation call.
3. **self_review:** the same problem plus the saved evidence-first response;
   the model rechecks and emits analysis/answer JSON. Two total generation calls,
   including the first pass. Every example is reviewed, irrespective of labels,
   task accuracy, confidence or correctness.

No deterministic X routing, CVSS-answer rules, prior-label retention, external
retrieval, gold labels or hand-selected corrections decide model answers. The
legacy rule pipeline remains archived and is not used by these commands.
Prompting and format constraints change together in the evidence-first arm;
their individual causal effects are not isolated. Self-review is a same-model
second pass, not an independent verifier or a complete replication of Self-Refine.

Matched controls per model: temperature 0.7 (the SECURE paper's reported setting),
seed 20261031, 512 output tokens per call, explicit 32,768-token context window,
same examples, original supplied evidence, no evidence truncation or compaction.
The maximum output allowance matches per call, not total compute for self-review.
A conservative UTF-8 byte bound reserves prompt/output/template space. Oversized
inputs fail instead of being intentionally shortened. Ollama requests also set
`truncate=false` and `shift=false`; verify server support/version in archived runs.

Primary model: `qwen2.5:3b-instruct`. Replication model: `llama3:8b`, the same
model family and nominal size as one SECURE baseline. Quantized/current builds
are not assumed identical to the original study. Model digests, quantization,
server version, full model metadata and exact requests are captured. The code,
dataset, prompts, model identity and settings are sealed before generation.
Changing them requires a new run directory; changing the method after inspecting
test errors requires a fresh independent evaluation, not retuning this test.

Run both models and seeds 20261031, 20261101, 20261102 before making robustness
claims. Start with a short runtime sample under the already fixed protocol.
Do not choose a favorable seed after seeing results. Further models, seeds or
external datasets are extensions and must be identified as such.

## Metrics and failure handling

- MAET/CWET/KCV: accuracy, macro F1, confusion counts, invalid predictions and
  abstention rate. KCV also reports binary coverage and accuracy among answers.
- VOOD: X accuracy, invalid predictions and unsupported binary answer rate.
  Since every VOOD gold label is X, this does not measure calibrated uncertainty
  on a natural distribution of answerable/unanswerable questions. Modern model
  training data may also include these 2024 vulnerabilities.
- RERT: ROUGE-L F1, lowercased ASCII alphanumeric tokenization, no stemming,
  beta=1. Empty, malformed and truncated outputs receive zero. Lexical overlap
  alone does not establish factual correctness. Original scorer parity is not
  verified, so do not claim to outperform its published 0.53/0.54 scores.
- CPST: valid-only MAD plus valid coverage and a conservative all-row MAD
  assigning the maximum absolute-error penalty of 10 to invalid answers.
  Never report only valid-case MAD while hiding failed generations.
- Every task: unique-prompt sensitivity, source-group count, paired changes and
  95% source-cluster bootstrap intervals (2,000 samples, seed 20261031).
- Execution: calls, prompt/output tokens, per-stage wall time, and the full
  two-call cost for self-review. Cache reuse is not zero inference cost.

There is no pooled score combining accuracy, ROUGE and MAD. Intervals are
exploratory and not multiplicity-corrected; do not select one significant task
and imply improvement across all six. Independent-source coverage is especially
limited for small tasks. Report uncertainty and all regressions.

All complete model responses and exact requests are durably appended before
moving to the next call. Malformed model answers are retained and scored as
failures, not retried until correct. Network failures stop and preserve prior
calls. Resume validates hashes, prompt dependencies and settings. An incomplete
run cannot produce a complete-split score. A damaged JSONL cache fails closed;
preserve it for audit before repairing a partial final filesystem write.
Checksums detect accidental changes; they do not authenticate a fabricated run.

## The user's three assessment questions

| Question | Evidence required | Current conclusion |
|---|---|---|
| Useful to future researchers? | Public runnable code, source/label audit, fixed splits, full raw responses, transferable findings, comparison to prior self-correction work | Infrastructure prepared; novelty and empirical findings pending |
| Actually trustworthy? | Complete unseen-source results, honest invalid accounting, paired uncertainty, cost, second-model/seed replication, manual factual audit | No new model trustworthiness result yet |
| How much better than original work? | Matched same-model deltas per task; separate descriptive comparison with published scores on the full compatible dataset | Unmeasured; no superiority claim |

The scorer includes all seven original Table V model rows as historical context.
The scientifically primary improvement is our method minus original prompts on
the **same model, questions and settings**. Accuracy changes are percentage
points; CPST improvement is a reduction in MAD. Historical comparisons with
different models, data subsets, model ages and manual output correction cannot
attribute any difference to our method. A new Llama3-8B run is a contemporary
reproduction attempt, not the original authors' cached run.

Before paper submission: inspect prespecified samples of RERT outputs against
the supplied overview and all unexpected KCV errors; have a second reviewer
check a subset with a written rubric. Record actual annotations and disagreements,
never invented reviewer scores. Test transfer to a separately curated, dated
source-disjoint external collection before making broader deployment claims.
Neither this protocol nor a larger dataset guarantees publication.
