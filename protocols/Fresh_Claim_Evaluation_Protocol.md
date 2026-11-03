# Fixed candidate and fresh structured-fact extension

Status: source data collected and automatically checked; model outcomes are
unmeasured; independent human review is pending.

## Decision before inference

Select the single-pass BM25 top-three cited-answer approach from version-2
validation. Do not add the same-model verifier by default. All completed pilot,
six-task test and validation results remain reportable, including failures.

This extension fixes the two measurement/design issues found during the audit:

- Every input explicitly names the target CVE in the claim and cited/format
  prompts. There is no ambiguous target record.
- Primary `verdict_accuracy` measures the explicit T/F/X decision. The existing
  `accuracy` field remains the stricter complete-response endpoint, and
  `complete_response_rate` is reported separately. A correct X with an empty
  explanation is a correct decision with an incomplete response. The prospective
  citation schema additionally requires a nonempty explanation.

No old score is silently rewritten. Applying this code update changes the code
fingerprint, so resuming an old inference folder will be rejected. Preserve its
original ZIP and use the new `outputs/fresh_v3_*` directory for this experiment.

## Fresh data, accurately described

The draft contains **40 distinct 2025 CVEs and 160 template-generated claims**:
80 T and 80 F. Each record supplies one true and one false statement about CNA
CVSS v3.1 base severity, and one true and one false statement about the reported
user-interaction field. Each statement says precisely that it concerns the
CNA's CVSS v3.1 assessment, not the whole vulnerability's behavior.

The label is derived from an explicit JSON pointer and stored source value.
The false claim changes that value to a different valid category. These are
automatically generated, source-verifiable structured-fact claims. They are not
expert-authored natural-language reasoning questions or independent human labels.
They are a controlled extension alongside the original 4,068-question benchmark.
Do not present performance here as general cybersecurity accuracy.

Source selection is independent of any model predictions:

1. Start from the official CISA KEV mirror at revision
   `7ccba8e1223ccec82d12c7736bf3fae11e5bee86` (catalog 2026.10.30).
2. Filter identifiers to CVE-2025 and order by the hash of `[20261102, CVE ID]`.
3. Fetch CVE List records at revision
   `6125463712241a8a7cca97b445c85415f26a8b09`.
4. Require PUBLISHED state, a 2025 publication date, and complete, nonconflicting
   CNA CVSS v3.1 severity/user-interaction values. Reject oversized inputs
   instead of truncating the baseline's original record.
5. Take the first 40 eligible records. Record every exclusion and source hash.

The source audit found zero CVE-ID overlap with the original prepared SECURE
data, and checked all 160 gold labels directly against their source pointers.
The maximum prepared prompt length is 17,820 UTF-8 bytes, below the conservative
32,768-context guard. Before the fortieth eligible record, 62 records were
excluded for missing/incomplete/conflicting v3.1 facts or non-2025 publication.
The pinned manifest gives the complete list. No exclusion uses model accuracy.

This population is biased toward KEV-listed exploited vulnerabilities with
these available fields and manageable record lengths. It is not representative
of all CVEs, industrial systems, or the original benchmark's six tasks. New CVE
IDs do not prove absence of pretrained-model contamination. Snapshots retain
their original source terms; the project's MIT license does not relicense them.

## Matched comparisons: 480 calls

| Arm | Input / output | Calls |
|---|---|---:|
| Original-style baseline | Full original JSON record, explicit CVE claim, T/F/X | 160 |
| Retrieval/format control | Same top-three passages as the candidate, JSON verdict only | 160 |
| Selected candidate | Top-three passages, model verdict, quote, comparison and confidence | 160 |

These are three single-pass methods; there is no verifier stage. All use the
same installed model build, temperature 0.7, seed 20261102, 512 output tokens
and requested 32,768-token context. The quick mode runs one CVE (four claims,
12 calls) and resumes those calls in the full run.

The baseline is modeled on the SECURE prompt interface, not a reproduction of
the original authors' model/provider runs. The format control changes both
retrieval representation and output format relative to full JSON. Comparing it
with the cited candidate tests the combined citation/reasoning/confidence
instruction bundle; it is not a pure isolated test of citations alone.

## Prespecified interpretation

Report the three verdict accuracies, complete-response accuracies, response
completeness, errors, binary coverage, citation fidelity and model confidence
diagnostics separately. Report a fixed always-F baseline (50% on this balanced
set), paired CVE-cluster bootstrap intervals, total calls/tokens/wall time and
all regressions. Four statements about one CVE are not independent sources.
Selecting this candidate after validation makes this a prospective extension;
do not change the method after inspecting these new model results and still
call the same collection an untouched test.

No facts or gold labels select an inference verdict. Citation checking audits
literal text matching and never substitutes T/F/X. It does not establish that
the quote entails the claim. Twenty source-separated cited outputs are sampled
for a human evidence audit after completion; annotations remain null until
someone performs that review.

`data/fresh_claims/label_review.jsonl` also contains the 40 source records' claim
lists and field pointers, with blank reviewer/confirmation fields. Review the
actual source values and labels; do not claim that those reviews occurred
merely because the template exists. Report this as machine-labeled data until
review evidence is available.

## Stopping rule and the research goal

Finish this fixed comparison once, then assess whether the candidate beats
both the original-style baseline and retrieval/format control at acceptable
cost. If it does not, retain the result and frame the paper as a bounded
evaluation/failure analysis rather than repeatedly choosing new prompts on the
same test until a positive number appears.

The intended useful finding concerns when evidence selection and citations
help or fail small models. Reliability claims still need human entailment
review and a second model/seed; a broad real-world claim needs independently
curated natural claims. Publication readiness and superiority over the original
SECURE paper are not guaranteed by this controlled extension.

Sources:
- https://github.com/cisagov/kev-data
- https://github.com/CVEProject/cvelistV5
- https://doi.org/10.1109/ACSAC63791.2024.00019
