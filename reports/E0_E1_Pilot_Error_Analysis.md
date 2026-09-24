# SECURE-RAG E0–E1 Pilot Error Analysis

## Experiment identity

- Benchmark: SECURE KCV and VOOD
- Official upstream revision: `a2412e8ab4b6051ba7381d4f590cb48d9743c7c8`
- Pilot seed: `20260920`
- Pilot size: 20 CVE sources, 72 KCV examples, and 72 paired VOOD examples
- Generator: `qwen2.5:3b-instruct` (`357c53fb659c` in the recorded Ollama run)
- E1 embedding model: `embeddinggemma:latest` (`85462619ee72`)
- E0: official prompts without retrieval
- E1: top-five dense retrieval over evidence supplied inside each example

These are pilot results. They must not be compared directly with the SECURE
paper's full-dataset model scores.

## Main results

| Metric | E0 | E1 dense | Change |
|---|---:|---:|---:|
| Overall accuracy | 53.47% | 70.14% | +16.67 pp |
| Overall macro-F1 | 54.70% | 64.61% | +9.91 pp |
| KCV accuracy | 45.83% | 40.28% | -5.56 pp |
| KCV macro-F1 | 60.93% | 57.13% | -3.80 pp |
| KCV coverage | 55.56% | 48.61% | -6.94 pp |
| Accuracy among answered KCV examples | 82.50% | 82.86% | +0.36 pp |
| VOOD accuracy | 61.11% | 100.00% | +38.89 pp |
| Unsupported-answer rate | 38.89% | 0.00% | -38.89 pp |
| Abstention F1 | 59.46% | 79.56% | +20.10 pp |

E1's overall gain came from 28 corrected VOOD examples minus a net loss of four
correct KCV examples. Dense retrieval did not improve supported-question
accuracy or coverage.

## Paired transitions

### KCV

| Transition | Examples |
|---|---:|
| Correct in both E0 and E1 | 26 |
| Correct in E0, wrong in E1 | 7 |
| Wrong in E0, correct in E1 | 3 |
| Wrong in both | 36 |

E1 produced 37 KCV abstentions, up from 32 in E0. It recovered three E0 errors
but introduced seven regressions.

### VOOD

| Transition | Examples |
|---|---:|
| Correct in both | 44 |
| Wrong in E0, correct in E1 | 28 |
| Correct in E0, wrong in E1 | 0 |

All 72 E1 VOOD predictions were `X`.

## Retrieval versus reasoning failures

Manual inspection of the seven KCV regressions found answer-bearing evidence
inside the retrieved top five for six cases:

| ID | Gold | E1 | Evidence position | Finding |
|---|---:|---:|---:|---|
| `kcv-0009` | F | X | 1 | Explicit “not aware of exploitation” negation retrieved |
| `kcv-0106` | F | X | 3 | CVSS base severity `MEDIUM` retrieved |
| `kcv-0133` | F | X | 2 | CVSS base score `5.3` retrieved |
| `kcv-0134` | F | X | 1–2 | Multiple affected Keenetic models retrieved |
| `kcv-0196` | F | X | 4–5 | Remote/network exploitation explicitly retrieved |
| `kcv-0307` | F | X | Not found | Top five lacked decisive mitigation evidence |
| `kcv-0400` | T | F | 1 | REQUEST_URI_RAW workaround explicitly retrieved |

This separates one primary retrieval miss from six generator
reasoning/calibration failures. A generic reranker cannot fix cases where the
decisive evidence is already ranked first and the generator still returns the
wrong label or abstains.

## Similarity score is not an evidence-sufficiency score

| E1 KCV outcome | Examples | Mean top dense score | Median |
|---|---:|---:|---:|
| Correct | 29 | 0.5593 | 0.5625 |
| Abstained | 37 | 0.5683 | 0.5766 |
| Wrong `T/F` answer | 6 | 0.5955 | 0.5635 |

Wrong and abstained examples did not have lower similarity scores than correct
examples. A threshold on dense similarity would therefore be unjustified on
this pilot.

## VOOD interpretation limitation

E1 writes “No evidence was provided” when an example contains no retrievable
context. This makes missing evidence explicit to the generator. The 100% VOOD
result is valid for the implemented system, but it is not evidence that dense
semantic ranking alone solved OOD detection. It combines retrieval state with
a changed prompt. This limitation must be disclosed in any paper or classroom
presentation.

## E2 decision

The next controlled ablation is BM25 plus dense retrieval with reciprocal rank
fusion. It keeps the same chunks, top-five evidence budget, generator, prompt,
seed, and pilot. Only passage ranking changes. The E1 embedding cache is reused.

BM25 is justified because exact identifiers, CVSS fields, version numbers, and
negation-bearing security terms can be underweighted by dense similarity. The
expected gain is limited because six of seven observed regressions already
contained decisive evidence in the top five.

If E2 does not recover KCV coverage, the next component should target
evidence-to-claim verification rather than apply a generic relevance reranker.
