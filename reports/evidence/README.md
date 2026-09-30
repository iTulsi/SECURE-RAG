# Result evidence and provenance

The repository distinguishes original saved caches from terminal reconstructions.

| Artifact | Origin and permitted interpretation |
|---|---|
| `e0_original_predictions.jsonl` | Original historical E0 cache recovered from the author's project archive |
| `e1_original_predictions.jsonl` | Original historical E1 cache recovered from the same archive |
| `evidence_first_reconstructed_predictions.jsonl` | Labels reconstructed from the uploaded terminal transcript; not the original model-response cache |
| `decision_review_terminal_labels.txt` | The 30 displayed review decisions supplied after completion; transcription omits timings and local paths |
| `decision_review_reconstructed_predictions.jsonl` | Complete labels assembled from those 30 review decisions, earlier reconstructed labels, and unchanged deterministic pipeline rules |
| `decision_review_reconstructed_metrics.json` | Existing evaluator applied to the reconstructed labels; reproduces 87.50% KCV and 93.75% overall |
| `provenance.json` | SHA-256 hashes, routing counts, provenance, and original-cache availability |

The reconstruction has exactly the reviewed IDs selected by `prepare_decision_review`. It does not infer labels from prose, use gold labels to choose answers, invent missing responses, or pretend that reconstructed rows are an original resumable cache. Original full responses, usage records, prompt hashes, and the final run manifest remain on the author's Mac and have not been uploaded here. The reconstruction establishes label counts, not independent model reproducibility or response authenticity.

Prepare the pinned benchmark using the root README, then score the published labels with the existing evaluator. The recomputation matches 63/72 KCV, 72/72 VOOD, and 135/144 overall. Do not use reconstructed labels for training or inference caching. F1 and coverage in reconstructed metrics are derived from these labels, not copied from an uploaded original metrics file.

The best pilot pipeline retains earlier model answers and uses deterministic rules as well as new model calls. Pilot errors informed development. Held-out performance has not yet been reported. See [Latest_Results.md](../Latest_Results.md) for interpretation and the original Mac artifact paths.
