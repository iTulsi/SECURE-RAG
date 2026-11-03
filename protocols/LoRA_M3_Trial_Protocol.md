# Qwen2.5 3B: small MLX adapter trial on an M3 Pro

Status: software checks completed in Linux; native Mac/Metal training and accuracy
are unmeasured until the author runs the kit. No guaranteed accuracy gain or
publication novelty is claimed.

## Purpose and scope

Test whether supervised adapter training improves KCV claim verification and
VOOD abstention relative to the same quantized base model in the same MLX runtime.
This is a small feasibility/optimization trial, not the final trained model or a
new untouched test. It reuses the pinned official SECURE data and source splits.
No new labels, generated reasoning traces, external training corpus, or answer
rules are introduced. All inference verdicts come from the model.

The original six-task study, V2 validation and fresh V3 results remain preserved.
This trial cannot establish improvement across all six tasks. If it succeeds,
later work must measure regressions on the other four tasks and use a genuinely
unobserved source-grouped test. Re-evaluating already observed test results after
training is a diagnostic, not a new generalization result.

## Fixed setup

- Author's hardware: Apple M3 Pro, 19,327,352,832 bytes = 18 GiB unified memory.
- Native arm64 Python 3.11+ and macOS; separate `.venv-lora` environment.
- `mlx-lm[train]==0.32.2`, `mlx==0.32.2`; other resolved core package versions are
  recorded and checked on resume.
- Public MLX checkpoint: `mlx-community/Qwen2.5-3B-Instruct-4bit`, revision
  `4f83f8f146fdf28b512a06562b671d7af4fab457`. The model's original license applies.
- 4-bit base weights; LoRA rank 8, scale 16, no dropout, final four layers.
- Batch size 1, gradient accumulation 1, gradient checkpointing, learning rate
  0.00001, seed 20261101, maximum training sequence 4,096 tokens.
- Prompt tokens are masked from training loss. The completion is only T, F or X.
- Smoke: 8 iterations on the two longest surviving development pairs, with the
  two longest surviving validation pairs for loss checks. This exercises the
  largest sequence rather than relying on random short examples. Then compare
  both models on two validation KCV/VOOD pairs (8 inference calls).
- Fixed trial: 100 iterations from the original base, then all 106 validation
  examples on both models (212 calls). Smoke weights are not continued.
- Final checkpoint is used. There is no silent selection of the best of many runs.

The MLX allocator is capped at 12 GiB with a 256 MiB cache. This is a conservative
configuration choice, not a measured guarantee of fitting or a cap on total
system memory. Stop the loaded Ollama model and close memory-heavy applications
before the smoke run. If native training fails, preserve its log and adjust a
new, explicitly versioned run; do not claim the unsuccessful run was completed.
No runtime estimate is claimed before the Mac smoke measurement.

## Data and evidence preservation

Only the original development split is used for training: 131 KCV plus 131 VOOD
examples from 34 connected source groups. Original validation supplies 53 KCV and
53 VOOD examples from 14 separate source groups. The development labels are
46 T, 84 F and 132 X (including the original KCV X, which is retained).
This small training set can overfit and is insufficient to assume broad adaptation.

Training and loss-validation messages are tokenized using the pinned model's
native chat template. If either member of a KCV/VOOD pair exceeds 4,096 tokens,
both members are omitted from training/loss data and explicitly reported. There
is no evidence or answer truncation. In an offline check using the pinned BPE
tokenizer and its no-tools message serialization, all 368 messages fit; the
maximum complete sequence was 3,726 tokens. The native Mac preparation is the
authoritative check and writes exact per-example counts.

Validation inference always includes the full 106 examples, even if an example
were omitted from loss computation. Input messages contain no gold labels or
assistant answers. Labels are kept in a separate scoring file. No old test or
fresh V3 example is added to training. Data/model/code fingerprints and the
native runtime environment are recorded. A source-manifest mismatch is rejected.

## Matched evaluation

Both arms use the exact same full original prompt, explicit Qwen system message,
MLX tokenizer, 4-bit base, greedy decoding and 16-token output budget. Do not
compare the trained MLX adapter directly with the old Ollama Q4_K_M scores as
though only training changed. The checkpoint conversion, quantization and runtime
are different; the matched MLX base is the control.

Decisions must be exactly T/F/X after whitespace stripping and finish normally.
Length-limited output, extra prose or invalid labels are counted as invalid.
Report KCV and VOOD separately: accuracy, macro-F1, invalid counts, coverage,
confusion counts, paired fixes/regressions, source-cluster bootstrap intervals
and observed call time. No pooled headline metric is used. A constant label
chosen from development separately for each task is a diagnostic control:
always F for KCV, always X for VOOD. These controls are not inference rules.

Smoke accuracy has no research interpretation. The 100-step run is still observed
validation, one model build and one training seed. A gain is a selection finding,
not proof of general reliability, novelty, or superiority over the original paper.
Using LoRA alone is not a new research contribution.

## Running on the Mac

After applying the patch to the completed V3 project:

```bash
cd ~/SECURE-RAG-benchmark
ollama stop qwen2.5:3b-instruct
bash RUN_LORA_TRIAL.command quick
```

The first call installs an isolated MLX environment and downloads the pinned
public model (several GB including dependencies/cache). It prepares data,
performs the eight training steps and makes the eight matched inference calls.
It does not modify the installed Ollama model or completed result folders.

Once the smoke run completes without memory/runtime errors:

```bash
cd ~/SECURE-RAG-benchmark
caffeinate -i bash RUN_LORA_TRIAL.command run
```

The fixed run stores `outputs/lora_m3_v1_run/`. To share complete evidence:

```bash
cd ~/SECURE-RAG-benchmark
zip -qr ~/Desktop/SECURE_RAG_LoRA_Trial_Results.zip \
  outputs/lora_m3_v1_run data/lora_trial_v1
open -R ~/Desktop/SECURE_RAG_LoRA_Trial_Results.zip
```

Completed training is reused after validating hashes; inference resumes successful
cached responses. An interrupted training job is preserved and cannot silently
restart because native optimizer state is not being restored. Use a new output
folder for an explicit restart:

```bash
PYTHONPATH=src .venv-lora/bin/python -m secure_rag.lora_trial quick \
  --output-dir outputs/lora_m3_v1_quick_retry1
```

## References

- MLX native training interface: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- Base conversion: https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit
- Original model: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- LoRA: https://arxiv.org/abs/2106.09685

Read APIs and CLI defaults against the pinned 0.32.2 wheel, rather than assuming
the latest documentation and installed software always match.
