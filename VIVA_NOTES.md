# SECURE-RAG: five-minute viva notes

**Tulsi Sanskrati Tomar · S24CSEU0084**

## My project in one minute

My project checks whether a small local language model can verify cybersecurity claims from evidence. It uses Qwen2.5 3B and the SECURE benchmark. I compared original prompts, evidence-first reasoning and self-review. I also tested retrieval with citations and trained a small LoRA adapter.

The important finding is that extra reasoning or a citation does not automatically make an answer correct. My cited retrieval candidate performed worse than the full-record baseline on an additional dataset. I kept that negative result in the paper.

The adapter gave a positive result in a controlled comparison. Using the same MLX base model and the same prompts, KCV validation accuracy increased from 64.15% to 79.25%. This is a validation result, and testing on new CVEs is still pending.

## What did the original paper do?

SECURE introduced six cybersecurity task datasets and evaluated multiple language models and customized methods. My work uses its data to run a local evaluation, examine where evidence handling fails, and measure a small supervised adapter. I do not claim to have invented SECURE, RAG or LoRA.

## What do T, F and X mean?

- T: the permitted evidence supports the claim.
- F: the permitted evidence contradicts the claim.
- X: the permitted evidence does not resolve the claim.

False and unknown are different. If a record explicitly says no user interaction is needed, a claim that it is required is false. If the record says nothing about the condition, it may be unknown under that evidence boundary.

## What is LoRA?

LoRA trains small low-rank matrices while keeping the main model weights frozen. It reduces the number of trainable parameters. My adapter trains about 1.66 million parameters, roughly 0.054% of the base model. It uses rank eight in the final four layers and 100 updates.

The base model is stored in 4-bit form. The adapter tensors are F32. I should not describe every part of the system as 4-bit.

## Why train this adapter?

The base model often answered F, including for true claims. The adapter improved true-claim recall from 4/20 to 13/20. It fixed 11 KCV errors but introduced three regressions. That is why the paper reports the class breakdown and does not show only the overall improvement.

## Why use the same MLX base?

Ollama and MLX can differ in quantization and execution. Comparing an Ollama score with an MLX adapter score would mix several changes. My matched experiment uses the same MLX checkpoint, tokenizer, prompt and greedy decoding. Only the adapter changes.

## What data did I use?

The six-task official preparation has 4,068 scorable examples. The recorded test uses 2,367. LoRA uses only 262 development KCV/VOOD examples for training and 106 validation examples for comparison. Training and validation source groups are separate.

The additional V3 collection has 160 claims from 40 CVEs. The labels come automatically from explicit structured fields. Independent human review is not yet complete.

## What failed in retrieval?

The full-record original baseline got 146/160 correct, or 91.25%. The cited candidate got 109/160, or 68.125%. It especially struggled with user-interaction claims. Some answers quoted the correct NONE field but interpreted it backwards.

Of 142 complete answers with a matching quote, 47 decisions were still wrong. A matching citation shows where text came from; it does not prove that the model understood it.

## Why not report one score for everything?

The tasks use different metrics. Classification uses accuracy. RERT uses ROUGE-L text overlap. CPST uses score error, where lower is better. Averaging these would mix incompatible scales.

## Why is 98.11% VOOD not proof of general reliability?

Every validation VOOD label is X. An always-X system gets 100%. The adapter learned to avoid most unsupported binary answers under these prompts, but this alone does not prove broad uncertainty calibration.

## What is my contribution?

My contribution is a reproducible engineering study: task-specific prompt comparisons, a retained negative retrieval result, citation failure analysis and a matched adapter validation improvement on a small local model. The algorithmic components are established. Wider transfer is a future experiment.

## What did I verify?

I checked saved requests and responses, hashes, source separation and adapter tensors. The software has 87 passing tests. I reproduced the archived metrics. These checks validate the artifacts; they are not a claim that another machine independently repeated training.

## What will I do next?

Freeze the step-100 adapter and run a matched base comparison on previously unobserved CVEs, with supported, contradicted and insufficient-evidence examples. Then repeat training seeds and check regressions on the other tasks.
