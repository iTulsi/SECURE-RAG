# Dataset and split reproduction

| Partition | Examples | Membership |
| --- | ---: | --- |
| LoRA training subset of development | 262 | [train IDs](splits/train_ids.txt) |
| Benchmark development | 897 | [development IDs](splits/development_ids.txt) |
| Benchmark validation | 804 | [validation IDs](splits/validation_ids.txt) |
| Benchmark test | 2,367 | [test IDs](splits/test_ids.txt) |

The LoRA trial uses 106 KCV/VOOD validation examples. See the [trial manifest](manifests/lora_trial_v1.json) and [protocol](../protocols/LoRA_M3_Trial_Protocol.md).

From the repository root, use Python 3.11 or later to regenerate prompts and labels from the original benchmark. This step requires no model.

```bash
git clone https://github.com/aiforsec/SECURE.git data/raw/SECURE
git -C data/raw/SECURE checkout --detach a2412e8ab4b6051ba7381d4f590cb48d9743c7c8
PYTHONPATH=src python3 -m secure_rag.benchmark prepare --data-dir data/raw/SECURE/Dataset --output-dir data/full
python3 -c 'import json; from pathlib import Path; assert json.loads(Path("data/full/manifest.json").read_text()) == json.loads(Path("data/manifests/full_benchmark.json").read_text())'
```

The generated `data/full/inputs.jsonl` and `data/full/labels.jsonl` use matching example IDs. The inputs carry the benchmark partition in the `split` field. Use the training IDs above to select the adapter training rows. For the exact MLX chat-format files, follow the trial protocol.

The upstream files remain with the [SECURE benchmark](https://github.com/aiforsec/SECURE). The project tracks split membership and input hashes.
