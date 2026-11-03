"""Small, development-only MLX adapter trial with a matched validation baseline."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from .benchmark_data import digest, read_prepared
from .benchmark_metrics import paired_interval, score_task
from .dataset import sha256_file
from .evaluation import _read_jsonl, write_metrics
from .pilot import write_jsonl

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
REVISION = "4f83f8f146fdf28b512a06562b671d7af4fab457"
MLX_LM_VERSION = "0.32.2"
SEED = 20261101
MAX_SEQUENCE = 4096
MEMORY_LIMIT = 12 * 1024**3
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def messages(row: dict, target: str | None = None) -> list[dict]:
    result = [{"role": "system", "content": SYSTEM},
              {"role": "user", "content": row["prompt"]}]
    if target is not None:
        result.append({"role": "assistant", "content": target})
    return result


def prepare_trial(data_dir: Path, output_dir: Path, identity: dict,
                  token_count: Callable[[list[dict]], int]) -> dict:
    rows, gold, source_manifest = read_prepared(data_dir)
    selected = [r for r in rows if r["task"] in {"KCV", "VOOD"}
                and r["split"] in {"development", "validation"}]
    if not selected or any(gold[r["id"]] not in {"T", "F", "X"} for r in selected):
        raise ValueError("Missing or invalid claim training data")
    ownership, pairs = {}, defaultdict(list)
    for row in selected:
        if ownership.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("Source group crosses training and validation")
        pairs[row["id"].split("-", 1)[1]].append(row)
    train, valid, included, excluded = [], [], [], []
    for pair, members in sorted(pairs.items()):
        if (len(members) != 2 or {r["task"] for r in members} != {"KCV", "VOOD"}
                or len({(r["split"], r["group_id"]) for r in members}) != 1):
            raise ValueError(f"Unaligned KCV/VOOD pair: {pair}")
        lengths = {r["id"]: token_count(messages(r, gold[r["id"]])) for r in members}
        if max(lengths.values()) > MAX_SEQUENCE:
            excluded.append({"pair": pair, "split": members[0]["split"], "tokens": lengths,
                             "reason": "Whole pair excluded from training/loss only; no truncation"})
            continue
        destination = train if members[0]["split"] == "development" else valid
        for row in members:
            destination.append({"messages": messages(row, gold[row["id"]])})
            included.append({"id": row["id"], "split": row["split"], "tokens": lengths[row["id"]]})
    if not train or not valid:
        raise ValueError("No complete pairs fit the fixed training sequence budget")
    # Validation inference retains every original validation example, including long ones.
    evaluation = [r for r in selected if r["split"] == "validation"]
    evaluation_labels = [{"id": r["id"], "gold_label": gold[r["id"]]} for r in evaluation]
    included_ids = {r["id"] for r in included}
    training_rows = [r for r in selected if r["split"] == "development" and r["id"] in included_ids]
    payloads = {"train.jsonl": train, "valid.jsonl": valid,
                "validation_inputs.jsonl": evaluation, "validation_labels.jsonl": evaluation_labels}
    # Exercise the longest surviving examples during the memory smoke check.
    for split, name in (("development", "train"), ("validation", "valid")):
        lengths_by_pair = defaultdict(int)
        for item in included:
            if item["split"] == split:
                pair = item["id"].split("-", 1)[1]
                lengths_by_pair[pair] = max(lengths_by_pair[pair], item["tokens"])
        longest = sorted(lengths_by_pair, key=lambda p: (-lengths_by_pair[p], p))[:2]
        payloads[f"smoke/{name}.jsonl"] = [
            {"messages": messages(r, gold[r["id"]])} for r in selected
            if r["split"] == split and r["id"].split("-", 1)[1] in longest]
    manifest = {"study": "mlx-lora-feasibility-v1", "model": identity,
                "source_manifest_sha256": digest(source_manifest), "seed": SEED,
                "max_sequence": MAX_SEQUENCE, "included": included, "excluded_pairs": excluded,
                "train_examples": len(train), "loss_validation_examples": len(valid),
                "validation_examples": len(evaluation),
                "train_labels": dict(Counter(gold[r["id"]] for r in training_rows)),
                "train_labels_by_task": {task: dict(Counter(gold[r["id"]] for r in training_rows
                                                           if r["task"] == task))
                                         for task in ("KCV", "VOOD")},
                "files_sha256": {name: digest(value) for name, value in payloads.items()},
                "policy": "Development-only training; observed validation for selection. No fresh-test claim."
                }
    path = output_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Prepared trial changed; preserve the old data and choose a new directory")
    for name, value in payloads.items():
        write_jsonl(output_dir / name, value)
    write_metrics(path, manifest)
    return manifest


def read_trial(data_dir: Path) -> tuple[list[dict], dict[str, str], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    payloads = {name: _read_jsonl(data_dir / name) for name in manifest["files_sha256"]}
    if any(digest(value) != manifest["files_sha256"][name] for name, value in payloads.items()):
        raise ValueError("Prepared training/validation file hash mismatch")
    rows, labels = payloads["validation_inputs.jsonl"], payloads["validation_labels.jsonl"]
    if (len({r["id"] for r in rows}) != len(rows)
            or len({r["id"] for r in labels}) != len(labels)
            or {r["id"] for r in rows} != {r["id"] for r in labels}):
        raise ValueError("Validation IDs are not aligned")
    return rows, {r["id"]: r["gold_label"] for r in labels}, manifest


def training_config(model_dir: Path, data_dir: Path, output_dir: Path, smoke: bool) -> dict:
    training_data = data_dir / "smoke" if smoke else data_dir
    return {"model": str(model_dir.resolve()), "data": str(training_data.resolve()),
            "adapter_path": str((output_dir / "adapter").resolve()), "train": True,
            "fine_tune_type": "lora", "seed": SEED, "num_layers": 4,
            "batch_size": 1, "grad_accumulation_steps": 1, "learning_rate": 1e-5,
            "iters": 8 if smoke else 100, "max_seq_length": MAX_SEQUENCE,
            "grad_checkpoint": True, "mask_prompt": True,
            "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0},
            "steps_per_report": 1 if smoke else 10, "steps_per_eval": 8 if smoke else 50,
            "save_every": 8 if smoke else 50, "val_batches": 4 if smoke else -1,
            "trust_remote_code": False, "clear_cache_threshold": 256 * 1024**2}


def runtime() -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Native Apple Silicon Python is required for this MLX trial")
    if (importlib.metadata.version("mlx-lm") != MLX_LM_VERSION
            or importlib.metadata.version("mlx") != "0.32.2"):
        raise ValueError(f"Use the isolated environment with mlx-lm=={MLX_LM_VERSION}, mlx==0.32.2")
    import mlx.core as mx
    if not mx.metal.is_available():
        raise ValueError("Metal GPU is unavailable")
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    if memory < 16 * 1024**3:
        raise ValueError("This fixed trial requires at least 16 GiB physical memory")
    mx.set_memory_limit(MEMORY_LIMIT)
    mx.set_cache_limit(256 * 1024**2)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "memory_bytes": memory, "mlx_memory_limit_bytes": MEMORY_LIMIT,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("mlx-lm", "mlx", "transformers", "huggingface-hub")}}


def model_files(path: Path) -> dict:
    files = sorted(p for p in path.iterdir() if p.suffix in {".json", ".txt", ".safetensors"})
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Model weights are missing")
    return {p.name: sha256_file(p) for p in files}


def train_adapter(data_dir: Path, output_dir: Path, smoke: bool) -> None:
    _, _, manifest = read_trial(data_dir)
    environment = runtime()
    model_dir = Path(manifest["model"]["path"])
    if model_files(model_dir) != manifest["model"]["files_sha256"]:
        raise ValueError("Base model files changed")
    config = training_config(model_dir, data_dir, output_dir, smoke)
    frozen = {"config": config, "data_sha256": digest(manifest), "environment": environment,
              "code_sha256": sha256_file(Path(__file__)), "smoke": smoke}
    path = output_dir / "training_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["run"] != frozen:
            raise ValueError("Training configuration/environment changed; use a new output directory")
        if previous["complete"]:
            if model_files(output_dir / "adapter") != previous["adapter_files_sha256"]:
                raise ValueError("Completed adapter files changed")
            return
        raise ValueError("Interrupted training preserved. Use a new output directory; optimizer resume is not implicit.")
    write_metrics(path, {"run": frozen, "complete": False})
    config_path = output_dir / "lora_config.json"
    write_metrics(config_path, config)  # JSON is accepted by MLX's native YAML config reader.
    child = ("import mlx.core as mx; "
             f"mx.set_memory_limit({MEMORY_LIMIT}); mx.set_cache_limit({256 * 1024**2}); "
             "from mlx_lm.lora import main; main()")
    command = [sys.executable, "-u", "-c", child, "--config", str(config_path),
               "--clear-cache-threshold", str(config["clear_cache_threshold"])]
    started = time.perf_counter()
    with (output_dir / "training.log").open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise ValueError(f"Training exited with status {code}; see {output_dir / 'training.log'}")
    if not (output_dir / "adapter/adapters.safetensors").is_file():
        raise ValueError("Training finished without adapter weights")
    write_metrics(path, {"run": frozen, "complete": True,
                         "training_wall_seconds": time.perf_counter() - started,
                         "adapter_files_sha256": model_files(output_dir / "adapter")})


def verdict(response: dict) -> str | None:
    text = response["text"].strip()
    return text if response["finish_reason"] == "stop" and text in {"T", "F", "X"} else None


def score_comparison(rows: list[dict], gold: dict[str, str], records: list[dict]) -> dict:
    index = {(r["id"], r["arm"]): r for r in records}
    if (len(index) != len(records)
            or set(index) != {(r["id"], a) for r in rows for a in ("base", "adapter")}):
        raise ValueError("Incomplete or duplicate matched predictions cannot receive complete scores")
    answers = {a: {r["id"]: verdict(index[r["id"], a]["response"]) for r in rows}
               for a in ("base", "adapter")}
    scores, paired = {}, {}
    for task in ("KCV", "VOOD"):
        selected = [r for r in rows if r["task"] == task]
        if not selected:
            raise ValueError("Both KCV and VOOD are required")
        scores[task] = {a: score_task(selected, gold, answers[a]) for a in answers}
        delta = [int(answers["adapter"][r["id"]] == gold[r["id"]])
                 - int(answers["base"][r["id"]] == gold[r["id"]]) for r in selected]
        paired[task] = {"accuracy_change": sum(delta) / len(delta), "fixes": delta.count(1),
                        "regressions": delta.count(-1), "ci95_cluster_bootstrap": paired_interval(selected, delta)}
    return {"by_task": scores, "paired": paired,
            "policy": "Observed validation only. A smoke run is not a research result; no new-test claim."}


def compare_adapter(data_dir: Path, output_dir: Path) -> dict:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    rows, gold, manifest = read_trial(data_dir)
    training = json.loads((output_dir / "training_manifest.json").read_text())
    environment = runtime()
    if (not training["complete"] or digest(manifest) != training["run"]["data_sha256"]
            or training["run"]["code_sha256"] != sha256_file(Path(__file__))
            or training["run"]["environment"] != environment):
        raise ValueError("Completed training/data/code/environment mismatch")
    if model_files(output_dir / "adapter") != training["adapter_files_sha256"]:
        raise ValueError("Adapter hash mismatch")
    model_dir = Path(manifest["model"]["path"])
    if model_files(model_dir) != manifest["model"]["files_sha256"]:
        raise ValueError("Base model hash mismatch")
    if training["run"]["smoke"]:
        pair_ids = sorted({r["id"].split("-", 1)[1] for r in rows})[:2]
        rows = [r for r in rows if r["id"].split("-", 1)[1] in pair_ids]
    configuration = {"training_sha256": digest(training), "ids": [r["id"] for r in rows],
                     "temperature": 0, "max_tokens": 16, "max_context": 32768}
    run_sha = digest(configuration)
    cache = output_dir / "responses.jsonl"
    records = _read_jsonl(cache) if cache.exists() else []
    indexed = {(r["id"], r["arm"]): r for r in records}
    expected = {(r["id"], a) for r in rows for a in ("base", "adapter")}
    by_id = {r["id"]: r for r in rows}
    if len(indexed) != len(records) or set(indexed) - expected:
        raise ValueError("Duplicate or unexpected comparison cache record")
    for record in records:
        if (record["run_sha256"] != run_sha
                or record["request"] != messages(by_id[record["id"]])
                or record["response_sha256"] != digest(record["response"])):
            raise ValueError("Comparison cache hash/request mismatch")
    write_metrics(output_dir / "comparison_manifest.json", {"config": configuration, "run_sha256": run_sha})
    for arm in ("base", "adapter"):
        if all((r["id"], arm) in indexed for r in rows):
            continue
        adapter_path = str(output_dir / "adapter") if arm == "adapter" else None
        model, tokenizer = load(str(model_dir), adapter_path=adapter_path, trust_remote_code=False)
        for row in rows:
            if (row["id"], arm) in indexed:
                continue
            request = messages(row)
            tokens = tokenizer.apply_chat_template(request, add_generation_prompt=True, return_dict=False)
            if len(tokens) + configuration["max_tokens"] > configuration["max_context"]:
                raise ValueError(f"{row['id']} exceeds context; do not truncate validation")
            started = time.perf_counter()
            text, last = "", None
            for chunk in stream_generate(model, tokenizer, prompt=tokens, max_tokens=16,
                                         sampler=make_sampler(temp=0)):
                text += chunk.text
                last = chunk
            if last is None:
                raise ValueError("MLX returned no generation record")
            response = {"text": text, "finish_reason": last.finish_reason,
                        "prompt_tokens": last.prompt_tokens, "generation_tokens": last.generation_tokens,
                        "peak_memory_gb": last.peak_memory}
            record = {"id": row["id"], "arm": arm, "request": request, "response": response,
                      "response_sha256": digest(response), "run_sha256": run_sha,
                      "wall_seconds": time.perf_counter() - started}
            with cache.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            indexed[row["id"], arm] = record
            records.append(record)
            print(f"Saved {len(records)}/{len(expected)} {row['id']} {arm}", flush=True)
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
    result = score_comparison(rows, gold, records)
    result["development_constant_controls"] = {}
    for task, counts in manifest["train_labels_by_task"].items():
        majority = min(counts, key=lambda label: (-counts[label], label))
        selected = [r for r in rows if r["task"] == task]
        result["development_constant_controls"][task] = {
            "label": majority, "accuracy": sum(gold[r["id"]] == majority for r in selected) / len(selected)}
    result.update(smoke=training["run"]["smoke"], saved_calls=len(records), complete=True,
                  run_sha256=run_sha,
                  wall_seconds_by_arm={a: sum(r["wall_seconds"] for r in records if r["arm"] == a)
                                       for a in ("base", "adapter")})
    write_metrics(output_dir / "metrics.json", result)
    lines = ["# MLX LoRA feasibility trial", "", result["policy"], "",
             f"Smoke: {result['smoke']}. Complete calls: {len(records)}.", "",
             "| Task | MLX base | MLX adapter | Development-majority control |", "|---|---:|---:|---:|"]
    for task, values in result["by_task"].items():
        control = result["development_constant_controls"][task]
        lines.append(f"| {task} | {values['base']['value']:.2%} | {values['adapter']['value']:.2%} | "
                     f"{control['accuracy']:.2%} (always {control['label']}) |")
    lines += ["", "One training seed. Training uses only development data.",
              "Different runtime/quantization from Ollama: use the matched MLX baseline.",
              "This does not establish improvement on all six tasks or on unseen CVEs."]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "quick", "run"))
    parser.add_argument("--source-data", type=Path, default=Path("data/full"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/lora_trial_v1"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        runtime()
        if args.mode == "prepare" or not (args.data_dir / "manifest.json").exists():
            _, _, source_manifest = read_prepared(args.source_data)
            frozen_source = Path(__file__).resolve().parents[2] / "data/manifests/full_benchmark.json"
            if source_manifest != json.loads(frozen_source.read_text()):
                raise ValueError("Source preparation differs from the frozen official benchmark manifest")
            from huggingface_hub import snapshot_download
            from mlx_lm.utils import load_tokenizer

            path = Path(snapshot_download(MODEL, revision=REVISION, token=False,
                                          allow_patterns=["*.json", "*.safetensors", "*.txt"]))
            identity = {"repo": MODEL, "revision": REVISION, "path": str(path.resolve()),
                        "files_sha256": model_files(path)}
            tokenizer = load_tokenizer(path, tokenizer_config_extra={"trust_remote_code": False})
            def count(ms: list[dict]) -> int:
                return len(tokenizer.apply_chat_template(ms, tools=None, return_dict=False))

            manifest = prepare_trial(args.source_data, args.data_dir, identity, count)
            print(json.dumps({k: v for k, v in manifest.items() if k not in {"included", "model"}}, indent=2))
        if args.mode != "prepare":
            output = args.output_dir or Path(f"outputs/lora_m3_v1_{args.mode}")
            train_adapter(args.data_dir, output, smoke=args.mode == "quick")
            print(json.dumps(compare_adapter(args.data_dir, output), indent=2))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
