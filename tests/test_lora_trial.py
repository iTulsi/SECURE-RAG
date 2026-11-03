from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from secure_rag.benchmark_data import digest
from secure_rag.evaluation import _read_jsonl, write_metrics
from secure_rag.lora_trial import (
    MAX_SEQUENCE, compare_adapter, messages, model_files, prepare_trial, read_trial, score_comparison,
    train_adapter, training_config, verdict,
)
from secure_rag.pilot import write_jsonl
from secure_rag.dataset import sha256_file


def source_data(root: Path) -> Path:
    rows, labels = [], []
    for n, split in enumerate(("development", "validation", "test")):
        for task in ("KCV", "VOOD"):
            identifier = f"{task.lower()}-{n:04d}"
            rows.append({"id": identifier, "task": task, "group_id": f"g{n}", "split": split,
                         "prompt": f"Original {task} evidence for case {n}; return T/F/X"})
            labels.append({"id": identifier, "gold_label": "T" if task == "KCV" else "X"})
    source = root / "source"
    write_jsonl(source / "inputs.jsonl", rows)
    write_jsonl(source / "labels.jsonl", labels)
    write_metrics(source / "manifest.json", {"inputs_sha256": digest(rows), "labels_sha256": digest(labels)})
    return source


class LoraTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = source_data(self.root)
        self.output = self.root / "prepared"

    def prepare(self, count=lambda ms: 10):
        return prepare_trial(self.source, self.output, {"repo": "fixture"}, count)

    def test_uses_development_only_and_keeps_targets_out_of_inference(self):
        manifest = self.prepare()
        train = _read_jsonl(self.output / "train.jsonl")
        self.assertEqual(manifest["train_examples"], 2)
        self.assertEqual({r["messages"][-1]["content"] for r in train}, {"T", "X"})
        self.assertTrue(all("case 0" in r["messages"][1]["content"] for r in train))
        rows, gold, _ = read_trial(self.output)
        self.assertEqual(set(gold), {"kcv-0001", "vood-0001"})
        self.assertTrue(all("gold_label" not in r for r in rows))
        self.assertEqual([m["role"] for m in messages(rows[0])], ["system", "user"])
        self.assertEqual(messages(rows[0])[1]["content"], rows[0]["prompt"])

    def test_excludes_whole_training_pair_without_truncating_or_dropping_validation_inference(self):
        rows = _read_jsonl(self.source / "inputs.jsonl")
        labels = _read_jsonl(self.source / "labels.jsonl")
        for task in ("KCV", "VOOD"):
            rows.append({"id": f"{task.lower()}-0003", "task": task, "group_id": "g3",
                         "split": "development", "prompt": "oversized" if task == "KCV" else "short"})
            labels.append({"id": f"{task.lower()}-0003", "gold_label": "T" if task == "KCV" else "X"})
        for row in rows:
            if row["id"] == "kcv-0001":
                row["prompt"] = "oversized"
        write_jsonl(self.source / "inputs.jsonl", rows)
        write_jsonl(self.source / "labels.jsonl", labels)
        write_metrics(self.source / "manifest.json", {"inputs_sha256": digest(rows), "labels_sha256": digest(labels)})
        def count(ms):
            return MAX_SEQUENCE + 1 if ms[1]["content"] == "oversized" else 10

        # A second validation pair keeps the loss-validation set nonempty.
        rows.extend([{**r, "id": r["id"].replace("0001", "0004"), "group_id": "g4", "prompt": "short"}
                     for r in rows.copy() if r["split"] == "validation"])
        labels.extend([{**r, "id": r["id"].replace("0001", "0004")}
                       for r in labels.copy() if r["id"].endswith("0001")])
        write_jsonl(self.source / "inputs.jsonl", rows)
        write_jsonl(self.source / "labels.jsonl", labels)
        write_metrics(self.source / "manifest.json", {"inputs_sha256": digest(rows), "labels_sha256": digest(labels)})
        manifest = self.prepare(count)
        self.assertEqual({r["pair"] for r in manifest["excluded_pairs"]}, {"0001", "0003"})
        self.assertEqual(manifest["train_examples"], 2)
        self.assertEqual(len(read_trial(self.output)[0]), 4)
        self.assertEqual(manifest["loss_validation_examples"], 2)

    def test_changed_training_file_is_rejected(self):
        self.prepare()
        path = self.output / "train.jsonl"
        values = _read_jsonl(path)
        values[0]["messages"][-1]["content"] = "F"
        write_jsonl(path, values)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            read_trial(self.output)

    def test_group_overlap_and_incomplete_pairs_are_rejected(self):
        rows = _read_jsonl(self.source / "inputs.jsonl")
        labels = _read_jsonl(self.source / "labels.jsonl")
        for r in rows:
            r["group_id"] = "shared"
        write_jsonl(self.source / "inputs.jsonl", rows)
        write_metrics(self.source / "manifest.json", {"inputs_sha256": digest(rows), "labels_sha256": digest(labels)})
        with self.assertRaisesRegex(ValueError, "Source group crosses"):
            self.prepare()

    def test_unpaired_tasks_are_rejected(self):
        rows = _read_jsonl(self.source / "inputs.jsonl")[:-1]
        labels = _read_jsonl(self.source / "labels.jsonl")[:-1]
        rows = [r for r in rows if r["id"] != "vood-0000"]
        labels = [r for r in labels if r["id"] != "vood-0000"]
        write_jsonl(self.source / "inputs.jsonl", rows)
        write_jsonl(self.source / "labels.jsonl", labels)
        write_metrics(self.source / "manifest.json", {"inputs_sha256": digest(rows), "labels_sha256": digest(labels)})
        with self.assertRaisesRegex(ValueError, "Unaligned"):
            self.prepare()

    def test_rejects_changed_preparation_in_same_directory(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "Prepared trial changed"):
            self.prepare(lambda ms: 11)

    def test_strict_decision_does_not_salvage_cutoff_or_reasoning(self):
        self.assertEqual(verdict({"text": " T\n", "finish_reason": "stop"}), "T")
        for text, reason in (("T", "length"), ("Answer: T", "stop"), ("t", "stop")):
            self.assertIsNone(verdict({"text": text, "finish_reason": reason}))

    def test_complete_scoring_preserves_wrong_model_answers_and_separates_tasks(self):
        self.prepare()
        rows, gold, _ = read_trial(self.output)
        records = [{"id": r["id"], "arm": a,
                    "response": {"text": "T" if a == "adapter" else "X", "finish_reason": "stop"}}
                   for r in rows for a in ("base", "adapter")]
        result = score_comparison(rows, gold, records)
        self.assertEqual(result["paired"]["KCV"]["accuracy_change"], 1)
        self.assertEqual(result["paired"]["VOOD"]["accuracy_change"], -1)
        self.assertEqual(result["by_task"]["VOOD"]["adapter"]["value"], 0)
        for invalid in (records[:-1], records + [records[0]]):
            with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
                score_comparison(rows, gold, invalid)

    def test_config_masks_prompt_and_smoke_does_not_reuse_full_adapter(self):
        short = training_config(self.root / "model", self.output, self.root / "smoke", True)
        full = training_config(self.root / "model", self.output, self.root / "full", False)
        self.assertEqual((short["iters"], full["iters"]), (8, 100))
        self.assertNotEqual(short["adapter_path"], full["adapter_path"])
        self.assertTrue(full["mask_prompt"] and full["grad_checkpoint"])
        self.assertEqual(full["batch_size"], 1)
        self.assertEqual(Path(short["data"]), self.output / "smoke")
        self.assertEqual(Path(full["data"]), self.output)

    def test_interrupted_training_cannot_silently_restart(self):
        self.prepare()
        run = self.root / "run"
        run.mkdir()
        # A mismatching existing run must fail before spawning any trainer.
        write_metrics(run / "training_manifest.json", {"run": {}, "complete": False})
        manifest = json.loads((self.output / "manifest.json").read_text())
        manifest["model"]["path"] = str(self.root / "model")
        manifest["model"]["files_sha256"] = {}
        write_metrics(self.output / "manifest.json", manifest)
        with patch("secure_rag.lora_trial.runtime", return_value={}), \
             patch("secure_rag.lora_trial.model_files", return_value={}), \
             patch("secure_rag.lora_trial.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "configuration/environment changed"):
                train_adapter(self.output, run, False)
            spawn.assert_not_called()

    def test_native_comparison_checkpoints_resumes_and_rejects_tampering(self):
        from secure_rag import lora_trial

        self.prepare()
        model_dir = self.root / "model"
        adapter_dir = self.root / "run/adapter"
        for p in (model_dir, adapter_dir):
            p.mkdir(parents=True)
            (p / "adapters.safetensors").write_bytes(b"fixture, not actual model weights")
        manifest = json.loads((self.output / "manifest.json").read_text())
        manifest["model"].update(path=str(model_dir), files_sha256=model_files(model_dir))
        write_metrics(self.output / "manifest.json", manifest)
        run = adapter_dir.parent
        write_metrics(run / "training_manifest.json", {
            "complete": True, "adapter_files_sha256": model_files(adapter_dir),
            "run": {"data_sha256": digest(manifest), "environment": {}, "smoke": False,
                    "code_sha256": sha256_file(Path(lora_trial.__file__))}})

        class Tokenizer:
            def apply_chat_template(self, request, **kwargs):
                self_test.assertTrue(all(m["role"] != "assistant" for m in request))
                return [10, 20]

        self_test = self
        mlx = types.ModuleType("mlx")
        core = types.ModuleType("mlx.core")
        core.clear_cache = lambda: None
        mlx.core = core
        library = types.ModuleType("mlx_lm")
        calls = []

        def load(path, adapter_path=None, **kwargs):
            calls.append(adapter_path)
            return ("T" if adapter_path else "X"), Tokenizer()

        def generate(model, tokenizer, **kwargs):
            yield types.SimpleNamespace(text=model, finish_reason="stop", prompt_tokens=2,
                                        generation_tokens=1, peak_memory=0.01)

        library.load, library.stream_generate = load, generate
        sampling = types.ModuleType("mlx_lm.sample_utils")
        sampling.make_sampler = lambda **kwargs: None
        modules = {"mlx": mlx, "mlx.core": core, "mlx_lm": library, "mlx_lm.sample_utils": sampling}
        with patch.dict(sys.modules, modules), patch("secure_rag.lora_trial.runtime", return_value={}):
            first = compare_adapter(self.output, run)
            self.assertEqual(first["saved_calls"], 4)
            self.assertEqual(first["paired"]["VOOD"]["regressions"], 1)
            self.assertEqual(compare_adapter(self.output, run), first)
            self.assertEqual(len(calls), 2)  # Complete caches cause no repeated inference.
            records = _read_jsonl(run / "responses.jsonl")
            records[0]["response"]["text"] = "F"
            write_jsonl(run / "responses.jsonl", records)
            with self.assertRaisesRegex(ValueError, "cache hash/request mismatch"):
                compare_adapter(self.output, run)


if __name__ == "__main__":
    unittest.main()
