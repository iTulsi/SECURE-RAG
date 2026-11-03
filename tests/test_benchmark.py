from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from secure_rag.benchmark import (
    ARMS, evaluate_experiment, inference_request, model_identity, run_experiment,
)
from secure_rag.benchmark_data import (
    FIELDS, TASKS, assign_groups, load_benchmark, prepare_benchmark, read_prepared,
)
from secure_rag.benchmark_metrics import (
    paired_interval, parse_answer, rouge_l_f1, score_suite, score_task,
)
from secure_rag.dataset import DatasetError
from secure_rag.evaluation import _read_jsonl, write_metrics
from secure_rag.runner import BaselineError


def fixture_data(root: Path) -> None:
    root.mkdir(parents=True)
    for task in TASKS:
        with (root / f"SECURE - {task}.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS[task], delimiter="\t")
            writer.writeheader()
            for i in range(2):
                row = {key: "text" for key in FIELDS[task]}
                row["Prompt"] = f"{task} problem {i}"
                row["Correct Answer"] = "A" if task in {"MAET", "CWET"} else (
                    "7.8" if task == "CPST" else "risk sentence" if task == "RERT" else
                    "X" if task == "VOOD" else "T")
                for field in ("URL", "source-url"):
                    if field in row:
                        row[field] = f"https://example.com/{task if task not in {'KCV','VOOD'} else 'CVE'}/{i}"
                if "Question" in row:
                    row["Question"] = f"Question {i}"
                if "CVSS v3 Vector String" in row:
                    row["CVSS v3 Vector String"] = f"vector-{i}"
                writer.writerow(row)
            if task == "CWET":
                writer.writerow({key: "" for key in FIELDS[task]})


def prepared_fixture(root: Path) -> Path:
    raw, prepared = root / "raw", root / "prepared"
    fixture_data(raw)
    _, audit = load_benchmark(raw)
    hashes = {task: a["sha256"] for task, a in audit.items()}
    historical = {"dataset_sha256": {t: hashes[t] for t in ("KCV", "VOOD")},
                  "selected_sources": []}
    pilot, holdout = root / "pilot.json", root / "holdout.json"
    write_metrics(pilot, historical)
    write_metrics(holdout, historical)
    with patch("secure_rag.benchmark_data.OFFICIAL_SHA256", hashes):
        prepare_benchmark(raw, prepared, pilot, holdout)
    return prepared


IDENTITY = {"model": "fake:test", "digest": "fake-digest", "details": {},
            "show": {}, "ollama_version": "fixture"}


def fake_response(*args) -> dict:
    payload = args[3]
    text = json.dumps({"analysis": "Checked the problem.", "answer": "T"}) if "format" in payload else "T"
    return {"message": {"content": text}, "done": True, "done_reason": "stop",
            "prompt_eval_count": 20, "eval_count": 10}


class BenchmarkDataTests(unittest.TestCase):
    def test_full_loader_preserves_labels_and_quarantines_only_blank_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_data(root / "raw")
            rows, audit = load_benchmark(root / "raw")
            self.assertEqual(len(rows), 12)
            self.assertEqual(audit["CWET"]["blank_row_indices"], [2])
            self.assertEqual(next(r for r in rows if r["task"] == "CPST")["gold_label"], "7.8")
            path = root / "raw" / "SECURE - MAET.tsv"
            path.write_text(path.read_text().replace("\tA\n", "\t\n", 1))
            with self.assertRaisesRegex(DatasetError, "Missing prompt/label"):
                load_benchmark(root / "raw")

    def test_groups_join_cross_task_cves_duplicates_and_normalized_urls(self):
        rows = [
            {"id": "a", "task": "KCV", "source_url": "http://example.com/one/", "prompt": "CVE-2024-0001", "question": "q"},
            {"id": "b", "task": "VOOD", "source_url": "https://example.com/one", "prompt": "claim", "question": "q"},
            {"id": "c", "task": "RERT", "source_url": "https://cisa.gov/advisory", "prompt": "CVE-2024-0001", "question": ""},
            {"id": "d", "task": "CWET", "source_url": "https://example.com/other", "prompt": "same prompt", "question": "q"},
            {"id": "e", "task": "CWET", "source_url": "https://example.com/another", "prompt": "same prompt", "question": "q"},
        ]
        groups = assign_groups(rows)
        self.assertEqual(groups["a"], groups["b"])
        self.assertEqual(groups["a"], groups["c"])
        self.assertEqual(groups["d"], groups["e"])
        self.assertNotEqual(groups["a"], groups["d"])

    def test_prepared_inputs_exclude_gold_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = prepared_fixture(root)
            rows, labels, manifest = read_prepared(prepared)
            self.assertEqual(len(rows), len(labels))
            self.assertTrue(all("gold_label" not in row for row in rows))
            splits = {}
            for row in rows:
                splits.setdefault(row["group_id"], set()).add(row["split"])
            self.assertTrue(all(len(s) == 1 for s in splits.values()))
            self.assertEqual(manifest["scorable_examples"], 12)
            path = prepared / "inputs.jsonl"
            path.write_text(path.read_text().replace("problem", "changed", 1))
            with self.assertRaisesRegex(ValueError, "hashes"):
                read_prepared(prepared)

    def test_unknown_official_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_data(root / "raw")
            with self.assertRaisesRegex(DatasetError, "pinned official"):
                prepare_benchmark(root / "raw", root / "out", root / "x", root / "y")

    def test_prior_pilot_sources_are_forced_to_development(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = prepared_fixture(root)
            hashes = {t: v["sha256"] for t, v in json.loads((prepared / "manifest.json").read_text())["audit"].items()}
            pilot = json.loads((root / "pilot.json").read_text())
            pilot["selected_sources"] = ["https://example.com/CVE/0"]
            write_metrics(root / "pilot.json", pilot)
            with patch("secure_rag.benchmark_data.OFFICIAL_SHA256", hashes):
                prepare_benchmark(root / "raw", root / "new", root / "pilot.json", root / "holdout.json")
            rows, _, _ = read_prepared(root / "new")
            self.assertTrue(all(r["split"] == "development" for r in rows if r["id"] in {"kcv-0000", "vood-0000"}))


class BenchmarkMetricTests(unittest.TestCase):
    def test_strict_parsing_rejects_guesses_nonfinite_scores_and_extra_json(self):
        self.assertEqual(parse_answer("KCV", "T", False), "T")
        self.assertEqual(parse_answer("CPST", "7.8", False), 7.8)
        for value in ("nan", "inf", "-1", "11", "7.8 because", ""):
            self.assertIsNone(parse_answer("CPST", value, False))
        self.assertIsNone(parse_answer("KCV", "I think T", False))
        self.assertIsNone(parse_answer("KCV", '{"analysis":"ok","answer":"T","gold":"T"}', True))

    def test_rouge_l_known_cases(self):
        self.assertEqual(rouge_l_f1("Read files!", "read files"), 1.)
        self.assertEqual(rouge_l_f1("", "read files"), 0.)
        self.assertEqual(rouge_l_f1("read files", "destroy systems"), 0.)
        self.assertAlmostEqual(rouge_l_f1("a b c", "a c"), .8)

    def test_invalid_cvss_does_not_disappear_from_denominator(self):
        rows = [{"id": "a", "task": "CPST", "group_id": "g", "prompt": "one"},
                {"id": "b", "task": "CPST", "group_id": "h", "prompt": "two"}]
        result = score_task(rows, {"a": "7.8", "b": "5.0"}, {"a": 7.8, "b": None})
        self.assertEqual(result["value"], 5.)
        self.assertEqual(result["mad_valid"], 0.)
        self.assertEqual(result["valid_coverage"], .5)

    def test_paired_cluster_bootstrap_keeps_constant_differences(self):
        rows = [{"group_id": "a"}, {"group_id": "a"}, {"group_id": "b"}]
        self.assertEqual(paired_interval(rows, [.25, .25, .25]), [.25, .25])
        self.assertIsNone(paired_interval(rows[:2], [.25, .25]))

    def test_missing_predictions_rejected_and_truncated_outputs_count_invalid(self):
        rows = [{"id": "a", "task": "KCV", "group_id": "g", "prompt": "one"}]
        with self.assertRaisesRegex(ValueError, "Missing"):
            score_suite(rows, {"a": "T"}, [], ("official",))
        record = {"id": "a", "arm": "official", "response": {
            "message": {"content": "T"}, "done_reason": "length"}}
        result = score_suite(rows, {"a": "T"}, [record], ("official",))
        self.assertEqual(result["by_arm"]["official"]["KCV"]["invalid_predictions"], 1)
        self.assertIsNone(result["pooled_score"])


class BenchmarkRunnerTests(unittest.TestCase):
    def test_inference_uses_original_problem_without_gold_or_answer_rules(self):
        row = {"id": "a", "prompt": "No record supplied", "gold_label": "SECRET_GOLD"}
        config = {"temperature": .7, "seed": 2, "max_tokens": 512, "context_window": 32768}
        for arm in ("official", "evidence_first"):
            payload = inference_request(row, arm, None, "model", config)
            self.assertNotIn("SECRET_GOLD", json.dumps(payload))
            self.assertEqual(payload["options"]["temperature"], .7)
        self.assertEqual(inference_request(row, "official", None, "model", config)["messages"][0]["content"], row["prompt"])
        with self.assertRaisesRegex(ValueError, "Self-review needs"):
            inference_request(row, "self_review", None, "model", config)
        with self.assertRaisesRegex(ValueError, "context budget"):
            inference_request({**row, "prompt": "x" * 33000}, "official", None, "model", config)

    @patch("secure_rag.benchmark.model_identity", return_value=IDENTITY)
    @patch("secure_rag.benchmark.request_json", side_effect=fake_response)
    def test_model_only_run_checkpoints_resumes_and_scores_complete_dataset(self, request, identity):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            prepared, out = prepared_fixture(root), root / "run"
            partial = run_experiment(prepared, out, "http://localhost", "fake:test", split="full", max_new_examples=1)
            self.assertFalse(partial["complete"])
            self.assertEqual(request.call_count, 3)
            self.assertFalse((out / "metrics.json").exists())
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                evaluate_experiment(prepared, out)
            result = run_experiment(prepared, out, "http://localhost", "fake:test", split="full")
            self.assertTrue(result["complete"])
            self.assertEqual(request.call_count, 36)
            metrics = json.loads((out / "metrics.json").read_text())
            self.assertEqual(set(metrics["by_arm"]), set(ARMS))
            self.assertEqual(len(metrics["by_arm"]["official"]), 6)
            self.assertEqual(metrics["self_review_total_calls"], 24)
            self.assertTrue((out / "RESULTS.md").exists())
            run_experiment(prepared, out, "http://localhost", "fake:test", split="full")
            self.assertEqual(request.call_count, 36)
            with self.assertRaisesRegex(ValueError, "Frozen experiment"):
                run_experiment(prepared, out, "http://localhost", "fake:test", split="full", temperature=0)

    @patch("secure_rag.benchmark.model_identity", return_value=IDENTITY)
    @patch("secure_rag.benchmark.request_json", side_effect=fake_response)
    def test_tampered_cache_is_rejected_before_any_additional_inference(self, request, identity):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            prepared, out = prepared_fixture(root), root / "run"
            run_experiment(prepared, out, "http://localhost", "fake:test", split="full", max_new_examples=1)
            path = out / "responses.jsonl"
            rows = _read_jsonl(path)
            rows[0]["response"]["message"]["content"] = "F"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "Cache response"):
                run_experiment(prepared, out, "http://localhost", "fake:test", split="full")
            self.assertEqual(request.call_count, 3)

    @patch("secure_rag.benchmark.model_identity", return_value=IDENTITY)
    def test_network_failure_preserves_prior_complete_calls(self, identity):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            prepared, out = prepared_fixture(root), root / "run"
            with patch("secure_rag.benchmark.request_json", side_effect=[
                {"message": {"content": "T"}, "done": True, "done_reason": "stop"},
                BaselineError("offline"),
            ]):
                with self.assertRaisesRegex(BaselineError, "offline"):
                    run_experiment(prepared, out, "http://localhost", "fake:test", split="full")
            self.assertEqual(len(_read_jsonl(out / "responses.jsonl")), 1)

    def test_model_preflight_requires_installed_digest(self):
        with patch("secure_rag.benchmark.request_json", return_value={"models": []}):
            with self.assertRaisesRegex(BaselineError, "not installed"):
                model_identity("http://localhost", "missing", 1)


if __name__ == "__main__":
    unittest.main()
