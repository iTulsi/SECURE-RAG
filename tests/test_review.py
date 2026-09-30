from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from secure_rag.cli import main
from secure_rag.corpus import build_corpus
from secure_rag.pilot import write_jsonl
from secure_rag.runner import BaselineError


class ReviewTests(unittest.TestCase):
    def test_decision_review_cli_scores_complete_outputs_and_resumes(self):
        argv = ["run-kcv-review", "--examples", str(self.examples),
                "--predictions", str(self.baseline), "--output-dir", str(self.output)]
        response = json.dumps({"evidence": "remote widget attack",
                               "comparison": "Claim matches the record.", "verdict": "T"})
        original = self.baseline.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "secure_rag.runner._request_prediction", return_value=(response, {})
        ) as request:
            self.assertEqual(main(argv), 0)
        request.assert_called_once()
        metrics = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(metrics["overall"]["accuracy"], 1)
        self.assertEqual(metrics["overall"]["examples"], 2)
        self.assertEqual(self.baseline.read_bytes(), original)
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["counts"]["model_review"], 1)
        self.assertIn("Provided prior predictions", (self.output / "RESULTS.md").read_text())
        self.assertTrue((self.output / "comparison.json").exists())
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "secure_rag.runner._request_prediction"
        ) as request:
            self.assertEqual(main(argv), 0)
        request.assert_not_called()
        changed = [*argv, "--model", "different"]
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "secure_rag.runner._request_prediction"
        ) as request:
            with self.assertRaisesRegex(ValueError, "cache does not match"):
                main(changed)
        request.assert_not_called()

    def test_decision_review_cli_handles_zero_requests_and_rejects_wrong_ids(self):
        argv = ["run-kcv-review", "--examples", str(self.examples),
                "--predictions", str(self.baseline), "--output-dir", str(self.output)]
        write_jsonl(self.baseline, [{"id": "kcv-1", "raw_output": "T"},
                                   {"id": "vood-1", "raw_output": "F"}])
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "secure_rag.runner._request_prediction"
        ) as request:
            self.assertEqual(main(argv), 0)
        request.assert_not_called()
        self.assertEqual((self.output / "review_predictions.jsonl").read_text(), "")
        metrics = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(metrics["overall"]["accuracy"], 1)
        write_jsonl(self.baseline, [{"id": "wrong", "raw_output": "X"}])
        with patch("secure_rag.runner._request_prediction") as request:
            with self.assertRaisesRegex(ValueError, "IDs do not match"):
                main(argv)
        request.assert_not_called()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        cve = self.root / "cve.json"
        cve.write_text(json.dumps({
            "cveMetadata": {"cveId": "CVE-2024-0007", "state": "PUBLISHED"},
            "containers": {"cna": {"descriptions": [
                {"value": "remote widget attack", "lang": "en"}
            ]}},
        }))
        self.corpus = self.root / "corpus.jsonl"
        build_corpus([("cve", cve)], self.corpus)
        self.examples = self.root / "examples.jsonl"
        write_jsonl(self.examples, [
            {
                "id": "kcv-1", "task": "KCV", "gold_label": "T",
                "question": "remote widget attack", "source_url": "",
                "prompt": 'You are given the following JSON data as context: '
                          '{"id":"CVE-2024-0007","description":"remote widget attack"}'
                          '  Based on the context, you have to analyze the following statement: x',
            },
            {
                "id": "vood-1", "task": "VOOD", "gold_label": "X",
                "question": "remote widget attack", "source_url": "",
                "prompt": "No supplied context",
            },
        ])
        self.baseline = self.root / "baseline.jsonl"
        write_jsonl(self.baseline, [
            {"id": "kcv-1", "raw_output": "X"},
            {"id": "vood-1", "raw_output": "X"},
        ])
        self.output = self.root / "run"
        self.argv = [
            "run-review", "--examples", str(self.examples),
            "--corpus", str(self.corpus), "--output-dir", str(self.output),
            "--baseline-predictions", str(self.baseline),
        ]

    def test_failure_keeps_checkpoint_and_resume_writes_complete_results(self):
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "secure_rag.runner._request_prediction",
            side_effect=[("T", {}), BaselineError("offline")],
        ):
            with self.assertRaisesRegex(SystemExit, "Inference incomplete"):
                main(self.argv)
        self.assertEqual(len((self.output / "predictions.jsonl").read_text().splitlines()), 1)
        self.assertFalse((self.output / "metrics.json").exists())
        self.assertFalse((self.output / "RESULTS.md").exists())
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "secure_rag.runner._request_prediction", return_value=("X", {})
        ) as request:
            self.assertEqual(main(self.argv), 0)
        request.assert_called_once()
        metrics = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(metrics["overall"]["examples"], 2)
        self.assertEqual(metrics["overall"]["invalid_predictions"], 0)
        self.assertTrue((self.output / "comparison.json").exists())
        self.assertTrue((self.output / "run_manifest.json").exists())
        self.assertIn("Provided baseline predictions", (self.output / "RESULTS.md").read_text())
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "secure_rag.runner._request_prediction"
        ) as request:
            self.assertEqual(main(self.argv), 0)
        request.assert_not_called()

    def test_rejects_mismatched_baseline_before_inference(self):
        write_jsonl(self.baseline, [{"id": "wrong", "raw_output": "X"}])
        with patch("secure_rag.runner._request_prediction") as request:
            with self.assertRaisesRegex(ValueError, "IDs do not match"):
                main(self.argv)
        request.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_rejects_input_overwrite(self):
        self.argv[self.argv.index("--output-dir") + 1] = str(self.root)
        original = self.examples.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite inputs"):
            main(self.argv)
        self.assertEqual(self.examples.read_bytes(), original)
