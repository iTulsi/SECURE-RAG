from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from secure_rag.runner import parse_reasoned_verdict, run_baseline


class RunnerTests(unittest.TestCase):
    def test_reasoned_verdict_is_strict_and_cache_is_isolated(self) -> None:
        self.assertEqual(parse_reasoned_verdict("EVIDENCE: score 5.3\nVERDICT: F"), "F")
        self.assertIsNone(parse_reasoned_verdict("VERDICT: F\nVERDICT: T because maybe"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "examples.jsonl"
            output = root / "predictions.jsonl"
            examples.write_text(json.dumps({"id": "kcv-1", "prompt": "Claim and facts"}) + "\n")
            with patch("secure_rag.runner._request_prediction",
                       return_value=("EVIDENCE: baseSeverity MEDIUM\nVERDICT: F", {})) as call:
                self.assertEqual(run_baseline(examples, output, "http://localhost/v1",
                                              "qwen", 1, 7, None, reasoned=True), (1, 1))
                self.assertEqual(call.call_args.kwargs["max_tokens"], 256)
            saved = json.loads(output.read_text())
            self.assertEqual(saved["raw_output"], "F")
            self.assertTrue(saved["verdict_valid"])
            self.assertIn("baseSeverity", saved["model_response"])
            with patch("secure_rag.runner._request_prediction") as call:
                self.assertEqual(run_baseline(examples, output, "http://localhost/v1",
                                              "qwen", 1, 7, None, reasoned=True), (0, 1))
                call.assert_not_called()
            with self.assertRaisesRegex(ValueError, "cache does not match"):
                run_baseline(examples, output, "http://localhost/v1", "qwen", 1, 7, None)
            with patch("secure_rag.runner._request_prediction",
                       return_value=("I cannot tell", {})):
                run_baseline(examples, root / "invalid.jsonl", "http://localhost/v1",
                             "qwen", 1, 7, None, reasoned=True)
            invalid = json.loads((root / "invalid.jsonl").read_text())
            self.assertFalse(invalid["verdict_valid"])
            self.assertEqual(invalid["raw_output"], "I cannot tell")

    def test_resumes_cached_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples_path = root / "examples.jsonl"
            output_path = root / "predictions.jsonl"
            examples_path.write_text(
                json.dumps({"id": "kcv-0000", "prompt": "first"})
                + "\n"
                + json.dumps({"id": "kcv-0001", "prompt": "second"})
                + "\n",
                encoding="utf-8",
            )
            output_path.write_text(
                json.dumps({"id": "kcv-0000", "raw_output": "T", "model": "test-model",
                            "prompt_sha256": hashlib.sha256(b"first").hexdigest(),
                            "request_parameters": {"temperature": 0, "max_tokens": 4, "seed": 7}}) + "\n",
                encoding="utf-8",
            )

            with patch(
                "secure_rag.runner._request_prediction", return_value=("F", {"total_tokens": 3})
            ) as request:
                new_count, total_count = run_baseline(
                    examples_path=examples_path,
                    output_path=output_path,
                    base_url="http://localhost:1234/v1",
                    model="test-model",
                    timeout_seconds=1,
                    seed=7,
                    limit=None,
                )

            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual((new_count, total_count), (1, 2))
            self.assertEqual([row["id"] for row in rows], ["kcv-0000", "kcv-0001"])
            self.assertEqual(rows[1]["raw_output"], "F")
            request.assert_called_once()

            examples_path.write_text(
                json.dumps({"id": "kcv-0000", "prompt": "changed"}) + "\n"
                + json.dumps({"id": "kcv-0001", "prompt": "second"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cache does not match"):
                run_baseline(examples_path, output_path, "http://localhost:1234/v1",
                             "test-model", 1, 7, None)

    def test_rejects_zero_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples_path = root / "examples.jsonl"
            examples_path.write_text(
                json.dumps({"id": "kcv-0000", "prompt": "first"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "at least 1"):
                run_baseline(
                    examples_path=examples_path,
                    output_path=root / "predictions.jsonl",
                    base_url="http://localhost:1234/v1",
                    model="test-model",
                    timeout_seconds=1,
                    seed=None,
                    limit=0,
                )


if __name__ == "__main__":
    unittest.main()
