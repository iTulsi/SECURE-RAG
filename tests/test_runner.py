from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from secure_rag.runner import (
    BaselineError, VERDICT_FORMAT, _request_prediction, parse_reasoned_verdict,
    parse_structured_verdict, repair_reasoned_predictions, run_baseline,
)


class RunnerTests(unittest.TestCase):
    def test_structured_verdict_rejects_bad_shapes_and_extra_fields(self) -> None:
        valid = {"evidence": "UI:N", "comparison": "No interaction required.", "verdict": "T"}
        self.assertEqual(parse_structured_verdict(json.dumps(valid)), "T")
        for response in ("VERDICT: T", "{", "[]", json.dumps({**valid, "verdict": "true"}),
                         json.dumps({**valid, "verdict": ["T"]}),
                         json.dumps({**valid, "evidence": ""}),
                         json.dumps({**valid, "extra": "F"})):
            with self.subTest(response=response):
                self.assertIsNone(parse_structured_verdict(response))

    def test_request_sends_native_schema_without_new_dependencies(self) -> None:
        from io import BytesIO
        body = json.dumps({"choices": [{"message": {"content": "response"}}]}).encode()
        with patch("secure_rag.runner.urllib.request.urlopen", return_value=BytesIO(body)) as call:
            self.assertEqual(_request_prediction("http://localhost/v1", None, "qwen",
                                                 "facts", 1, 7, 512, VERDICT_FORMAT),
                             ("response", {}))
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload["response_format"], VERDICT_FORMAT)
        self.assertEqual(payload["max_tokens"], 512)

    def test_structured_run_checkpoints_and_resumes_after_invalid_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples, output = root / "examples.jsonl", root / "predictions.jsonl"
            examples.write_text("".join(json.dumps({"id": str(i), "prompt": f"fact {i}"})
                                        + "\n" for i in range(2)))
            response = json.dumps({"evidence": "XSS", "comparison": "Different mechanism.",
                                   "verdict": "F"})
            with patch("secure_rag.runner._request_prediction",
                       side_effect=[(response, {}), ("{", {})]):
                with self.assertRaisesRegex(BaselineError, "Invalid structured verdict"):
                    run_baseline(examples, output, "http://localhost/v1", "qwen", 1, 7,
                                 None, structured=True)
            self.assertEqual(len(output.read_text().splitlines()), 1)
            with patch("secure_rag.runner._request_prediction", return_value=(response, {})) as call:
                self.assertEqual(run_baseline(examples, output, "http://localhost/v1", "qwen",
                                              1, 7, None, structured=True), (1, 2))
                call.assert_called_once()
            saved = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(saved[0]["raw_output"], "F")
            self.assertEqual(saved[0]["model_response"], response)
            with self.assertRaisesRegex(ValueError, "cache does not match"):
                run_baseline(examples, output, "http://localhost/v1", "different", 1, 7,
                             None, structured=True)
            with self.assertRaisesRegex(ValueError, "cache does not match"):
                run_baseline(examples, output, "http://localhost/v1", "qwen", 1, 7, None)

    def test_verdict_accepts_markdown_and_explanation_without_guessing(self) -> None:
        for response in (
            "**VERDICT: F**", "VERDICT: F\n\nReason: contradicts the record.",
            "Reason: contradiction.\n**VERDICT:** **F**", "`VERDICT: F`",
            "VERDICT: F.\nMore explanation.", "F",
        ):
            with self.subTest(response=response):
                self.assertEqual(parse_reasoned_verdict(response), "F")
        for response in (
            "VERDICT: T\nVERDICT: F", "The statement is false.",
            "VERDICT: F because of evidence", "VERDICT: T\nVERDICT: uncertain",
            "EVIDENCE: long text cut off before a verdict",
        ):
            with self.subTest(response=response):
                self.assertIsNone(parse_reasoned_verdict(response))

    def test_repair_preserves_responses_and_does_not_call_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples, predictions, output = [root / name for name in (
                "examples.jsonl", "predictions.jsonl", "repaired.jsonl"
            )]
            examples.write_text(json.dumps({"id": "k1", "prompt": "evidence"}) + "\n")
            response = "**VERDICT: F**\nReason: contradiction."
            predictions.write_text(json.dumps({
                "id": "k1", "raw_output": response, "model_response": response,
                "reasoned_verdict": True, "verdict_valid": False,
            }) + "\n")
            original = predictions.read_bytes()
            with patch("secure_rag.runner._request_prediction") as request:
                counts = repair_reasoned_predictions(examples, predictions, output)
            request.assert_not_called()
            self.assertEqual(counts["valid_verdicts"], 1)
            repaired = json.loads(output.read_text())
            self.assertEqual(repaired["raw_output"], "F")
            self.assertEqual(repaired["model_response"], response)
            self.assertEqual(predictions.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "new output"):
                repair_reasoned_predictions(examples, predictions, predictions)
            examples.write_text(json.dumps({"id": "other", "prompt": "evidence"}) + "\n")
            with self.assertRaisesRegex(ValueError, "IDs do not match"):
                repair_reasoned_predictions(examples, predictions, output)

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
