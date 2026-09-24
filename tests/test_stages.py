from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from secure_rag.evaluation import evaluate
from secure_rag.stages import (
    apply_verification,
    prepare_reranked_examples,
    prepare_verification,
)


def save(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class StageTests(unittest.TestCase):
    def test_reranking_selects_model_relevant_passage_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "examples.jsonl"
            retrieval = root / "retrieval.jsonl"
            output = root / "e3.jsonl"
            cache = root / "reranking.jsonl"
            save(examples, [{"id": "kcv-1", "question": "Was X patched?", "prompt": "original"}])
            save(retrieval, [{
                "id": "kcv-1",
                "retrieved": [{"text": "generic description"}],
                "candidate_pool": [{"text": "generic description"}, {"text": "patched in version 2"}],
            }])
            with patch(
                "secure_rag.stages._request_prediction",
                return_value=("2", {"total_tokens": 6}),
            ) as request:
                count = prepare_reranked_examples(
                    examples, retrieval, output, cache, "http://localhost/v1", "local", 5, 7, 1
                )
                self.assertEqual(count, 1)
                request.assert_called_once()
            row = json.loads(output.read_text().splitlines()[0])
            self.assertIn("patched in version 2", row["prompt"])
            self.assertNotIn("generic description", row["prompt"])
            with patch("secure_rag.stages._request_prediction") as request:
                self.assertEqual(prepare_reranked_examples(
                    examples, retrieval, output, cache, "http://localhost/v1", "local", 5, 7, 1
                ), 0)
                request.assert_not_called()
            with self.assertRaisesRegex(ValueError, "cache"):
                prepare_reranked_examples(
                    examples, retrieval, output, cache, "http://localhost/v1", "changed", 5, 7, 1
                )
            with patch("secure_rag.stages._request_prediction", return_value=("1,1", {})):
                self.assertEqual(prepare_reranked_examples(
                    examples, retrieval, output, root / "bad.jsonl",
                    "http://localhost/v1", "local", 5, 7, 2
                ), 1)
            repaired = json.loads((root / "bad.jsonl").read_text().splitlines()[0])
            self.assertEqual(repaired["indices"], [1, 2])
            self.assertFalse(repaired["ranking_valid"])
            with patch("secure_rag.stages._request_prediction", return_value=("I cannot rank", {})):
                prepare_reranked_examples(
                    examples, retrieval, output, root / "unparsed.jsonl",
                    "http://localhost/v1", "local", 5, 7, 2
                )
            self.assertEqual(json.loads((root / "unparsed.jsonl").read_text())["indices"],
                             [1, 2])

    def test_verification_gate_abstains_on_disagreement_and_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples, generator, verifier = (root / name for name in (
                "examples.jsonl", "generator.jsonl", "verifier.jsonl"
            ))
            prepared, output = root / "verification.jsonl", root / "e4.jsonl"
            save(examples, [
                {"id": "kcv-1", "task": "KCV", "gold_label": "T", "question": "Statement A",
                 "prompt": "Use only the evidence below to evaluate the statement.\n\nEvidence:\n[1] fact\n\nStatement: Statement A\n\nReturn T if supported."},
                {"id": "kcv-2", "task": "KCV", "gold_label": "F", "question": "Statement B",
                 "prompt": "Use only the evidence below to evaluate the statement.\n\nEvidence:\n[1] fact\n\nStatement: Statement B\n\nReturn T if supported."},
                {"id": "vood-1", "task": "VOOD", "gold_label": "X", "question": "Statement C",
                 "prompt": "Use only the evidence below to evaluate the statement.\n\nEvidence:\nNo evidence was provided.\n\nStatement: Statement C\n\nReturn T if supported."},
            ])
            self.assertEqual(prepare_verification(examples, prepared), 3)
            prompts = [json.loads(line)["prompt"] for line in prepared.read_text().splitlines()]
            self.assertTrue(all("gold_label" not in prompt for prompt in prompts))
            self.assertIn("No evidence was provided", prompts[2])
            save(generator, [{"id": "kcv-1", "raw_output": "T"},
                             {"id": "kcv-2", "raw_output": "F"},
                             {"id": "vood-1", "raw_output": "T"}])
            save(verifier, [{"id": "kcv-1", "raw_output": "T"},
                            {"id": "kcv-2", "raw_output": "T"},
                            {"id": "vood-1", "raw_output": "X"}])
            self.assertEqual(apply_verification(examples, generator, verifier, output),
                             {"agreed": 1, "abstained": 2, "invalid_verifier": 0})
            self.assertEqual([json.loads(line)["raw_output"] for line in output.read_text().splitlines()],
                             ["T", "X", "X"])
            self.assertEqual(evaluate(examples, output)["overall"]["examples"], 3)
            save(verifier, [{"id": "kcv-1", "raw_output": "T"}])
            with self.assertRaisesRegex(ValueError, "do not match"):
                apply_verification(examples, generator, verifier, output)


if __name__ == "__main__":
    unittest.main()
