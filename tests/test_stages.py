from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from secure_rag.evaluation import evaluate
from secure_rag.stages import (
    apply_verification,
    prepare_claim_evidence,
    prepare_context_prompts,
    prepare_reranked_examples,
    prepare_verification,
    route_by_context,
)
from secure_rag.retrieval import _retrieval_prompt


def save(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class StageTests(unittest.TestCase):
    def test_claim_evidence_keeps_facts_and_missing_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples, output = root / "examples.jsonl", root / "reasoned.jsonl"
            context = {"containers": {"cna": {
                "descriptions": [{"value": "Remote input issue", "supportingMedia": [
                    {"value": "<p>Remote input issue</p>"}]}],
                "metrics": [{"cvssV3_1": {"baseSeverity": "MEDIUM"}}],
                "solutions": [{"value": "Upgrade the firmware"}],
                "workarounds": [{"value": "Disable feature", "supportingMedia": [
                    {"value": "Additional patch detail"}]}],
                "references": [{"url": "https://example.invalid"}],
            }}}
            prompt = ("You are given the following JSON data as context: "
                      + json.dumps(context)
                      + "  Based on the context, you have to analyze the following statement: "
                      + "The severity is medium.")
            save(examples, [
                {"id": "kcv-1", "task": "KCV", "question": "The severity is medium.",
                 "gold_label": "F", "prompt": prompt},
                {"id": "vood-1", "task": "VOOD", "question": "Unknown?",
                 "gold_label": "X", "prompt": "No record"},
            ])
            self.assertEqual(prepare_claim_evidence(examples, output),
                             {"context_present": 1, "context_missing": 1})
            prepared = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertIn("MEDIUM", prepared[0]["prompt"])
            self.assertIn("Upgrade the firmware", prepared[0]["prompt"])
            self.assertIn("Additional patch detail", prepared[0]["prompt"])
            self.assertNotIn("<p>Remote input issue</p>", prepared[0]["prompt"])
            self.assertNotIn("example.invalid", prepared[0]["prompt"])
            self.assertIn("There is no CVE record", prepared[1]["prompt"])
            self.assertNotIn("VERDICT: X\n", prepared[1]["prompt"])
            with self.assertRaisesRegex(ValueError, "new output path"):
                prepare_claim_evidence(examples, examples)

    def test_context_router_uses_prompts_without_consulting_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "original.jsonl"
            dense_examples = root / "dense.jsonl"
            original = root / "original_predictions.jsonl"
            dense = root / "dense_predictions.jsonl"
            routed_examples = root / "routed_examples.jsonl"
            routed_predictions = root / "routed_predictions.jsonl"
            kcv_prompt = (
                'You are given the following JSON data as context: {"affected":true}'
                '  Based on the context, you have to analyze the following statement: Affected?'
            )
            save(examples, [
                {"id": "kcv-1", "task": "KCV", "question": "Affected?",
                 "gold_label": "T", "prompt": kcv_prompt},
                {"id": "vood-1", "task": "VOOD", "question": "Affected?",
                 "gold_label": "X", "prompt": "No CVE context. Affected?"},
            ])
            save(dense_examples, [
                {"id": "kcv-1", "task": "KCV", "question": "Affected?",
                 "gold_label": "T", "prompt": "Different retrieval prompt"},
                {"id": "vood-1", "task": "VOOD", "question": "Affected?",
                 "gold_label": "X", "prompt": _retrieval_prompt("Affected?", [])},
            ])
            save(original, [{"id": "kcv-1", "raw_output": "T", "model": "qwen"},
                            {"id": "vood-1", "raw_output": "F", "model": "qwen"}])
            save(dense, [{"id": "kcv-1", "raw_output": "X", "model": "qwen"},
                         {"id": "vood-1", "raw_output": "X", "model": "qwen"}])
            self.assertEqual(route_by_context(
                examples, dense_examples, original, dense,
                routed_examples, routed_predictions,
            ), {"context_present": 1, "context_missing": 1})
            routed = [json.loads(line) for line in routed_predictions.read_text().splitlines()]
            self.assertEqual([(row["selected_run"], row["raw_output"]) for row in routed],
                             [("E0", "T"), ("E1", "X")])
            self.assertEqual(evaluate(routed_examples, routed_predictions)["overall"]["accuracy"], 1)
            fresh_prompts = root / "fresh_prompts.jsonl"
            self.assertEqual(prepare_context_prompts(examples, fresh_prompts),
                             {"context_present": 1, "context_missing": 1})
            self.assertEqual(fresh_prompts.read_text(), routed_examples.read_text())

            save(dense_examples, [
                {"id": "kcv-1", "task": "KCV", "question": "Affected?",
                 "gold_label": "T", "prompt": "Different retrieval prompt"},
                {"id": "vood-1", "task": "VOOD", "question": "Affected?",
                 "gold_label": "X", "prompt": "Unexpected hint"},
            ])
            with self.assertRaisesRegex(ValueError, "prompt changed"):
                route_by_context(examples, dense_examples, original, dense,
                                 routed_examples, routed_predictions)

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
