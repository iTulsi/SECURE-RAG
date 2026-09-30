from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from secure_rag.kcv_facts import apply_cvss_facts, cvss_answer


METRIC = {"baseScore": 5.3, "baseSeverity": "MEDIUM", "attackComplexity": "LOW"}
CONTEXT = {"containers": {"cna": {"metrics": [{"cvssV3_1": METRIC}]}}}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CvssFactTests(unittest.TestCase):
    def test_vector_interaction_and_unsupported_negation_or_compound(self) -> None:
        context = {"containers": {"cna": {"metrics": [{"cvssV3_1": {
            **METRIC, "vectorString": "CVSS:3.1/AV:N/UI:N/S:U",
        }}]}}}
        claim = "The CVSS vector string indicates that user interaction is not required for the exploit."
        self.assertEqual(cvss_answer(claim, context), "T")
        self.assertEqual(cvss_answer(claim.replace("not ", ""), context), "F")
        context["containers"]["cna"]["metrics"][0]["cvssV3_1"]["userInteraction"] = "REQUIRED"
        self.assertIsNone(cvss_answer(claim, context))
        for claim in ("The base score is not 5.3.",
                      "The base score is 5.3 and it allows remote code execution.",
                      "The base score is 5.3, allowing remote code execution."):
            self.assertIsNone(cvss_answer(claim, CONTEXT))

    def test_direct_comparisons_and_compound_claim(self) -> None:
        self.assertEqual(cvss_answer(
            "The CVSS v3.1 base score is higher than 7.", CONTEXT
        ), "F")
        self.assertEqual(cvss_answer(
            "The base score is 5.3, indicating a medium severity vulnerability.", CONTEXT
        ), "T")
        self.assertEqual(cvss_answer(
            "The base score is 5.3, indicating a critical severity vulnerability.", CONTEXT
        ), "F")
        self.assertEqual(cvss_answer("The severity is categorized as High.", CONTEXT), "F")
        self.assertEqual(cvss_answer(
            "Attack complexity is rated as high in CVSS v3.1.", CONTEXT
        ), "F")

    def test_missing_or_conflicting_metrics_fall_back(self) -> None:
        self.assertIsNone(cvss_answer("The base score is 5.3.", {}))
        self.assertIsNone(cvss_answer("The base score is 5.3.", {
            "containers": {"cna": {"metrics": [
                {"cvssV3_1": METRIC},
                {"cvssV3_1": {**METRIC, "baseScore": 7.0}},
            ]}}
        }))
        self.assertIsNone(cvss_answer("The affected version is 5.3.", CONTEXT))

    def test_overrides_only_kcv_and_rejects_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples, predictions, output = (root / name for name in
                                             ("examples.jsonl", "predictions.jsonl", "output.jsonl"))
            prompt = (
                "You are given the following JSON data as context: "
                + json.dumps(CONTEXT)
                + "  Based on the context, you have to analyze the following statement: "
                + "The base score is higher than 7."
            )
            write_rows(examples, [
                {"id": "k1", "task": "KCV", "question": "The base score is higher than 7.",
                 "gold_label": "T", "prompt": prompt},
                {"id": "v1", "task": "VOOD", "question": "The base score is higher than 7.",
                 "gold_label": "X", "prompt": prompt},
            ])
            write_rows(predictions, [{"id": "k1", "raw_output": "X"},
                                     {"id": "v1", "raw_output": "X"}])
            self.assertEqual(apply_cvss_facts(examples, predictions, output), 1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]["raw_output"], "F")
            self.assertEqual(rows[0]["original_raw_output"], "X")
            self.assertEqual(rows[1], {"id": "v1", "raw_output": "X"})
            write_rows(predictions, [{"id": "k1", "raw_output": "X"}])
            with self.assertRaisesRegex(ValueError, "do not match"):
                apply_cvss_facts(examples, predictions, output)


if __name__ == "__main__":
    unittest.main()
