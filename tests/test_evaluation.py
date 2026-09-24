from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from secure_rag.evaluation import evaluate, parse_label


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class EvaluationTests(unittest.TestCase):
    def test_parser_is_strict(self) -> None:
        self.assertEqual(parse_label(" f\n"), "F")
        self.assertIsNone(parse_label("False"))
        self.assertIsNone(parse_label(None))

    def test_scores_kcv_and_vood(self) -> None:
        examples = [
            {"id": "kcv-0000", "task": "KCV", "gold_label": "T"},
            {"id": "kcv-0001", "task": "KCV", "gold_label": "F"},
            {"id": "vood-0000", "task": "VOOD", "gold_label": "X"},
            {"id": "vood-0001", "task": "VOOD", "gold_label": "X"},
        ]
        predictions = [
            {"id": "kcv-0000", "raw_output": "T"},
            {"id": "kcv-0001", "raw_output": "explanation"},
            {"id": "vood-0000", "raw_output": "X"},
            {"id": "vood-0001", "raw_output": "F"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "examples.jsonl", examples)
            write_jsonl(root / "predictions.jsonl", predictions)

            metrics = evaluate(root / "examples.jsonl", root / "predictions.jsonl")

        self.assertEqual(metrics["overall"]["accuracy"], 0.5)
        self.assertEqual(metrics["kcv"]["invalid_predictions"], 1)
        self.assertEqual(metrics["abstention"]["precision"], 1.0)
        self.assertEqual(metrics["abstention"]["recall"], 0.5)

    def test_rejects_missing_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "examples.jsonl",
                [{"id": "kcv-0000", "task": "KCV", "gold_label": "T"}],
            )
            write_jsonl(root / "predictions.jsonl", [])

            with self.assertRaisesRegex(ValueError, "1 missing"):
                evaluate(root / "examples.jsonl", root / "predictions.jsonl")


if __name__ == "__main__":
    unittest.main()
