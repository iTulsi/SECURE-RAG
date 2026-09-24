from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from secure_rag.comparison import compare_runs


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class ComparisonTests(unittest.TestCase):
    def test_reports_paired_improvements_and_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = [
                {"id": "kcv-1", "task": "KCV", "gold_label": "T", "question": "q1"},
                {"id": "kcv-2", "task": "KCV", "gold_label": "F", "question": "q2"},
                {"id": "vood-1", "task": "VOOD", "gold_label": "X", "question": "q3"},
            ]
            baseline = [
                {"id": "kcv-1", "raw_output": "T"},
                {"id": "kcv-2", "raw_output": "X"},
                {"id": "vood-1", "raw_output": "F"},
            ]
            candidate = [
                {"id": "kcv-1", "raw_output": "X"},
                {"id": "kcv-2", "raw_output": "F"},
                {"id": "vood-1", "raw_output": "X"},
            ]
            retrieval = [
                {"id": "kcv-1", "retrieved": [{"score": 0.8}]},
                {"id": "kcv-2", "retrieved": [{"score": 0.4}]},
                {"id": "vood-1", "retrieved": []},
            ]
            for name, rows in (
                ("examples.jsonl", examples),
                ("baseline.jsonl", baseline),
                ("candidate.jsonl", candidate),
                ("retrieval.jsonl", retrieval),
            ):
                write_jsonl(root / name, rows)

            result = compare_runs(
                root / "examples.jsonl",
                root / "baseline.jsonl",
                root / "candidate.jsonl",
                root / "retrieval.jsonl",
            )

            kcv = result["tasks"]["kcv"]
            self.assertEqual(kcv["correctness_transitions"], {
                "correct->wrong": 1,
                "wrong->correct": 1,
            })
            self.assertEqual(len(result["changed_correctness"]), 3)
            self.assertEqual(
                result["kcv_top_retrieval_score_by_candidate_outcome"]["correct"]["mean"],
                0.4,
            )
            self.assertEqual(
                result["tasks"]["vood"]["baseline"]["answered_accuracy"], 0.0
            )
            self.assertIsNone(
                result["tasks"]["vood"]["candidate"]["answered_accuracy"]
            )

    def test_rejects_mismatched_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "examples.jsonl",
                [{"id": "kcv-1", "task": "KCV", "gold_label": "T"}],
            )
            write_jsonl(root / "baseline.jsonl", [{"id": "other", "raw_output": "T"}])
            write_jsonl(root / "candidate.jsonl", [{"id": "kcv-1", "raw_output": "T"}])

            with self.assertRaisesRegex(ValueError, "baseline IDs"):
                compare_runs(
                    root / "examples.jsonl",
                    root / "baseline.jsonl",
                    root / "candidate.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
