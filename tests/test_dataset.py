from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from secure_rag.dataset import DatasetError, load_and_validate, load_task


FIELDS = ["URL", "Prompt", "Question", "Correct Answer"]


def write_task(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(FIELDS)
        writer.writerows(rows)


class DatasetTests(unittest.TestCase):
    def test_load_and_validate_paired_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = ["https://example.com/CVE-1", "prompt", "question"]
            write_task(root / "SECURE - KCV.tsv", [[*common, "T"]])
            write_task(root / "SECURE - VOOD.tsv", [[*common, "X"]])

            kcv, vood = load_and_validate(root)

            self.assertEqual(kcv[0].id, "kcv-0000")
            self.assertEqual(vood[0].gold_label, "X")

    def test_rejects_invalid_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(
                root / "SECURE - KCV.tsv",
                [["https://example.com/CVE-1", "prompt", "question", "YES"]],
            )

            with self.assertRaisesRegex(DatasetError, "invalid label"):
                load_task(root, "KCV")

    def test_rejects_misaligned_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_task(
                root / "SECURE - KCV.tsv",
                [["https://example.com/CVE-1", "prompt", "question one", "F"]],
            )
            write_task(
                root / "SECURE - VOOD.tsv",
                [["https://example.com/CVE-1", "prompt", "question two", "X"]],
            )

            with self.assertRaisesRegex(DatasetError, "differ"):
                load_and_validate(root)


if __name__ == "__main__":
    unittest.main()
