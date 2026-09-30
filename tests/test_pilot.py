from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from secure_rag.cli import main
from secure_rag.dataset import Example
from secure_rag.pilot import select_paired_pilot


def example(task: str, row_index: int, source: str, label: str) -> Example:
    return Example(
        id=f"{task.lower()}-{row_index:04d}",
        task=task,
        row_index=row_index,
        source_url=source,
        prompt="prompt",
        question=f"question {row_index}",
        gold_label=label,
    )


class PilotTests(unittest.TestCase):
    def test_selection_is_deterministic_and_paired(self) -> None:
        kcv = [
            example("KCV", index, f"https://example.com/{index // 2}", "T")
            for index in range(6)
        ]
        vood = [
            example("VOOD", index, f"https://example.com/{index // 2}", "X")
            for index in range(6)
        ]

        first, sources = select_paired_pilot(kcv, vood, source_count=2, seed=7)
        second, _ = select_paired_pilot(kcv, vood, source_count=2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual({row.source_url for row in first}, set(sources))
        self.assertEqual(
            {row.row_index for row in first if row.task == "KCV"},
            {row.row_index for row in first if row.task == "VOOD"},
        )

    def test_rejects_too_many_sources(self) -> None:
        kcv = [example("KCV", 0, "https://example.com/1", "T")]
        vood = [example("VOOD", 0, "https://example.com/1", "X")]
        with self.assertRaises(ValueError):
            select_paired_pilot(kcv, vood, source_count=2, seed=7)

    def test_excludes_previous_sources_before_sampling(self) -> None:
        kcv = [example("KCV", i, f"https://example.com/{i}", "T") for i in range(4)]
        vood = [example("VOOD", i, f"https://example.com/{i}", "X") for i in range(4)]
        excluded = frozenset({"https://example.com/1", "https://example.com/2"})
        selected, sources = select_paired_pilot(
            kcv, vood, source_count=2, seed=7, excluded_sources=excluded
        )
        self.assertFalse(excluded.intersection(sources))
        self.assertEqual(len(selected), 4)
        with self.assertRaises(ValueError):
            select_paired_pilot(
                kcv, vood, source_count=3, seed=7, excluded_sources=excluded
            )

    def test_cli_manifest_excludes_a_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task, label in (("KCV", "T"), ("VOOD", "X")):
                (root / f"SECURE - {task}.tsv").write_text(
                    "URL\tPrompt\tQuestion\tCorrect Answer\n"
                    + f"https://example.com/a\tQuestion A\tQuestion A\t{label}\n"
                    + f"https://example.com/b\tQuestion B\tQuestion B\t{label}\n",
                    encoding="utf-8",
                )
            excluded = root / "excluded.json"
            excluded.write_text(json.dumps({"selected_sources": ["https://example.com/a"]}))
            output = root / "holdout"
            with redirect_stdout(StringIO()):
                main(["prepare-pilot", "--data-dir", str(root),
                      "--output-dir", str(output), "--sources", "1",
                      "--exclude-manifest", str(excluded)])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["selected_sources"], ["https://example.com/b"])
            self.assertEqual(manifest["excluded_sources"], ["https://example.com/a"])


if __name__ == "__main__":
    unittest.main()
