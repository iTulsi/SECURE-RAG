from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
