from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


TASK_FILENAMES = {
    "KCV": "SECURE - KCV.tsv",
    "VOOD": "SECURE - VOOD.tsv",
}
EXPECTED_FIELDS = ("URL", "Prompt", "Question", "Correct Answer")
VALID_LABELS = frozenset({"T", "F", "X"})


class DatasetError(ValueError):
    """Raised when official benchmark data violates the expected schema."""


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    task: str
    row_index: int
    source_url: str
    prompt: str
    question: str
    gold_label: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_task(data_dir: Path, task: str) -> list[Example]:
    normalized_task = task.upper()
    if normalized_task not in TASK_FILENAMES:
        raise DatasetError(f"Unsupported task {task!r}; expected KCV or VOOD")

    path = data_dir / TASK_FILENAMES[normalized_task]
    if not path.is_file():
        raise DatasetError(f"Missing official dataset file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise DatasetError(
                f"{path.name} has fields {reader.fieldnames}; expected {EXPECTED_FIELDS}"
            )

        examples: list[Example] = []
        for row_index, row in enumerate(reader):
            source_url = row["URL"].strip()
            prompt = row["Prompt"].strip()
            question = row["Question"].strip()
            gold_label = row["Correct Answer"].strip().upper()
            line_number = row_index + 2

            if not _valid_url(source_url):
                raise DatasetError(f"{path.name}:{line_number} has an invalid URL")
            if not prompt or not question:
                raise DatasetError(f"{path.name}:{line_number} has empty text")
            if gold_label not in VALID_LABELS:
                raise DatasetError(
                    f"{path.name}:{line_number} has invalid label {gold_label!r}"
                )

            examples.append(
                Example(
                    id=f"{normalized_task.lower()}-{row_index:04d}",
                    task=normalized_task,
                    row_index=row_index,
                    source_url=source_url,
                    prompt=prompt,
                    question=question,
                    gold_label=gold_label,
                )
            )

    if not examples:
        raise DatasetError(f"{path.name} contains no examples")
    return examples


def validate_pair(kcv: list[Example], vood: list[Example]) -> None:
    if len(kcv) != len(vood):
        raise DatasetError(f"KCV has {len(kcv)} rows but VOOD has {len(vood)}")

    for left, right in zip(kcv, vood, strict=True):
        if left.row_index != right.row_index:
            raise DatasetError("KCV and VOOD row indices are not aligned")
        if (left.source_url, left.question) != (right.source_url, right.question):
            raise DatasetError(
                f"KCV and VOOD differ at row index {left.row_index}"
            )
        if right.gold_label != "X":
            raise DatasetError(
                f"VOOD row index {right.row_index} must use abstention label X"
            )


def summarize(examples: Iterable[Example]) -> dict[str, object]:
    rows = list(examples)
    return {
        "rows": len(rows),
        "sources": len({row.source_url for row in rows}),
        "labels": dict(sorted(Counter(row.gold_label for row in rows).items())),
    }


def load_and_validate(data_dir: Path) -> tuple[list[Example], list[Example]]:
    kcv = load_task(data_dir, "KCV")
    vood = load_task(data_dir, "VOOD")
    validate_pair(kcv, vood)
    return kcv, vood
