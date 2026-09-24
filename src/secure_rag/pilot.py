from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .dataset import Example, TASK_FILENAMES, sha256_file


def _source_order(source_url: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{source_url}".encode()).hexdigest()


def select_paired_pilot(
    kcv: list[Example], vood: list[Example], source_count: int, seed: int
) -> tuple[list[Example], list[str]]:
    all_sources = sorted({row.source_url for row in kcv})
    if source_count < 1 or source_count > len(all_sources):
        raise ValueError(
            f"source_count must be between 1 and {len(all_sources)}, got {source_count}"
        )

    selected_sources = sorted(
        all_sources, key=lambda url: _source_order(url, seed)
    )[:source_count]
    selected = set(selected_sources)
    examples = [row for row in (*kcv, *vood) if row.source_url in selected]
    return examples, selected_sources


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_pilot(
    data_dir: Path,
    output_dir: Path,
    kcv: list[Example],
    vood: list[Example],
    source_count: int,
    seed: int,
    upstream_commit: str | None,
) -> dict[str, object]:
    examples, sources = select_paired_pilot(kcv, vood, source_count, seed)
    examples_path = output_dir / "examples.jsonl"
    write_jsonl(examples_path, [row.to_dict() for row in examples])

    task_counts = Counter(row.task for row in examples)
    label_counts = Counter(f"{row.task}:{row.gold_label}" for row in examples)
    manifest: dict[str, object] = {
        "benchmark": "SECURE",
        "tasks": ["KCV", "VOOD"],
        "seed": seed,
        "source_count": source_count,
        "example_count": len(examples),
        "task_counts": dict(sorted(task_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "selected_sources": sources,
        "upstream_commit": upstream_commit,
        "dataset_sha256": {
            task: sha256_file(data_dir / filename)
            for task, filename in TASK_FILENAMES.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def make_prediction_template(examples_path: Path, output_path: Path) -> int:
    rows: list[dict[str, object]] = []
    with examples_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                example = json.loads(line)
                rows.append({"id": example["id"], "raw_output": "X"})
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid example JSONL at line {line_number}"
                ) from error
    write_jsonl(output_path, rows)
    return len(rows)
