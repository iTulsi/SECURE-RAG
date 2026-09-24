from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .evaluation import LABELS, parse_label


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"Invalid row at {path}:{line_number}")
            rows.append(row)
    return rows


def _index_unique(rows: list[dict[str, object]], name: str) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        identifier = str(row["id"])
        if identifier in indexed:
            raise ValueError(f"Duplicate {name} id: {identifier}")
        indexed[identifier] = row
    return indexed


def _prediction_labels(
    rows: list[dict[str, object]], name: str
) -> dict[str, str | None]:
    indexed = _index_unique(rows, name)
    return {
        identifier: parse_label(row.get("raw_output"))
        for identifier, row in indexed.items()
    }


def _task_summary(
    identifiers: list[str],
    examples: dict[str, dict[str, object]],
    baseline: dict[str, str | None],
    candidate: dict[str, str | None],
) -> dict[str, object]:
    label_transitions = Counter(
        f"{baseline[identifier] or 'INVALID'}->{candidate[identifier] or 'INVALID'}"
        for identifier in identifiers
    )
    correctness_transitions: Counter[str] = Counter()
    for identifier in identifiers:
        gold = str(examples[identifier]["gold_label"])
        before = "correct" if baseline[identifier] == gold else "wrong"
        after = "correct" if candidate[identifier] == gold else "wrong"
        correctness_transitions[f"{before}->{after}"] += 1

    def metrics(predictions: dict[str, str | None]) -> dict[str, object]:
        correct = sum(
            predictions[identifier] == examples[identifier]["gold_label"]
            for identifier in identifiers
        )
        answered = sum(predictions[identifier] in {"T", "F"} for identifier in identifiers)
        answered_correct = sum(
            predictions[identifier] in {"T", "F"}
            and predictions[identifier] == examples[identifier]["gold_label"]
            for identifier in identifiers
        )
        return {
            "accuracy": correct / len(identifiers),
            "coverage": answered / len(identifiers),
            "answered_accuracy": (
                None if answered == 0 else answered_correct / answered
            ),
            "correct": correct,
            "answered": answered,
            "answered_correct": answered_correct,
            "predicted_labels": dict(
                sorted(
                    Counter(
                        predictions[identifier] or "INVALID"
                        for identifier in identifiers
                    ).items()
                )
            ),
        }

    return {
        "examples": len(identifiers),
        "baseline": metrics(baseline),
        "candidate": metrics(candidate),
        "label_transitions": dict(sorted(label_transitions.items())),
        "correctness_transitions": dict(sorted(correctness_transitions.items())),
    }


def compare_runs(
    examples_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    retrieval_path: Path | None = None,
) -> dict[str, object]:
    examples = _index_unique(_read_jsonl(examples_path), "example")
    baseline = _prediction_labels(_read_jsonl(baseline_path), "baseline prediction")
    candidate = _prediction_labels(_read_jsonl(candidate_path), "candidate prediction")
    expected_ids = set(examples)
    for name, values in (("baseline", baseline), ("candidate", candidate)):
        if set(values) != expected_ids:
            raise ValueError(f"{name} IDs do not match examples")

    for identifier, row in examples.items():
        if row.get("task") not in {"KCV", "VOOD"}:
            raise ValueError(f"Example {identifier} has an invalid task")
        if row.get("gold_label") not in LABELS:
            raise ValueError(f"Example {identifier} has an invalid gold label")

    retrieval: dict[str, dict[str, object]] = {}
    if retrieval_path is not None:
        retrieval = _index_unique(_read_jsonl(retrieval_path), "retrieval")
        if set(retrieval) != expected_ids:
            raise ValueError("retrieval IDs do not match examples")

    tasks: dict[str, object] = {}
    for task in ("KCV", "VOOD"):
        identifiers = [
            identifier
            for identifier, row in examples.items()
            if row["task"] == task
        ]
        if identifiers:
            tasks[task.lower()] = _task_summary(
                identifiers, examples, baseline, candidate
            )

    changes: list[dict[str, object]] = []
    score_groups: defaultdict[str, list[float]] = defaultdict(list)
    for identifier, example in examples.items():
        gold = str(example["gold_label"])
        before_correct = baseline[identifier] == gold
        after_correct = candidate[identifier] == gold
        retrieved = retrieval.get(identifier, {}).get("retrieved", [])
        top_score: float | None = None
        if isinstance(retrieved, list) and retrieved:
            raw_score = retrieved[0].get("score")
            if isinstance(raw_score, (int, float)):
                top_score = float(raw_score)

        if example["task"] == "KCV" and top_score is not None:
            outcome = (
                "correct"
                if after_correct
                else "abstain"
                if candidate[identifier] == "X"
                else "wrong_answer"
            )
            score_groups[outcome].append(top_score)

        if before_correct == after_correct:
            continue
        changes.append(
            {
                "id": identifier,
                "task": example["task"],
                "question": example.get("question"),
                "gold_label": gold,
                "baseline_label": baseline[identifier],
                "candidate_label": candidate[identifier],
                "change": "improvement" if after_correct else "regression",
                "top_retrieval_score": top_score,
            }
        )

    score_summary = {
        outcome: {
            "examples": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "minimum": min(scores),
            "maximum": max(scores),
        }
        for outcome, scores in sorted(score_groups.items())
    }
    return {
        "examples": len(examples),
        "tasks": tasks,
        "changed_correctness": changes,
        "kcv_top_retrieval_score_by_candidate_outcome": score_summary,
    }


def write_comparison(path: Path, comparison: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
