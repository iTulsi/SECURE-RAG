from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


LABELS = ("T", "F", "X")


def parse_label(raw_output: object) -> str | None:
    if not isinstance(raw_output, str):
        return None
    candidate = raw_output.strip().upper()
    return candidate if candidate in LABELS else None


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _f1_for_label(gold: list[str], predicted: list[str | None], label: str) -> float:
    true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
    false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
    false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _metrics(gold: list[str], predicted: list[str | None]) -> dict[str, object]:
    if not gold:
        raise ValueError("Cannot evaluate an empty set")
    present_labels = sorted(set(gold))
    correct = sum(g == p for g, p in zip(gold, predicted))
    answered = sum(p in {"T", "F"} for p in predicted)
    invalid = sum(p is None for p in predicted)
    return {
        "examples": len(gold),
        "accuracy": correct / len(gold),
        "macro_f1": sum(_f1_for_label(gold, predicted, label) for label in present_labels)
        / len(present_labels),
        "coverage": answered / len(gold),
        "invalid_predictions": invalid,
        "gold_labels": dict(sorted(Counter(gold).items())),
        "predicted_labels": dict(
            sorted(Counter(p if p is not None else "INVALID" for p in predicted).items())
        ),
    }


def evaluate(examples_path: Path, predictions_path: Path) -> dict[str, object]:
    examples = _read_jsonl(examples_path)
    prediction_rows = _read_jsonl(predictions_path)

    for row in examples:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Every example requires a non-empty string id")
        if row.get("task") not in {"KCV", "VOOD"}:
            raise ValueError(f"Example {row['id']} has an invalid task")
        if row.get("gold_label") not in LABELS:
            raise ValueError(f"Example {row['id']} has an invalid gold label")

    predictions: dict[str, str | None] = {}
    for row in prediction_rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every prediction requires a non-empty string id")
        if identifier in predictions:
            raise ValueError(f"Duplicate prediction id: {identifier}")
        predictions[identifier] = parse_label(row.get("raw_output"))

    example_ids = {str(row["id"]) for row in examples}
    if len(example_ids) != len(examples):
        raise ValueError("Example IDs must be unique")
    missing = sorted(example_ids - predictions.keys())
    extra = sorted(predictions.keys() - example_ids)
    if missing or extra:
        raise ValueError(
            f"Prediction IDs do not match examples: {len(missing)} missing, {len(extra)} extra"
        )

    results: dict[str, object] = {}
    all_gold: list[str] = []
    all_predicted: list[str | None] = []
    for task in ("KCV", "VOOD"):
        task_rows = [row for row in examples if row.get("task") == task]
        gold = [str(row["gold_label"]) for row in task_rows]
        predicted = [predictions[str(row["id"])] for row in task_rows]
        results[task.lower()] = _metrics(gold, predicted)
        all_gold.extend(gold)
        all_predicted.extend(predicted)

    results["overall"] = _metrics(all_gold, all_predicted)

    true_positive = sum(g == "X" and p == "X" for g, p in zip(all_gold, all_predicted))
    false_positive = sum(g != "X" and p == "X" for g, p in zip(all_gold, all_predicted))
    false_negative = sum(g == "X" and p != "X" for g, p in zip(all_gold, all_predicted))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = 0.0 if precision_denominator == 0 else true_positive / precision_denominator
    recall = 0.0 if recall_denominator == 0 else true_positive / recall_denominator
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    results["abstention"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsupported_answer_rate": (
            0.0 if recall_denominator == 0 else false_negative / recall_denominator
        ),
    }
    return results


def write_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
