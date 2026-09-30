"""Read unambiguous CVSS facts from the CVE JSON supplied in a KCV prompt."""

from __future__ import annotations

import re
from pathlib import Path

from .evaluation import _read_jsonl
from .pilot import write_jsonl
from .retrieval import extract_context
from .runner import _load_examples


def _cvss31(context: object) -> dict[str, object] | None:
    if not isinstance(context, dict):
        return None
    containers = context.get("containers")
    if not isinstance(containers, dict):
        return None
    metrics: list[dict[str, object]] = []
    for container in containers.values():
        if not isinstance(container, dict):
            continue
        entries = container.get("metrics")
        if not isinstance(entries, list):
            continue
        for item in entries:
            if isinstance(item, dict) and isinstance(item.get("cvssV3_1"), dict):
                metrics.append(item["cvssV3_1"])
    # Conflicting or missing records need the model, not a guessed override.
    return metrics[0] if metrics and all(m == metrics[0] for m in metrics) else None


def cvss_answer(question: str, context: object) -> str | None:
    """Return T/F only for a supported, explicit CVSS v3.1 claim."""
    metric = _cvss31(context)
    if metric is None:
        return None
    claim = question.lower().strip()
    checks: list[bool] = []

    if "base score" in claim:
        score = metric.get("baseScore")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            match = re.search(r"\b(?:higher|greater) than\s+(\d+(?:\.\d+)?)\b", claim)
            if match:
                checks.append(score > float(match.group(1)))
            else:
                match = re.search(
                    r"\b(?:is|of)\s+(\d+(?:\.\d+)?)\b",
                    claim.split("base score", 1)[1],
                )
                if match:
                    checks.append(score == float(match.group(1)))
        else:
            return None
    severity = re.search(r"\b(low|medium|high|critical) severity\b", claim)
    if not severity:
        severity = re.search(
            r"\bseverity\b[^.]{0,80}?\b(?:categorized|rated|classified)\s+as\s+"
            r"(low|medium|high|critical)\b", claim
        )
    if severity and ("severity" in claim or "base score" in claim):
        actual = metric.get("baseSeverity")
        if isinstance(actual, str):
            checks.append(actual.lower() == severity.group(1))
        else:
            return None

    if "attack complexity" in claim:
        match = re.search(r"\b(?:low|high)\b(?=\s+(?:in cvss|for|attack complexity|$|[.,]))", claim)
        actual = metric.get("attackComplexity")
        if match and isinstance(actual, str):
            checks.append(actual.lower() == match.group())
        else:
            return None

    if not checks:
        return None
    return "T" if all(checks) else "F"


def apply_cvss_facts(examples_path: Path, predictions_path: Path, output_path: Path) -> int:
    """Override only directly readable KCV claims; retain all other predictions."""
    if output_path in {examples_path, predictions_path}:
        raise ValueError("Choose a new output path")
    examples = _load_examples(examples_path)
    predictions = _read_jsonl(predictions_path)
    by_id = {row.get("id"): row for row in examples}
    if (len(by_id) != len(examples)
            or len({row.get("id") for row in predictions}) != len(predictions)):
        raise ValueError("Duplicate example or prediction IDs")
    if {row.get("id") for row in predictions} != set(by_id):
        raise ValueError("Prediction IDs do not match examples")
    output: list[dict[str, object]] = []
    changed = 0
    for prediction in predictions:
        example = by_id[prediction["id"]]
        if not isinstance(prediction.get("raw_output"), str):
            raise ValueError(f"Missing raw output for {prediction['id']}")
        result = dict(prediction)
        if example.get("task") == "KCV":
            question = example.get("question")
            if not isinstance(question, str):
                raise ValueError(f"Missing question for {example['id']}")
            answer = cvss_answer(question, extract_context(str(example["prompt"])))
            if answer is not None:
                result["cvss_fact_answer"] = answer
                result["original_raw_output"] = result.get("raw_output")
                result["raw_output"] = answer
                changed += 1
        output.append(result)
    write_jsonl(output_path, output)
    return changed
