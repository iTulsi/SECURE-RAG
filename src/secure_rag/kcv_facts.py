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
    claim = re.sub(r"^true or false:\s*", "", claim)
    interaction = re.fullmatch(
        r"the cvss vector string indicates that user interaction is (not )?required"
        r"(?: for the exploit)?[.?!]?", claim,
    )
    if interaction:
        vector = metric.get("vectorString")
        vector_match = re.search(r"(?:^|/)UI:([NR])(?:/|$)", vector) if isinstance(vector, str) else None
        actual = metric.get("userInteraction")
        if actual is None and vector_match:
            actual = {"N": "NONE", "R": "REQUIRED"}[vector_match.group(1)]
        if not isinstance(actual, str) or actual not in {"NONE", "REQUIRED"}:
            return None
        if vector_match and actual != {"N": "NONE", "R": "REQUIRED"}[vector_match.group(1)]:
            return None
        expected = "NONE" if interaction.group(1) else "REQUIRED"
        return "T" if actual == expected else "F"
    # The narrow comparator below does not parse negation or unrelated clauses.
    # Send these to the model instead of partially evaluating a compound claim.
    if re.search(r"\b(?:not|never|without|and|or|but|while)\b|n't\b", claim):
        return None
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
    if not all(checks):
        return "F"
    # A matching field proves T only when the entire claim fits this grammar.
    number = r"\d+(?:\.\d+)?"
    cvss = r"cvss(?:\s+v?3(?:\.1)?)?"
    score_claim = (
        rf"(?:the )?(?:{cvss} )?base score"
        rf"(?: for (?:this vulnerability|cve-\d{{4}}-\d{{4,}}))?"
        rf"(?: under version 3\.1| according to {cvss})?"
        rf" (?:is|of) (?:(?:higher|greater) than )?{number}"
        r"(?:, indicating a (?:low|medium|high|critical) severity(?: vulnerability)?)?[.?!]?"
    )
    severity_claim = (
        r"(?:the )?severity is (?:categorized|rated|classified) as "
        r"(?:low|medium|high|critical)[.?!]?"
    )
    complexity_claim = (
        rf"(?:the )?attack complexity is (?:rated as )?(?:low|high)(?: in {cvss})?[.?!]?"
    )
    if any(re.fullmatch(pattern, claim) for pattern in
           (score_claim, severity_claim, complexity_claim)):
        return "T"
    return None


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
