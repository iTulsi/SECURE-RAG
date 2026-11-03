"""Task-specific scoring and paired uncertainty; no pooled six-task score."""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from statistics import fmean

from .benchmark_data import TASKS, digest

# Final ACSAC paper, Table V (DOI 10.1109/ACSAC63791.2024.00019).
# Classification values are fractions; RERT and CPST retain their native scales.
PUBLISHED_SCORES = {
    "ChatGPT-4": [.886, .896, .876, .879, .53, .81],
    "ChatGPT-3.5": [.828, .842, .783, .084, .48, 1.26],
    "Gemini-Pro": [.862, .878, .835, .867, .54, 1.0],
    "Llama3-70B": [.863, .904, .852, .271, .51, 1.54],
    "Llama3-8B": [.821, .839, .828, .564, .48, 1.77],
    "Mistral-7B": [.779, .801, .642, .571, .42, 1.82],
    "Mixtral-8x7B": [.809, .834, .796, .693, .39, 1.63],
}


def parse_answer(task: str, text: str, structured: bool) -> str | float | None:
    if structured:
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return None
        if (not isinstance(obj, dict) or set(obj) != {"analysis", "answer"}
                or not isinstance(obj["analysis"], str) or not obj["analysis"].strip()
                or not isinstance(obj["answer"], str)):
            return None
        text = obj["answer"]
    candidate = text.strip()
    if task in {"MAET", "CWET", "KCV", "VOOD"}:
        allowed = {"A", "B", "C", "D", "X"} if task in {"MAET", "CWET"} else {"T", "F", "X"}
        return candidate if candidate in allowed else None
    if task == "CPST":
        if not re.fullmatch(r"(?:10(?:\.0+)?|[0-9](?:\.[0-9]+)?)", candidate):
            return None
        value = float(candidate)
        return value if math.isfinite(value) and 0 <= value <= 10 else None
    if task == "RERT":
        return candidate if candidate else None
    raise ValueError(f"Unknown task: {task}")


def rouge_l_f1(reference: str, prediction: str) -> float:
    """ASCII alphanumeric tokens, lowercase, no stemming; beta=1 LCS F-score."""
    left, right = [re.findall(r"[a-z0-9]+", text.lower())
                   for text in (reference, prediction)]
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for word in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if word == other
                           else max(previous[index], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(left) + len(right))


def item_score(task: str, gold: str, prediction: str | float | None) -> float:
    if task == "CPST":
        # Invalid scores receive the maximum possible absolute error, never disappear.
        return 10.0 if prediction is None else abs(float(gold) - float(prediction))
    if task == "RERT":
        return rouge_l_f1(gold, str(prediction)) if prediction is not None else 0.0
    return float(gold == prediction)


def paired_interval(rows: list[dict], deltas: list[float], samples: int = 2000) -> list[float] | None:
    totals = defaultdict(lambda: [0.0, 0])
    for row, delta in zip(rows, deltas, strict=True):
        totals[row["group_id"]][0] += delta
        totals[row["group_id"]][1] += 1
    groups = [totals[key] for key in sorted(totals)]
    if len(groups) < 2:
        return None
    rng = random.Random(20261031)
    values = []
    for _ in range(samples):
        chosen = rng.choices(groups, k=len(groups))
        values.append(sum(g[0] for g in chosen) / sum(g[1] for g in chosen))
    values.sort()
    return [values[int(.025 * (samples - 1))], values[int(.975 * (samples - 1))]]


def score_task(rows: list[dict], labels: dict[str, str], answers: dict) -> dict:
    task = rows[0]["task"]
    predictions = [answers[r["id"]] for r in rows]
    scores = [item_score(task, labels[r["id"]], p)
              for r, p in zip(rows, predictions, strict=True)]
    metric = "mad_penalized" if task == "CPST" else "rouge_l_f1" if task == "RERT" else "accuracy"
    first_prompts = {}
    for row, score in zip(rows, scores, strict=True):
        first_prompts.setdefault(digest(row["prompt"]), score)
    result = {
        "examples": len(rows), "groups": len({r["group_id"] for r in rows}),
        "metric": metric, "value": fmean(scores),
        "invalid_predictions": sum(p is None for p in predictions),
        "unique_prompt_examples": len(first_prompts),
        "unique_prompt_value": fmean(first_prompts.values()),
        "ci95_cluster_bootstrap": paired_interval(rows, scores),
    }
    if task == "CPST":
        valid = [score for score, p in zip(scores, predictions, strict=True) if p is not None]
        result.update(mad_valid=fmean(valid) if valid else None,
                      valid_coverage=len(valid) / len(rows))
    elif task in {"MAET", "CWET", "KCV", "VOOD"}:
        gold = [labels[r["id"]] for r in rows]
        confusion = Counter(f"{g}->{p if p is not None else 'INVALID'}"
                            for g, p in zip(gold, predictions, strict=True))
        f1s = []
        for label in sorted(set(gold)):
            tp = sum(g == p == label for g, p in zip(gold, predictions, strict=True))
            fp = sum(g != label and p == label for g, p in zip(gold, predictions, strict=True))
            fn = sum(g == label and p != label for g, p in zip(gold, predictions, strict=True))
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        result.update(confusion=dict(sorted(confusion.items())), macro_f1=fmean(f1s),
                      abstention_rate=predictions.count("X") / len(rows))
        if task == "KCV":
            binary = [score for score, p in zip(scores, predictions, strict=True) if p in {"T", "F"}]
            result.update(binary_coverage=len(binary) / len(rows),
                          binary_answer_accuracy=fmean(binary) if binary else None)
        if task == "VOOD":
            result["unsupported_binary_answer_rate"] = sum(p in {"T", "F"} for p in predictions) / len(rows)
    return result


def score_suite(rows: list[dict], labels: dict[str, str], records: list[dict], arms: tuple[str, ...]) -> dict:
    indexed = {(r["id"], r["arm"]): r for r in records}
    expected = {(r["id"], arm) for r in rows for arm in arms}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("Missing, duplicate, or extra prediction IDs/arms; incomplete runs cannot be scored")
    results = {arm: {} for arm in arms}
    comparisons = {}
    for task in TASKS:
        selected = [r for r in rows if r["task"] == task]
        if not selected:
            continue
        answers = {}
        for arm in arms:
            answers[arm] = {
                r["id"]: (parse_answer(task, indexed[r["id"], arm]["response"]["message"]["content"],
                                       structured=arm != "official")
                          if indexed[r["id"], arm]["response"].get("done_reason") == "stop"
                          else None) for r in selected
            }
            results[arm][task] = score_task(selected, labels, answers[arm])
        pairs = [("official", arm) for arm in arms if arm != "official"]
        if "evidence_first" in arms and "self_review" in arms:
            pairs.append(("evidence_first", "self_review"))
        for baseline, candidate in pairs:
            # Positive always means improvement, including CPST where lower is better.
            direction = -1 if task == "CPST" else 1
            deltas = [direction * (
                item_score(task, labels[r["id"]], answers[candidate][r["id"]])
                - item_score(task, labels[r["id"]], answers[baseline][r["id"]])
            ) for r in selected]
            comparisons.setdefault(f"{baseline}->{candidate}", {})[task] = {
                "improvement": fmean(deltas),
                "ci95_cluster_bootstrap": paired_interval(selected, deltas),
                "improved_examples": sum(d > 0 for d in deltas),
                "regressed_examples": sum(d < 0 for d in deltas),
                "unit": "MAD reduction" if task == "CPST" else "ROUGE-L F1 increase"
                if task == "RERT" else "accuracy fraction (multiply by 100 for percentage points)",
            }
    return {
        "by_arm": results, "paired_comparisons": comparisons,
        "published_reference": {model: dict(zip(TASKS, values, strict=True))
                                for model, values in PUBLISHED_SCORES.items()},
        "published_reference_source": "ACSAC 2024 final paper, Table V, doi:10.1109/ACSAC63791.2024.00019",
        "uncertainty": "2000 paired source-cluster bootstrap samples, seed 20261031; exploratory per-task intervals, no multiplicity correction.",
        "rouge_definition": "ASCII alphanumeric lowercase tokens, no stemming, LCS F1 beta=1. Historical tokenizer/beta parity is unverified.",
        "historical_comparison": "Published scores are descriptive context only; different models, sampling, manual output correction and unverified ROUGE parity prevent causal improvement claims.",
        "pooled_score": None,
    }
