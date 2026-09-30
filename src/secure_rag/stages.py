"""Reranking and evidence-verification stages for the paired SECURE pilot."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from pathlib import Path

from .evaluation import _read_jsonl, parse_label
from .pilot import write_jsonl
from .retrieval import _retrieval_prompt, extract_context
from .runner import _load_examples, _request_prediction


def _indexed_rows(path: Path, expected_ids: set[str]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path):
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier in rows:
            raise ValueError(f"Invalid or duplicate ID in {path}")
        rows[identifier] = row
    if rows.keys() != expected_ids:
        raise ValueError(f"IDs in {path} do not match the examples")
    return rows


def _passages(row: dict[str, object], field: str) -> list[dict[str, object]]:
    passages = row.get(field)
    if not isinstance(passages, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("text"), str)
        for item in passages
    ):
        raise ValueError(f"Invalid {field} passages for {row.get('id')}")
    return passages


def prepare_context_prompts(examples_path: Path, output_path: Path) -> dict[str, int]:
    """Prepare a single inference run with E0 for context and E1 for absence."""
    if examples_path == output_path:
        raise ValueError("Choose a new output path")
    examples = _load_examples(examples_path)
    if len({str(row["id"]) for row in examples}) != len(examples):
        raise ValueError("Example IDs must be unique")
    counts = {"context_present": 0, "context_missing": 0}
    prepared: list[dict[str, object]] = []
    for example in examples:
        has_context = extract_context(str(example["prompt"])) is not None
        name = "context_present" if has_context else "context_missing"
        counts[name] += 1
        row = dict(example)
        row["selected_run"] = "E0" if has_context else "E1"
        if not has_context:
            question = row.get("question")
            if not isinstance(question, str):
                raise ValueError(f"Example {row['id']} lacks a question")
            row["prompt"] = _retrieval_prompt(question, [])
        prepared.append(row)
    write_jsonl(output_path, prepared)
    return counts


def prepare_claim_evidence(examples_path: Path, output_path: Path) -> dict[str, int]:
    """Keep original CVE facts but remove repeated display and provenance metadata."""
    if examples_path == output_path:
        raise ValueError("Choose a new output path")
    examples = _load_examples(examples_path)
    if len({str(row["id"]) for row in examples}) != len(examples):
        raise ValueError("Example IDs must be unique")
    counts = {"context_present": 0, "context_missing": 0}
    output: list[dict[str, object]] = []
    for example in examples:
        question = example.get("question")
        if not isinstance(question, str):
            raise ValueError(f"Example {example['id']} lacks a question")
        context = extract_context(str(example["prompt"]))
        row = dict(example)
        if context is None:
            counts["context_missing"] += 1
            row["prompt"] = (
                "There is no CVE record for this statement. Do not use outside "
                "knowledge. Use X if the record cannot decide. State a short "
                "reason, then end with VERDICT: T, VERDICT: F, or VERDICT: X.\n"
                f"Statement: {question}\nEVIDENCE:"
            )
        else:
            counts["context_present"] += 1

            def compact(value: object) -> object:
                if isinstance(value, list):
                    return [compact(item) for item in value]
                if not isinstance(value, dict):
                    return value
                media = value.get("supportingMedia")
                original = value.get("value")
                redundant_media = (
                    isinstance(original, str)
                    and isinstance(media, list)
                    and all(
                        isinstance(item, dict)
                        and item.get("base64") is not True
                        and isinstance(item.get("value"), str)
                        and " ".join(html.unescape(re.sub(
                            r"<[^>]*>", " ", item["value"]
                        )).split()).lower() == " ".join(original.split()).lower()
                        for item in media
                    )
                )
                return {
                    key: compact(child)
                    for key, child in value.items()
                    if key not in {"providerMetadata", "credits", "references", "x_generator"}
                    and (key != "supportingMedia" or not redundant_media)
                }

            # Keep all other fields, including descriptions, solutions, and
            # affected versions. No scoring label is consulted during preparation.
            facts = compact(context)
            row["prompt"] = (
                "Check the statement against this CVE record. Compare every part "
                "of the claim, including negation, version bounds, severity, and "
                "CVSS vector fields. A false claim is F even when one part is true. "
                "Use X only when the record cannot decide. State a short relevant "
                "fact, then explain the comparison. End with VERDICT: T, "
                "VERDICT: F, or VERDICT: X.\n\n"
                f"Statement: {question}\n\n"
                f"CVE record: {json.dumps(facts, ensure_ascii=False, separators=(',', ':'))}"
                f"\n\nStatement: {question}\n"
                "EVIDENCE:"
            )
        output.append(row)
    write_jsonl(output_path, output)
    return counts


def route_by_context(
    examples_path: Path,
    dense_examples_path: Path,
    original_predictions_path: Path,
    dense_predictions_path: Path,
    output_examples_path: Path,
    output_predictions_path: Path,
) -> dict[str, int]:
    """Use the original prompt with evidence and the E1 prompt without it."""
    inputs = {examples_path, dense_examples_path, original_predictions_path,
              dense_predictions_path}
    if (output_examples_path == output_predictions_path
            or output_examples_path in inputs or output_predictions_path in inputs):
        raise ValueError("Choose new paths for routed outputs")

    examples = _load_examples(examples_path)
    ids = {str(row["id"]) for row in examples}
    if len(ids) != len(examples):
        raise ValueError("Example IDs must be unique")
    dense_examples = _indexed_rows(dense_examples_path, ids)
    original = _indexed_rows(original_predictions_path, ids)
    dense = _indexed_rows(dense_predictions_path, ids)
    routed_examples: list[dict[str, object]] = []
    routed_predictions: list[dict[str, object]] = []
    counts = {"context_present": 0, "context_missing": 0}

    for example in examples:
        identifier = str(example["id"])
        has_context = extract_context(str(example["prompt"])) is not None
        name = "context_present" if has_context else "context_missing"
        counts[name] += 1
        if has_context:
            chosen_example, prediction = example, original[identifier]
        else:
            chosen_example, prediction = dense_examples[identifier], dense[identifier]
            if chosen_example["prompt"] != _retrieval_prompt(
                str(example["question"]), []
            ):
                raise ValueError(f"E1 missing-context prompt changed for {identifier}")
        if (
            chosen_example.get("gold_label") != example.get("gold_label")
            or chosen_example.get("question") != example.get("question")
            or not isinstance(prediction.get("raw_output"), str)
        ):
            raise ValueError(f"Incompatible example or prediction for {identifier}")
        if (original[identifier].get("model") != dense[identifier].get("model")
                or original[identifier].get("request_parameters")
                != dense[identifier].get("request_parameters")):
            raise ValueError(f"Model or inference settings mismatch for {identifier}")
        routed_examples.append({**chosen_example, "selected_run": "E0" if has_context else "E1"})
        routed_predictions.append({**prediction, "selected_run": "E0" if has_context else "E1"})

    write_jsonl(output_examples_path, routed_examples)
    write_jsonl(output_predictions_path, routed_predictions)
    return counts


def prepare_reranked_examples(
    examples_path: Path,
    retrieval_path: Path,
    output_path: Path,
    reranking_path: Path,
    base_url: str,
    model: str,
    timeout_seconds: float,
    seed: int | None,
    top_k: int,
) -> int:
    """Pick the most relevant hybrid passages in one model call per KCV row."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    examples = _load_examples(examples_path)
    identifiers = {str(row["id"]) for row in examples}
    if len(identifiers) != len(examples):
        raise ValueError("Example IDs must be unique")
    retrieval = _indexed_rows(retrieval_path, identifiers)
    cached = _read_jsonl(reranking_path) if reranking_path.exists() else []
    cache: dict[str, dict[str, object]] = {}
    for row in cached:
        identifier = row.get("id")
        if (
            not isinstance(identifier, str)
            or identifier in cache
            or identifier not in identifiers
            or row.get("model") != model
            or row.get("seed") != seed
            or row.get("top_k") != top_k
        ):
            raise ValueError("Incompatible or invalid reranking cache")
        cache[identifier] = row

    api_key = os.environ.get("SECURE_RAG_API_KEY")
    output_rows: list[dict[str, object]] = []
    new_rankings = 0
    for example in examples:
        identifier = str(example["id"])
        question = example.get("question")
        if not isinstance(question, str):
            raise ValueError(f"Example {identifier} lacks a question")
        source = retrieval[identifier]
        pool = _passages(source, "candidate_pool")
        if not pool and _passages(source, "retrieved"):
            raise ValueError("Hybrid retrieval must include candidate_pool; rerun prepare-hybrid")
        count = min(top_k, len(pool))
        fingerprint = hashlib.sha256(json.dumps(
            [question, [passage["text"] for passage in pool]],
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        if pool and identifier not in cache:
            choices = "\n".join(
                f"[{index}] {passage['text']}"
                for index, passage in enumerate(pool, start=1)
            )
            prompt = (
                f"Select the {count} passages most useful for deciding the statement. "
                "Rank by relevance, not by whether the statement is true. "
                f"Reply with exactly {count} distinct passage numbers separated "
                "by commas, with no explanation.\n\n"
                f"Statement: {question}\n\nPassages:\n{choices}"
            )
            started = time.perf_counter()
            raw, usage = _request_prediction(
                base_url, api_key, model, prompt, timeout_seconds, seed, max_tokens=32
            )
            formatted = re.fullmatch(r"\d+(?:\s*,\s*\d+)*", raw.strip())
            proposed = (
                [int(value.strip()) for value in raw.strip().split(",")]
                if formatted else []
            )
            valid = (
                len(proposed) == count
                and len(set(proposed)) == count
                and all(1 <= index <= len(pool) for index in proposed)
            )
            # Preserve the usable ranking, then fill missing slots in E2 order.
            # Log the repair so invalid model output remains visible in analysis.
            indices = list(dict.fromkeys(
                index for index in proposed if 1 <= index <= len(pool)
            ))[:count]
            indices.extend(
                index for index in range(1, len(pool) + 1)
                if index not in indices and len(indices) < count
            )
            row = {
                "id": identifier,
                "pool_sha256": fingerprint,
                "model": model,
                "seed": seed,
                "top_k": top_k,
                "indices": indices,
                "ranking_valid": valid,
                "raw_output": raw,
                "usage": usage,
                "latency_seconds": round(time.perf_counter() - started, 6),
            }
            cache[identifier] = row
            cached.append(row)
            write_jsonl(reranking_path, cached)
            new_rankings += 1
        if identifier in cache:
            selected_indices = cache[identifier].get("indices")
            if (
                cache[identifier].get("pool_sha256") != fingerprint
                or not isinstance(selected_indices, list)
                or len(selected_indices) != count
                or any(not isinstance(index, int) or index < 1 or index > len(pool)
                       for index in selected_indices)
                or len(set(selected_indices)) != count
            ):
                raise ValueError("Reranking cache has stale passages or invalid indices")
        else:
            selected_indices = []
        selected = [pool[index - 1] for index in selected_indices]
        result = dict(example)
        result["prompt"] = _retrieval_prompt(
            question, [str(item["text"]) for item in selected]
        )
        output_rows.append(result)
    expected_ids = {
        str(row["id"]) for row in examples
        if _passages(retrieval[str(row["id"])], "candidate_pool")
    }
    if cache.keys() != expected_ids:
        raise ValueError("Reranking cache has stale passages; choose a fresh cache path")
    write_jsonl(output_path, output_rows)
    return new_rankings


def prepare_verification(examples_path: Path, output_path: Path) -> int:
    """Ask an independent model whether the selected evidence entails T, F, or X."""
    examples = _load_examples(examples_path)
    output: list[dict[str, object]] = []
    for example in examples:
        prompt = str(example["prompt"])
        if not prompt.startswith("Use only the evidence below") or (
            "Evidence:\n" not in prompt or "\n\nStatement: " not in prompt
        ):
            raise ValueError("Verification requires prepared retrieval prompts")
        evidence = prompt.split("Evidence:\n", 1)[1].split("\n\nStatement: ", 1)[0]
        row = dict(example)
        row["prompt"] = (
            "Independently verify the statement using only the evidence. "
            "Return T if directly supported, F if directly contradicted, or X "
            "if either conclusion lacks sufficient evidence. Reply with one letter.\n\n"
            f"Evidence:\n{evidence}\n\nStatement: {example['question']}"
        )
        output.append(row)
    write_jsonl(output_path, output)
    return len(output)


def apply_verification(
    examples_path: Path,
    generator_path: Path,
    verifier_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Keep an answer only when generator and verifier agree on T or F."""
    examples = _load_examples(examples_path)
    identifiers = {str(row["id"]) for row in examples}
    if len(identifiers) != len(examples):
        raise ValueError("Example IDs must be unique")
    generator = _indexed_rows(generator_path, identifiers)
    verifier = _indexed_rows(verifier_path, identifiers)
    output: list[dict[str, object]] = []
    counts = {"agreed": 0, "abstained": 0, "invalid_verifier": 0}
    for example in examples:
        identifier = str(example["id"])
        answer = parse_label(generator[identifier].get("raw_output"))
        check = parse_label(verifier[identifier].get("raw_output"))
        if check is None:
            counts["invalid_verifier"] += 1
        agreed = answer in {"T", "F"} and answer == check
        counts["agreed" if agreed else "abstained"] += 1
        output.append(
            {
                "id": identifier,
                "raw_output": answer if agreed else "X",
                "generator_output": generator[identifier].get("raw_output"),
                "verifier_output": verifier[identifier].get("raw_output"),
                "decision": "agreed" if agreed else "abstained",
            }
        )
    write_jsonl(output_path, output)
    return counts
