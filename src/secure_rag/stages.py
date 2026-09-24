"""Reranking and evidence-verification stages for the paired SECURE pilot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from .evaluation import _read_jsonl, parse_label
from .pilot import write_jsonl
from .retrieval import _retrieval_prompt
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
