from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .pilot import write_jsonl


class BaselineError(RuntimeError):
    """Raised when the model endpoint returns an unusable response."""


def _load_examples(path: Path) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not isinstance(row.get("prompt"), str)
                or not row["prompt"]
            ):
                raise ValueError(f"Invalid example at {path}:{line_number}")
            examples.append(row)
    return examples


def _read_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                identifier = row["id"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid cached prediction at line {line_number}") from error
            if not isinstance(identifier, str) or identifier in seen:
                raise ValueError(f"Invalid or duplicate cached id at line {line_number}")
            seen.add(identifier)
            rows.append(row)
    return rows


def _request_prediction(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    timeout_seconds: float,
    seed: int | None,
    max_tokens: int = 4,
) -> tuple[str, dict[str, object]]:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise BaselineError(f"Model request failed: {error}") from error

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise BaselineError("Endpoint response lacks choices[0].message.content") from error
    if not isinstance(content, str):
        raise BaselineError("Endpoint returned non-text model content")
    usage = body.get("usage") if isinstance(body, dict) else None
    return content, usage if isinstance(usage, dict) else {}


def run_baseline(
    examples_path: Path,
    output_path: Path,
    base_url: str,
    model: str,
    timeout_seconds: float,
    seed: int | None,
    limit: int | None,
) -> tuple[int, int]:
    examples = _load_examples(examples_path)
    cached = _read_existing(output_path)
    if len({str(row["id"]) for row in examples}) != len(examples):
        raise ValueError("Example IDs must be unique")
    completed_ids = {str(row["id"]) for row in cached}
    example_ids = {str(row["id"]) for row in examples}
    unknown_ids = completed_ids - example_ids
    if unknown_ids:
        raise ValueError(f"Cache contains {len(unknown_ids)} unknown prediction IDs")
    examples_by_id = {str(row["id"]): row for row in examples}
    for row in cached:
        expected = hashlib.sha256(
            str(examples_by_id[str(row["id"])]["prompt"]).encode("utf-8")
        ).hexdigest()
        if (
            row.get("prompt_sha256") != expected
            or row.get("model") != model
            or row.get("request_parameters")
            != {"temperature": 0, "max_tokens": 4, "seed": seed}
        ):
            raise ValueError(
                "Prediction cache does not match prompts, model, or settings; "
                "use a fresh output path"
            )
    pending = [row for row in examples if str(row["id"]) not in completed_ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        pending = pending[:limit]

    api_key = os.environ.get("SECURE_RAG_API_KEY")
    new_rows: list[dict[str, object]] = []
    for example in pending:
        started = time.perf_counter()
        raw_output, usage = _request_prediction(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=str(example["prompt"]),
            timeout_seconds=timeout_seconds,
            seed=seed,
        )
        new_rows.append(
            {
                "id": example["id"],
                "raw_output": raw_output,
                "model": model,
                "prompt_sha256": hashlib.sha256(
                    str(example["prompt"]).encode("utf-8")
                ).hexdigest(),
                "latency_seconds": round(time.perf_counter() - started, 6),
                "request_parameters": {
                    "temperature": 0,
                    "max_tokens": 4,
                    "seed": seed,
                },
                "usage": usage,
            }
        )
        write_jsonl(output_path, [*cached, *new_rows])
    return len(new_rows), len(cached) + len(new_rows)
