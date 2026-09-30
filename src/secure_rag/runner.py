from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .pilot import write_jsonl


class BaselineError(RuntimeError):
    """Raised when the model endpoint returns an unusable response."""


VERDICT_PATTERN = re.compile(r"VERDICT\s*:\s*([TFX])\.?", re.IGNORECASE)
VERDICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidence_verdict", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "evidence": {"type": "string"},
                "comparison": {"type": "string"},
                "verdict": {"type": "string", "enum": ["T", "F", "X"]},
            },
            "required": ["evidence", "comparison", "verdict"],
            "additionalProperties": False,
        },
    },
}


def parse_structured_verdict(response: str) -> str | None:
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"evidence", "comparison", "verdict"}:
        return None
    if any(not isinstance(value[key], str) or not value[key].strip()
           for key in ("evidence", "comparison")):
        return None
    verdict = value["verdict"]
    return verdict if isinstance(verdict, str) and verdict in {"T", "F", "X"} else None


def parse_reasoned_verdict(response: str) -> str | None:
    # Accept explicit verdict lines wherever they occur, including Markdown.
    # Never infer a label from explanatory prose or pick between disagreements.
    if response.strip().upper() in {"T", "F", "X"}:
        return response.strip().upper()
    lines = [re.sub(r"[*_`]", "", line).strip() for line in response.splitlines()]
    declarations = [line for line in lines if re.match(r"VERDICT\s*:", line, re.I)]
    matches = [VERDICT_PATTERN.fullmatch(line) for line in declarations]
    if not matches or any(match is None for match in matches):
        return None
    labels = {match.group(1).upper() for match in matches}
    return next(iter(labels)) if len(labels) == 1 else None


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
    response_format: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if response_format is not None:
        payload["response_format"] = response_format

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
    reasoned: bool = False,
    progress: bool = False,
    structured: bool = False,
) -> tuple[int, int]:
    if structured and reasoned:
        raise ValueError("Choose one verdict format")
    max_tokens = 512 if structured else (256 if reasoned else 4)
    parameters = {"temperature": 0, "max_tokens": max_tokens, "seed": seed}
    if structured:
        parameters["response_format"] = VERDICT_FORMAT
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
            != parameters
            or row.get("reasoned_verdict", False) != reasoned
            or row.get("structured_verdict", False) != structured
            or (structured and (
                not isinstance(row.get("model_response"), str)
                or parse_structured_verdict(row["model_response"]) is None
                or parse_structured_verdict(row["model_response"]) != row.get("raw_output")
            ))
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
            max_tokens=max_tokens,
            response_format=VERDICT_FORMAT if structured else None,
        )
        structured_label = parse_structured_verdict(raw_output) if structured else None
        if structured and structured_label is None:
            raise BaselineError(
                f"Invalid structured verdict for {example['id']}; response: {raw_output!r}. "
                "Completed rows are saved; repeat the command to retry this row."
            )
        prediction = {
            "id": example["id"],
            "raw_output": (parse_reasoned_verdict(raw_output) or raw_output)
            if reasoned else raw_output,
            "model": model,
            "prompt_sha256": hashlib.sha256(
                str(example["prompt"]).encode("utf-8")
            ).hexdigest(),
            "latency_seconds": round(time.perf_counter() - started, 6),
            "request_parameters": parameters,
            "usage": usage,
        }
        if reasoned:
            prediction["reasoned_verdict"] = True
            prediction["model_response"] = raw_output
            prediction["verdict_valid"] = parse_reasoned_verdict(raw_output) is not None
        if structured:
            prediction["raw_output"] = structured_label
            prediction["structured_verdict"] = True
            prediction["model_response"] = raw_output
        new_rows.append(prediction)
        write_jsonl(output_path, [*cached, *new_rows])
        if progress:
            display = prediction["raw_output"]
            if reasoned and not prediction["verdict_valid"]:
                display = "INVALID verdict (full response saved)"
            print(
                f"Saved {len(cached) + len(new_rows)}/{len(examples)}: "
                f"{example['id']} -> {display!r} "
                f"({prediction['latency_seconds']:.1f}s)",
                flush=True,
            )
    return len(new_rows), len(cached) + len(new_rows)


def repair_reasoned_predictions(
    examples_path: Path, predictions_path: Path, output_path: Path,
) -> dict[str, int]:
    """Reparse explicit verdicts from saved model responses without new inference."""
    if output_path.resolve() in {examples_path.resolve(), predictions_path.resolve()}:
        raise ValueError("Choose a new output path")
    examples = _load_examples(examples_path)
    predictions = _read_existing(predictions_path)
    ids = {row["id"] for row in examples}
    if len(ids) != len(examples) or {row["id"] for row in predictions} != ids:
        raise ValueError("Prediction IDs do not match examples")
    repaired = []
    counts = {"examples": len(predictions), "valid_verdicts": 0, "invalid_verdicts": 0}
    for row in predictions:
        response = row.get("model_response")
        if row.get("reasoned_verdict") is not True or not isinstance(response, str):
            raise ValueError(f"Missing saved reasoned response for {row['id']}")
        label = parse_reasoned_verdict(response)
        counts["valid_verdicts" if label else "invalid_verdicts"] += 1
        repaired.append({
            **row,
            "previous_raw_output": row.get("raw_output"),
            "raw_output": label if label else response,
            "verdict_valid": label is not None,
            "verdict_parser": "explicit-line-v2",
        })
    write_jsonl(output_path, repaired)
    return counts
