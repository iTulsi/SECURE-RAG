from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

from .pilot import write_jsonl
from .runner import BaselineError, _load_examples


CONTEXT_PREFIX = "You are given the following JSON data as context: "
QUESTION_SUFFIX = "  Based on the context, you have to analyze the following statement:"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def extract_context(prompt: str) -> object | None:
    """Return the JSON context embedded in an official KCV prompt, if present."""
    start = prompt.find(CONTEXT_PREFIX)
    if start < 0:
        return None
    payload = prompt[start + len(CONTEXT_PREFIX) :]
    try:
        context, end = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Prompt contains an invalid JSON context") from error
    if QUESTION_SUFFIX not in payload[end:]:
        raise ValueError("Prompt JSON is not followed by the expected question text")
    return context


def _flatten(value: object, path: str = "root") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten(child, f"{path}[{index}]")
    else:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        yield f"{path} = {rendered}"


def chunk_context(context: object, max_chars: int = 1200) -> list[str]:
    """Flatten JSON into path-aware chunks without dropping oversized values."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for line in _flatten(context):
        pieces = [line[index : index + max_chars] for index in range(0, len(line), max_chars)]
        for piece in pieces:
            added = len(piece) + (1 if current else 0)
            if current and current_length + added > max_chars:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
            current.append(piece)
            current_length += len(piece) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _request_embeddings(
    base_url: str,
    api_key: str | None,
    model: str,
    texts: list[str],
    timeout_seconds: float,
) -> list[list[float]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise BaselineError(f"Embedding request failed: {error}") from error
    try:
        ordered = sorted(body["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in ordered]
    except (KeyError, TypeError) as error:
        raise BaselineError("Endpoint response lacks indexed embedding data") from error
    if len(vectors) != len(texts) or any(
        not isinstance(vector, list)
        or not vector
        or any(not isinstance(value, (int, float)) for value in vector)
        for vector in vectors
    ):
        raise BaselineError("Endpoint returned invalid embedding vectors")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise BaselineError("Endpoint returned inconsistent embedding dimensions")
    return [[float(value) for value in vector] for vector in vectors]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def bm25_scores(
    query: str,
    documents: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Score documents with Okapi BM25 using deterministic lexical tokens."""
    if k1 <= 0:
        raise ValueError("k1 must be positive")
    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1")
    if not documents:
        return []

    tokenized = [TOKEN_PATTERN.findall(document.lower()) for document in documents]
    query_terms = set(TOKEN_PATTERN.findall(query.lower()))
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    document_frequency = Counter(
        term for tokens in tokenized for term in set(tokens) if term in query_terms
    )
    scores: list[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length_ratio = 0.0 if average_length == 0 else len(tokens) / average_length
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1
                + (len(documents) - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * length_ratio)
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def rank_hybrid(
    question: str,
    chunks: list[str],
    embeddings: dict[str, list[float]],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, object]]:
    """Fuse dense and BM25 rankings with reciprocal rank fusion."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if not chunks:
        return []

    dense_scores = [
        _cosine(embeddings[question], embeddings[chunk]) for chunk in chunks
    ]
    lexical_scores = bm25_scores(question, chunks)
    dense_order = sorted(
        range(len(chunks)), key=lambda index: (-dense_scores[index], chunks[index])
    )
    lexical_order = sorted(
        range(len(chunks)), key=lambda index: (-lexical_scores[index], chunks[index])
    )
    dense_rank = {index: rank for rank, index in enumerate(dense_order, start=1)}
    lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}
    fused_scores = {
        index: 1 / (rrf_k + dense_rank[index]) + 1 / (rrf_k + lexical_rank[index])
        for index in range(len(chunks))
    }
    fused_order = sorted(
        range(len(chunks)),
        key=lambda index: (
            -fused_scores[index],
            -dense_scores[index],
            chunks[index],
        ),
    )[:top_k]
    return [
        {
            "rank": rank,
            "score": fused_scores[index],
            "dense_score": dense_scores[index],
            "bm25_score": lexical_scores[index],
            "dense_rank": dense_rank[index],
            "bm25_rank": lexical_rank[index],
            "text": chunks[index],
        }
        for rank, index in enumerate(fused_order, start=1)
    ]


def _text_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()


def _load_embedding_cache(path: Path, model: str) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    cached: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                key = row["key"]
                vector = row["embedding"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid embedding cache at line {line_number}") from error
            if row.get("model") != model:
                raise ValueError("Embedding cache model does not match requested model")
            if not isinstance(key, str) or key in cached or not isinstance(vector, list):
                raise ValueError(f"Invalid or duplicate embedding at line {line_number}")
            cached[key] = [float(value) for value in vector]
    return cached


def _embed_with_cache(
    texts: list[str],
    cache_path: Path,
    base_url: str,
    model: str,
    timeout_seconds: float,
    batch_size: int,
) -> dict[str, list[float]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    cached = _load_embedding_cache(cache_path, model)
    unique = list(dict.fromkeys(texts))
    missing = [text for text in unique if _text_key(model, text) not in cached]
    api_key = os.environ.get("SECURE_RAG_API_KEY")
    cache_rows = [
        {"key": key, "model": model, "embedding": vector}
        for key, vector in cached.items()
    ]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = _request_embeddings(base_url, api_key, model, batch, timeout_seconds)
        for text, vector in zip(batch, vectors, strict=True):
            key = _text_key(model, text)
            cached[key] = vector
            cache_rows.append({"key": key, "model": model, "embedding": vector})
        write_jsonl(cache_path, cache_rows)
    return {text: cached[_text_key(model, text)] for text in unique}


def _retrieval_prompt(question: str, evidence: list[str]) -> str:
    context = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(evidence, 1))
    if not context:
        context = "No evidence was provided."
    return (
        "Use only the evidence below to evaluate the statement.\n\n"
        f"Evidence:\n{context}\n\nStatement: {question}\n\n"
        "Return T if the statement is supported, F if it is contradicted, or X if "
        "the evidence is insufficient. Provide only T, F, or X."
    )


def _chunk_examples(
    examples: list[dict[str, object]], max_chunk_chars: int
) -> tuple[dict[str, list[str]], list[str]]:
    chunks_by_id: dict[str, list[str]] = {}
    texts: list[str] = []
    for example in examples:
        identifier = str(example["id"])
        context = extract_context(str(example["prompt"]))
        chunks = [] if context is None else chunk_context(context, max_chunk_chars)
        chunks_by_id[identifier] = chunks
        texts.extend(chunks)
        if chunks:
            texts.append(str(example["question"]))
    return chunks_by_id, texts


def prepare_dense_examples(
    examples_path: Path,
    output_path: Path,
    retrieval_path: Path,
    embedding_cache_path: Path,
    base_url: str,
    model: str,
    timeout_seconds: float,
    batch_size: int,
    top_k: int,
    max_chunk_chars: int,
) -> tuple[int, float]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    examples = _load_examples(examples_path)
    chunks_by_id, texts = _chunk_examples(examples, max_chunk_chars)

    started = time.perf_counter()
    embeddings = _embed_with_cache(
        texts=texts,
        cache_path=embedding_cache_path,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
    )
    retrieval_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    retrieved_count = 0
    for example in examples:
        identifier = str(example["id"])
        question = str(example["question"])
        chunks = chunks_by_id[identifier]
        ranked: list[tuple[float, str]] = []
        if chunks:
            ranked = sorted(
                ((_cosine(embeddings[question], embeddings[chunk]), chunk) for chunk in chunks),
                key=lambda item: (-item[0], item[1]),
            )[:top_k]
        evidence = [chunk for _, chunk in ranked]
        retrieved_count += len(evidence)
        output_row = dict(example)
        output_row["prompt"] = _retrieval_prompt(question, evidence)
        output_rows.append(output_row)
        retrieval_rows.append(
            {
                "id": identifier,
                "embedding_model": model,
                "candidate_count": len(chunks),
                "retrieved": [
                    {"rank": index, "score": score, "text": chunk}
                    for index, (score, chunk) in enumerate(ranked, start=1)
                ],
            }
        )
    write_jsonl(output_path, output_rows)
    write_jsonl(retrieval_path, retrieval_rows)
    return retrieved_count, round(time.perf_counter() - started, 6)


def prepare_hybrid_examples(
    examples_path: Path,
    output_path: Path,
    retrieval_path: Path,
    embedding_cache_path: Path,
    base_url: str,
    model: str,
    timeout_seconds: float,
    batch_size: int,
    top_k: int,
    max_chunk_chars: int,
    rrf_k: int,
) -> tuple[int, float]:
    """Prepare E2 prompts using BM25 and dense reciprocal-rank fusion."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    examples = _load_examples(examples_path)
    chunks_by_id, texts = _chunk_examples(examples, max_chunk_chars)
    started = time.perf_counter()
    embeddings = _embed_with_cache(
        texts=texts,
        cache_path=embedding_cache_path,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
    )
    retrieval_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    retrieved_count = 0
    for example in examples:
        identifier = str(example["id"])
        question = str(example["question"])
        chunks = chunks_by_id[identifier]
        pool = rank_hybrid(question, chunks, embeddings, max(top_k, 10), rrf_k)
        ranked = pool[:top_k]
        evidence = [str(item["text"]) for item in ranked]
        retrieved_count += len(evidence)
        output_row = dict(example)
        output_row["prompt"] = _retrieval_prompt(question, evidence)
        output_rows.append(output_row)
        retrieval_rows.append(
            {
                "id": identifier,
                "retrieval_strategy": "bm25+dense-rrf",
                "embedding_model": model,
                "rrf_k": rrf_k,
                "candidate_count": len(chunks),
                "retrieved": ranked,
                "candidate_pool": pool,
            }
        )
    write_jsonl(output_path, output_rows)
    write_jsonl(retrieval_path, retrieval_rows)
    return retrieved_count, round(time.perf_counter() - started, 6)
