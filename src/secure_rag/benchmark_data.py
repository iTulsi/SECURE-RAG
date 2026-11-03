"""Pinned, source-grouped preparation of all six official SECURE tasks."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from .dataset import DatasetError, sha256_file
from .evaluation import _read_jsonl, write_metrics
from .pilot import write_jsonl

TASKS = ("MAET", "CWET", "KCV", "VOOD", "RERT", "CPST")
UPSTREAM = "a2412e8ab4b6051ba7381d4f590cb48d9743c7c8"
OFFICIAL_SHA256 = {
    "MAET": "3f53a7f74e701f6f5ae8858e4803d63b09159b5dfdafa1a15cfd5165094f80e8",
    "CWET": "1c22ef2e434490afb7bdbf7c0221d163ebbcc88fd4d58c1b3eb99619434f14f3",
    "KCV": "9c9f3e1e376b87819918c62571bbc20f6be98f96fbfd1f8d32424f3692c0732b",
    "VOOD": "3449c07ade1d313a5f669a84d984ac00368e2dbd112d08a8ac78699c8ee5bbde",
    "RERT": "58cbdad2bebc7438ec51b3fdd1d57907cdce9ca2ae1082a935c387c631e1fa94",
    "CPST": "3d0bcb175d4725e8e265dfc7e1740f3e2f3c966d3b2c0b38c892524a53968f06",
}
FIELDS = {
    "MAET": ("URL", "Prompt", "Question", "Option A", "Option B", "Option C", "Option D", "Correct Answer"),
    "CWET": ("URL", "Prompt", "Question", "Option A", "Option B", "Option C", "Option D", "Correct Answer"),
    "KCV": ("URL", "Prompt", "Question", "Correct Answer"),
    "VOOD": ("URL", "Prompt", "Question", "Correct Answer"),
    "RERT": ("source-url", "Prompt", "Correct Answer", "vulnerability-overview"),
    "CPST": ("Prompt", "CVSS v3 Vector String", "Correct Answer"),
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def load_benchmark(data_dir: Path) -> tuple[list[dict], dict]:
    rows, audit = [], {}
    for task in TASKS:
        path = data_dir / f"SECURE - {task}.tsv"
        accepted, blanks, prompts = [], [], defaultdict(list)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if tuple(reader.fieldnames or ()) != FIELDS[task]:
                raise DatasetError(f"Unexpected {task} columns: {reader.fieldnames}")
            for index, row in enumerate(reader):
                if None in row or any(value is None for value in row.values()):
                    raise DatasetError(f"Malformed {task} row {index}")
                if not any(value.strip() for value in row.values()):
                    blanks.append(index)
                    continue
                prompt, answer = row["Prompt"].strip(), row["Correct Answer"].strip()
                if not prompt or not answer:
                    raise DatasetError(f"Missing prompt/label in {task} row {index}")
                if task in {"MAET", "CWET", "KCV", "VOOD"}:
                    allowed = {"A", "B", "C", "D"} if task in {"MAET", "CWET"} else {"T", "F", "X"}
                    if answer not in allowed or (task == "VOOD" and answer != "X"):
                        raise DatasetError(f"Invalid {task} label at row {index}")
                if task == "CPST":
                    score = float(answer)
                    if not math.isfinite(score) or not 0 <= score <= 10:
                        raise DatasetError(f"Invalid CVSS score at row {index}")
                source = row.get("URL", row.get("source-url", "")).strip()
                if task != "CPST" and (urlsplit(source).scheme not in {"http", "https"}
                                        or not urlsplit(source).netloc):
                    raise DatasetError(f"Invalid source in {task} row {index}")
                item = {
                    "id": f"{task.lower()}-{index:04d}", "task": task,
                    "row_index": index, "source_url": source, "prompt": prompt,
                    "gold_label": answer,
                    "question": row.get("Question", row.get("CVSS v3 Vector String", "")),
                }
                accepted.append(item)
                prompts[prompt].append(item)
        if not accepted:
            raise DatasetError(f"Empty task {task}")
        audit[task] = {
            "raw_rows": len(accepted) + len(blanks), "scorable_rows": len(accepted),
            "blank_row_indices": blanks, "sha256": sha256_file(path),
            "source_count": len({x["source_url"] for x in accepted}) if task != "CPST" else None,
            "duplicate_prompt_excess": sum(len(v) - 1 for v in prompts.values()),
            "conflicting_duplicate_ids": [
                [x["id"] for x in values] for values in prompts.values()
                if len({x["gold_label"] for x in values}) > 1
            ],
            "label_counts": dict(Counter(x["gold_label"] for x in accepted))
            if task != "RERT" else None,
        }
        rows.extend(accepted)
    kcv = [r for r in rows if r["task"] == "KCV"]
    vood = [r for r in rows if r["task"] == "VOOD"]
    if [(r["source_url"], r["question"]) for r in kcv] != [
        (r["source_url"], r["question"]) for r in vood
    ]:
        raise DatasetError("Official KCV/VOOD pairs are not aligned")
    return rows, audit


def source_key(url: str) -> str:
    parsed = urlsplit(url)
    return parsed.netloc.lower().removeprefix("www.") + parsed.path.rstrip("/")


def assign_groups(rows: list[dict]) -> dict[str, str]:
    """Join shared source URLs, CVEs, exact prompts and duplicate CVSS vectors."""
    parent = {r["id"]: r["id"] for r in rows}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for row in rows:
        tokens = {"prompt:" + digest(row["prompt"])}
        if row["source_url"]:
            tokens.add("url:" + source_key(row["source_url"]))
        if row["task"] in {"KCV", "VOOD", "RERT"}:
            tokens.update(re.findall(r"CVE-\d{4}-\d{4,}", row["prompt"], re.I))
        if row["task"] == "CPST":
            tokens.add("vector:" + row["question"].strip())
        for token in sorted(tokens):
            if token in owners:
                a, b = root(row["id"]), root(owners[token])
                parent[max(a, b)] = min(a, b)
            else:
                owners[token] = row["id"]
    members = defaultdict(list)
    for row in rows:
        members[root(row["id"])].append(row["id"])
    return {identifier: digest(sorted(ids))[:24]
            for ids in members.values() for identifier in ids}


def prepare_benchmark(data_dir: Path, output_dir: Path, pilot_manifest: Path,
                      holdout_manifest: Path, seed: int = 20261031) -> dict:
    rows, audit = load_benchmark(data_dir)
    if {task: value["sha256"] for task, value in audit.items()} != OFFICIAL_SHA256:
        raise DatasetError("Dataset bytes differ from the pinned official revision; do not silently change the benchmark")
    historical = [json.loads(p.read_text()) for p in (pilot_manifest, holdout_manifest)]
    for manifest in historical:
        for task, expected in manifest["dataset_sha256"].items():
            if audit[task]["sha256"] != expected:
                raise DatasetError("Historical source manifest does not match input data")
    groups = assign_groups(rows)
    pilot_sources, old_holdout_sources = [
        {source_key(s) for s in m["selected_sources"]} for m in historical
    ]
    pilot_groups = {groups[r["id"]] for r in rows
                    if source_key(r["source_url"]) in pilot_sources}
    holdout_groups = {groups[r["id"]] for r in rows
                      if source_key(r["source_url"]) in old_holdout_sources}
    for row in rows:
        group = groups[row["id"]]
        fraction = int(digest([seed, group])[:16], 16) / 2**64
        split = "development" if fraction < .2 else "validation" if fraction < .4 else "test"
        if group in holdout_groups:
            split = "test"
        if group in pilot_groups:
            split = "development"
        row.update(group_id=group, split=split)
    split_counts = {split: dict(Counter(r["task"] for r in rows if r["split"] == split))
                    for split in ("development", "validation", "test")}
    assignments = defaultdict(list)
    for row in rows:
        assignments[row["group_id"], row["split"]].append(row["id"])
    manifest = {
        "schema_version": 1, "upstream_repository": "https://github.com/aiforsec/SECURE",
        "upstream_commit": UPSTREAM, "seed": seed, "audit": audit,
        "scorable_examples": len(rows), "source_groups": len(set(groups.values())),
        "split_task_counts": split_counts,
        "historical_holdout_groups_also_linked_to_pilot": len(pilot_groups & holdout_groups),
        "training": "No fine-tuning; development is for method design, validation for selection.",
        "grouping": "Connected source URLs, CVE mentions (KCV/VOOD/RERT), exact prompts, CVSS vectors.",
        "assignments": [{"group_id": group, "split": split, "ids": sorted(ids)}
                        for (group, split), ids in sorted(assignments.items())],
    }
    inputs = [{k: r[k] for k in ("id", "task", "group_id", "split", "prompt")}
              for r in rows]
    labels = [{k: r[k] for k in ("id", "gold_label")} for r in rows]
    manifest.update(inputs_sha256=digest(inputs), labels_sha256=digest(labels))
    existing = output_dir / "manifest.json"
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise ValueError("Prepared dataset differs; choose a new output directory")
    write_jsonl(output_dir / "inputs.jsonl", inputs)
    write_jsonl(output_dir / "labels.jsonl", labels)
    write_metrics(existing, manifest)
    return manifest


def read_prepared(data_dir: Path) -> tuple[list[dict], dict[str, str], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rows = _read_jsonl(data_dir / "inputs.jsonl")
    labels = _read_jsonl(data_dir / "labels.jsonl")
    if digest(rows) != manifest["inputs_sha256"] or digest(labels) != manifest["labels_sha256"]:
        raise ValueError("Prepared data hashes do not match the manifest")
    if len({r["id"] for r in rows}) != len(rows) or len({r["id"] for r in labels}) != len(labels):
        raise ValueError("Duplicate prepared IDs")
    if {r["id"] for r in rows} != {r["id"] for r in labels}:
        raise ValueError("Input and label IDs differ")
    return rows, {r["id"]: r["gold_label"] for r in labels}, manifest
