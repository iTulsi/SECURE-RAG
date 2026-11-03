"""Collect pinned 2025 CVEs and build a transparent structured-fact test draft."""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .benchmark_data import digest
from .claim_study import make_request, passages_for, ranked_passages, read_study
from .dataset import sha256_file
from .evaluation import write_metrics
from .pilot import write_jsonl
from .retrieval import CONTEXT_PREFIX, QUESTION_SUFFIX

CVE_REVISION = "6125463712241a8a7cca97b445c85415f26a8b09"
KEV_REVISION = "7ccba8e1223ccec82d12c7736bf3fae11e5bee86"
KEV_URL = f"https://raw.githubusercontent.com/cisagov/kev-data/{KEV_REVISION}/known_exploited_vulnerabilities.json"
SEED = 20261102


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "SECURE-RAG-research/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read(6_000_001)
    if len(raw) > 6_000_000:
        raise ValueError("Source exceeds the 6 MB input limit")
    return raw


def record_url(cve: str) -> str:
    if not re.fullmatch(r"CVE-2025-\d{4,}", cve):
        raise ValueError("Expected a 2025 CVE identifier")
    bucket = int(cve.rsplit("-", 1)[1]) // 1000
    return f"https://raw.githubusercontent.com/CVEProject/cvelistV5/{CVE_REVISION}/cves/2025/{bucket}xxx/{cve}.json"


def fact_values(record: dict) -> tuple[str, str, int]:
    metadata = record.get("cveMetadata", {})
    if metadata.get("state") != "PUBLISHED" or not metadata.get("datePublished", "").startswith("2025-"):
        raise ValueError("Not a published 2025 record")
    assessments = []
    for index, item in enumerate(record.get("containers", {}).get("cna", {}).get("metrics", [])):
        value = item.get("cvssV3_1")
        if not isinstance(value, dict):
            continue
        severity, interaction = value.get("baseSeverity"), value.get("userInteraction")
        if (value.get("version") != "3.1" or severity not in {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
                or interaction not in {"NONE", "REQUIRED"}):
            raise ValueError("Incomplete CNA CVSS v3.1 assessment")
        assessments.append((severity, interaction, index))
    if not assessments or len({v[:2] for v in assessments}) != 1:
        raise ValueError("Missing or conflicting CNA CVSS v3.1 facts")
    return assessments[0]


def record_cases(record: dict, url: str) -> tuple[list[dict], list[dict]]:
    severity, interaction, index = fact_values(record)
    cve = record["cveMetadata"]["cveId"]
    if not re.fullmatch(r"CVE-2025-\d{4,}", cve):
        raise ValueError("Record identifier is not a 2025 CVE")
    levels = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    wrong_severity = levels[(levels.index(severity) + 1) % len(levels)]
    opposite_ui = "NONE" if interaction == "REQUIRED" else "REQUIRED"
    facts = [("baseSeverity", severity, severity), ("baseSeverity", severity, wrong_severity),
             ("userInteraction", interaction, interaction), ("userInteraction", interaction, opposite_ui)]
    context_prompt = CONTEXT_PREFIX + json.dumps(record, ensure_ascii=False) + QUESTION_SUFFIX
    example = SimpleNamespace(id=cve, prompt=context_prompt, source_url=url)
    pool = passages_for(example)
    cases, labels = [], []
    for field, actual, proposed in facts:
        if field == "baseSeverity":
            claim = f"The CNA's CVSS v3.1 base severity for {cve} is {proposed}."
        else:
            phrase = "required" if proposed == "REQUIRED" else "not required"
            claim = f"In the CNA's CVSS v3.1 assessment of {cve}, user interaction is {phrase}."
        identifier = f"{cve.lower()}-{digest(claim)[:12]}"
        official = context_prompt + " " + claim + " Return only T, F or X, where X means insufficient evidence."
        case = {"id": identifier, "pair_id": cve, "origin_id": cve, "group_id": cve,
                "condition": "fresh_cve", "target_cve_id": cve, "claim": claim,
                "official_prompt": official, "passages": ranked_passages(claim, pool),
                "arms": ["official", "format_control", "cited"]}
        for stage in case["arms"]:
            make_request(case, stage, None, {"model": "preflight", "seed": SEED, "context_window": 32768})
        cases.append(case)
        labels.append({"id": identifier, "gold_label": "T" if actual == proposed else "F",
                       "basis": f"/containers/cna/metrics/{index}/cvssV3_1/{field}",
                       "source_value": actual, "claim_value": proposed,
                       "label_origin": "template-generated from explicit source field; not human annotated"})
    return cases, labels


def collect(output_dir: Path, source_count: int = 40) -> dict:
    if source_count < 2:
        raise ValueError("At least two source groups are needed")
    path = output_dir / "manifest.json"
    if path.exists():
        _, _, manifest = read_study(output_dir)
        if manifest["groups"] != source_count:
            raise ValueError("Existing collection has a different source count")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "kev_catalog.json"
    if not catalog_path.exists():
        catalog_path.write_bytes(fetch_bytes(KEV_URL))
    catalog = json.loads(catalog_path.read_text())
    ids = sorted({v["cveID"] for v in catalog["vulnerabilities"]
                  if re.fullmatch(r"CVE-2025-\d{4,}", v["cveID"])}, key=lambda c: digest([SEED, c]))
    records_dir = output_dir / "records"
    records_dir.mkdir(exist_ok=True)

    def download(cve: str) -> tuple[str, Path | None, str | None]:
        local = records_dir / f"{cve}.json"
        try:
            if not local.exists():
                local.write_bytes(fetch_bytes(record_url(cve)))
            return cve, local, None
        except (OSError, ValueError, urllib.error.URLError) as error:
            return cve, None, str(error)

    selected, skipped, cases, labels, review = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(ids), 4):
            for cve, local, error in pool.map(download, ids[offset:offset+4]):
                if len(selected) >= source_count:
                    continue
                if error:
                    raise RuntimeError(f"Source download failed for {cve}: {error}. Rerun to resume cached sources.")
                try:
                    record = json.loads(local.read_text())
                    if record.get("cveMetadata", {}).get("cveId") != cve:
                        raise ValueError("CVE identity mismatch")
                    generated, answers = record_cases(record, record_url(cve))
                except (ValueError, KeyError, TypeError) as error:
                    skipped.append({"cve": cve, "reason": str(error)})
                    continue
                cases.extend(generated)
                labels.extend(answers)
                selected.append({"cve": cve, "url": record_url(cve), "sha256": sha256_file(local)})
                review.append({"cve": cve, "source_file": f"records/{cve}.json",
                               "claims": [{"claim": c["claim"], **a} for c, a in zip(generated, answers, strict=True)],
                               "reviewer": "", "labels_confirmed": None, "notes": ""})
                print(f"Collected {len(selected)}/{source_count}: {cve}", flush=True)
            if len(selected) >= source_count:
                break
    if len(selected) != source_count:
        raise ValueError(f"Only {len(selected)} eligible source records; no partial test was prepared")
    cases.sort(key=lambda c: (digest([SEED, c["pair_id"]]), digest(c["id"])))
    manifest = {
        "schema_version": 1, "study": "cited-claims-fresh-v3", "split": "external_structured_fact_test",
        "source_manifest_sha256": digest(selected), "source_count": source_count,
        "groups": source_count, "case_count": len(cases), "expected_calls": len(cases) * 3,
        "condition_counts": {"fresh_cve": len(cases)}, "seed": SEED,
        "cve_revision": CVE_REVISION, "kev_revision": KEV_REVISION,
        "catalog_url": KEV_URL, "catalog_sha256": sha256_file(catalog_path),
        "selected_sources": selected, "excluded_sources": skipped,
        "gold_labels": dict(Counter(r["gold_label"] for r in labels)),
        "inputs_sha256": digest(cases), "labels_sha256": digest(labels),
        "label_status": "machine-derived controlled facts; human review pending",
        "primary_endpoint": "verdict_accuracy", "chunk_chars": 1200, "top_k": 3,
        "scope": "2025 KEV-listed CVEs with short, nonconflicting CNA CVSS v3.1 severity and user-interaction facts; not a representative natural-claim or all-task benchmark.",
    }
    write_jsonl(output_dir / "inputs.jsonl", cases)
    write_jsonl(output_dir / "labels.jsonl", labels)
    write_jsonl(output_dir / "label_review.jsonl", review)
    write_metrics(path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fresh_claims"))
    parser.add_argument("--sources", type=int, default=40)
    args = parser.parse_args(argv)
    try:
        manifest = collect(args.output_dir, args.sources)
        print(json.dumps({k: v for k, v in manifest.items() if k not in {"selected_sources", "excluded_sources"}}, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
