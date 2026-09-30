"""Normalize official evidence files without importing benchmark questions or labels."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from .dataset import sha256_file
from .pilot import write_jsonl
from .retrieval import chunk_context

ENTITY_PATTERN = re.compile(
    r"\b(?:CVE-\d{4}-\d{4,}|CWE-\d+|T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE
)
SOURCES = frozenset({"cve", "nvd", "kev", "cwe", "attack"})


def entity_ids(text: str) -> list[str]:
    return sorted({match.upper() for match in ENTITY_PATTERN.findall(text)})


def _records(source: str, path: Path) -> list[tuple[str, str, object]]:
    """Return (entity id, canonical URL, evidence) from one native source file."""
    if source == "cwe":
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = [name for name in archive.namelist() if name.endswith(".xml")]
                if len(names) != 1 or archive.getinfo(names[0]).file_size > 100_000_000:
                    raise ValueError(
                        "CWE archive must contain one XML file under 100 MB"
                    )
                raw = archive.read(names[0])
        else:
            raw = path.read_bytes()
        xml = raw.decode("utf-8-sig")
        if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
            raise ValueError("CWE XML must not declare entities or a DTD")
        root = ET.fromstring(xml)
        records = []
        for node in root.findall(".//{*}Weakness"):
            identifier = f"CWE-{node.attrib['ID']}"
            evidence = {"id": identifier, "name": node.attrib["Name"]}
            for tag in ("Description", "Extended_Description"):
                child = node.find(f"{{*}}{tag}")
                if child is not None:
                    evidence[tag] = " ".join(" ".join(child.itertext()).split())
            records.append(
                (
                    identifier,
                    f"https://cwe.mitre.org/data/definitions/{node.attrib['ID']}.html",
                    evidence,
                )
            )
        return records

    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    if source == "cve":
        metadata = body["cveMetadata"]
        if metadata["state"] != "PUBLISHED":
            return []
        identifier = metadata["cveId"].upper()
        return [(identifier, f"https://www.cve.org/CVERecord?id={identifier}", body)]
    if source == "nvd":
        return [
            (
                item["cve"]["id"].upper(),
                f"https://nvd.nist.gov/vuln/detail/{item['cve']['id']}",
                item["cve"],
            )
            for item in body["vulnerabilities"]
        ]
    if source == "kev":
        return [
            (
                item["cveID"].upper(),
                "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
                item,
            )
            for item in body["vulnerabilities"]
        ]
    if source == "attack":
        records = []
        for item in body["objects"]:
            if (
                item.get("type") != "attack-pattern"
                or item.get("revoked")
                or item.get("x_mitre_deprecated")
            ):
                continue
            references = [
                ref
                for ref in item.get("external_references", [])
                if ref.get("source_name") == "mitre-attack" and ref.get("external_id")
            ]
            if not references:
                continue
            ref = references[0]
            evidence = {
                key: item[key]
                for key in ("name", "description", "modified", "kill_chain_phases")
                if key in item
            }
            records.append((ref["external_id"], ref["url"], evidence))
        return records
    raise ValueError(f"Unsupported source: {source}")


def build_corpus(
    inputs: list[tuple[str, Path]], output: Path, max_chars: int = 1200
) -> dict[str, object]:
    if not inputs:
        raise ValueError("Provide at least one evidence source")
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if output.resolve() in {path.resolve() for _, path in inputs}:
        raise ValueError("Corpus output must differ from input files")
    rows: dict[str, dict[str, object]] = {}
    snapshots = []
    for source, path in inputs:
        if source not in SOURCES:
            raise ValueError(f"Unsupported source: {source}")
        try:
            records = _records(source, path)
        except (
            KeyError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
            ET.ParseError,
        ) as error:
            raise ValueError(f"Invalid {source} evidence schema in {path}") from error
        snapshots.append(
            {
                "source": source,
                "file": path.name,
                "sha256": sha256_file(path),
                "records": len(records),
            }
        )
        for identifier, url, evidence in records:
            ids = entity_ids(identifier)
            if ids != [identifier] or not identifier.startswith(
                {
                    "cve": "CVE-",
                    "nvd": "CVE-",
                    "kev": "CVE-",
                    "cwe": "CWE-",
                    "attack": "T",
                }[source]
            ):
                raise ValueError(f"Invalid {source} entity: {identifier}")
            record_hash = hashlib.sha256(
                json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            for index, text in enumerate(chunk_context(evidence, max_chars)):
                # Include origin and entity in the text so rankers never lose identity.
                text = f"Source: {source}; Entity: {identifier}\n{text}"
                key = hashlib.sha256(
                    f"{source}\0{identifier}\0{text}".encode()
                ).hexdigest()
                rows[key] = {
                    "id": key,
                    "source": source,
                    "entity_id": identifier,
                    "source_url": url,
                    "record_sha256": record_hash,
                    "chunk_index": index,
                    "text": text,
                }
    if not rows:
        raise ValueError("No supported evidence records found")
    write_jsonl(output, [rows[key] for key in sorted(rows)])
    manifest = {
        "schema_version": 1,
        "corpus_sha256": sha256_file(output),
        "chunks": len(rows),
        "source_chunks": dict(
            sorted(Counter(str(row["source"]) for row in rows.values()).items())
        ),
        "snapshots": snapshots,
        "protocol": "Evidence only; no benchmark questions or answer labels. Current snapshots may differ from SECURE's historical context.",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_corpus(path: Path) -> list[dict[str, object]]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or any(
                not isinstance(row.get(key), str) or not row[key]
                for key in ("id", "source", "entity_id", "source_url", "text")
            )
            or row["source"] not in SOURCES
            or row["id"] in seen
            or entity_ids(row["entity_id"]) != [row["entity_id"]]
            or not row["source_url"].startswith("https://")
        ):
            raise ValueError(f"Invalid corpus record at line {number}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("Corpus is empty")
    return rows
