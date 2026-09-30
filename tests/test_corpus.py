from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from secure_rag.corpus import build_corpus, load_corpus
from secure_rag.pilot import write_jsonl
from secure_rag.retrieval import prepare_multisource_examples


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def save(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def inputs(self):
        cve = self.save(
            "CVE-2024-0007.json",
            {
                "cveMetadata": {"cveId": "CVE-2024-0007", "state": "PUBLISHED"},
                "containers": {
                    "cna": {
                        "descriptions": [
                            {"value": "remote widget attack", "lang": "en"}
                        ]
                    }
                },
            },
        )
        nvd = self.save(
            "nvd.json",
            {
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2024-0007",
                            "descriptions": [{"value": "remote widget attack"}],
                        }
                    }
                ]
            },
        )
        kev = self.save(
            "kev.json",
            {
                "vulnerabilities": [
                    {
                        "cveID": "CVE-2024-0007",
                        "shortDescription": "remote widget attack",
                    }
                ]
            },
        )
        cwe = self.root / "cwe.zip"
        with zipfile.ZipFile(cwe, "w") as archive:
            archive.writestr(
                "cwe.xml",
                '<Weakness_Catalog xmlns="urn:test"><Weaknesses><Weakness ID="79" Name="XSS"><Description>remote widget attack</Description></Weakness></Weaknesses></Weakness_Catalog>',
            )
        attack = self.save(
            "attack.json",
            {
                "objects": [
                    {
                        "type": "attack-pattern",
                        "name": "remote widget attack",
                        "description": "execution",
                        "external_references": [
                            {
                                "source_name": "mitre-attack",
                                "external_id": "T1059",
                                "url": "https://attack.mitre.org/techniques/T1059/",
                            }
                        ],
                    },
                    {"type": "attack-pattern", "revoked": True, "name": "old"},
                ]
            },
        )
        return [
            ("cve", cve),
            ("nvd", nvd),
            ("kev", kev),
            ("cwe", cwe),
            ("attack", attack),
        ]

    def test_native_formats_provenance_and_deduplication(self):
        inputs = self.inputs()
        manifest = build_corpus(inputs + [inputs[0]], self.root / "corpus.jsonl")
        rows = load_corpus(self.root / "corpus.jsonl")
        self.assertEqual(
            set(manifest["source_chunks"]), {"cve", "nvd", "kev", "cwe", "attack"}
        )
        self.assertEqual(len(rows), len({row["id"] for row in rows}))
        self.assertTrue(all(len(row["record_sha256"]) == 64 for row in rows))
        self.assertTrue(all("gold_label" not in row for row in rows))

    def test_rejects_unsafe_xml_and_empty_source(self):
        path = self.root / "bad.xml"
        path.write_text('<!DOCTYPE root [<!ENTITY data "payload">]><root/>')
        with self.assertRaisesRegex(ValueError, "DTD"):
            build_corpus([("cwe", path)], self.root / "corpus.jsonl")
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_corpus([], self.root / "corpus.jsonl")

    def test_rejects_invalid_entity_and_corpus_duplicates(self):
        source = self.save("bad.json", {"vulnerabilities": [{"cveID": "not-a-cve"}]})
        with self.assertRaisesRegex(ValueError, "Invalid kev entity"):
            build_corpus([("kev", source)], self.root / "corpus.jsonl")
        build_corpus(self.inputs(), self.root / "corpus.jsonl")
        path = self.root / "corpus.jsonl"
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "Invalid corpus record"):
            load_corpus(path)

    def examples(self):
        return [
            {
                "id": "with",
                "task": "KCV",
                "gold_label": "T",
                "question": "remote widget attack CWE-79 T1059",
                "source_url": "https://example.org/CVE-2024-0007.json",
                "prompt": 'You are given the following JSON data as context: {"id":"CVE-2024-0007","weakness":"CWE-79","description":"remote widget attack"}  Based on the context, you have to analyze the following statement: x',
            },
            {
                "id": "without",
                "task": "VOOD",
                "gold_label": "X",
                "question": "remote widget attack CVE-2024-0007",
                "source_url": "https://example.org/CVE-2024-0007.json",
                "prompt": "No supplied context",
            },
        ]

    def run_retrieval(self, examples, **kwargs):
        build_corpus(self.inputs(), self.root / "corpus.jsonl")
        write_jsonl(self.root / "examples.jsonl", examples)
        count, _ = prepare_multisource_examples(
            self.root / "examples.jsonl",
            self.root / "corpus.jsonl",
            self.root / "out.jsonl",
            self.root / "retrieval.jsonl",
            top_k=10,
            **kwargs,
        )
        output = [
            json.loads(line)
            for line in (self.root / "out.jsonl").read_text().splitlines()
        ]
        audit = [
            json.loads(line)
            for line in (self.root / "retrieval.jsonl").read_text().splitlines()
        ]
        return count, output, audit

    def test_real_bm25_retrieves_all_sources_and_preserves_missing_evidence(self):
        _, output, audit = self.run_retrieval(self.examples())
        self.assertEqual(
            {item["source"] for item in audit[0]["retrieved"]},
            {"secure-context", "cve", "nvd", "kev", "cwe", "attack"},
        )
        self.assertEqual(audit[1]["candidate_count"], 0)
        self.assertEqual(audit[1]["retrieved"], [])
        self.assertIn("No evidence was provided", output[1]["prompt"])
        self.assertEqual([row["gold_label"] for row in output], ["T", "X"])

    def test_routing_does_not_depend_on_task_or_gold(self):
        rows = self.examples()
        _, original, first = self.run_retrieval(rows)
        for row in rows:
            row["gold_label"] = "F"
            row["task"] = "changed"
        _, changed, second = self.run_retrieval(rows)
        self.assertEqual(
            [row["prompt"] for row in original], [row["prompt"] for row in changed]
        )
        self.assertEqual(first, second)

    def test_unrelated_cve_is_not_retrieved(self):
        rows = self.examples()
        rows[0]["prompt"] = rows[0]["prompt"].replace("CVE-2024-0007", "CVE-2024-9999")
        rows[0]["source_url"] = "https://example.org/CVE-2024-9999.json"
        _, _, audit = self.run_retrieval(rows)
        self.assertFalse(
            any(
                item["source"] in {"cve", "nvd", "kev"}
                for item in audit[0]["retrieved"]
            )
        )

    def test_hybrid_calls_existing_endpoint_and_keeps_source_metadata(self):
        with patch(
            "secure_rag.retrieval._request_embeddings",
            side_effect=lambda *args: [[1.0, 0.0] for _ in args[3]],
        ) as request:
            _, _, audit = self.run_retrieval(
                self.examples(),
                strategy="hybrid",
                embedding_cache_path=self.root / "embed.jsonl",
                base_url="http://localhost/v1",
                model="test",
            )
        request.assert_called_once()
        self.assertTrue(
            all(
                "dense_rank" in item and "source_url" in item
                for item in audit[0]["retrieved"]
            )
        )
        self.assertEqual(audit[1]["retrieved"], [])

    def test_rejects_path_collision_and_missing_hybrid_configuration(self):
        build_corpus(self.inputs(), self.root / "corpus.jsonl")
        write_jsonl(self.root / "examples.jsonl", self.examples())
        with self.assertRaisesRegex(ValueError, "distinct"):
            prepare_multisource_examples(
                self.root / "examples.jsonl",
                self.root / "corpus.jsonl",
                self.root / "examples.jsonl",
                self.root / "retrieval.jsonl",
            )
        with self.assertRaisesRegex(ValueError, "Hybrid retrieval needs"):
            prepare_multisource_examples(
                self.root / "examples.jsonl",
                self.root / "corpus.jsonl",
                self.root / "out.jsonl",
                self.root / "retrieval.jsonl",
                strategy="hybrid",
            )


if __name__ == "__main__":
    unittest.main()
