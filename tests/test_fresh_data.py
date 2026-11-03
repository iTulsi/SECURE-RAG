from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from secure_rag.benchmark_data import digest
from secure_rag.claim_study import (
    CITED_SCHEMA, FORMAT_SCHEMA, declared_verdict, make_request, parse_response, run_study,
)
from secure_rag.evaluation import write_metrics
from secure_rag.fresh_data import fact_values, record_cases, record_url
from secure_rag.pilot import write_jsonl


def source_record(number: int = 10001) -> dict:
    return {"cveMetadata": {"cveId": f"CVE-2025-{number}", "state": "PUBLISHED",
                            "datePublished": "2025-01-01T00:00:00Z"},
            "containers": {"cna": {"descriptions": [{"value": "A vulnerability."}],
                                    "metrics": [{"cvssV3_1": {"version": "3.1", "baseSeverity": "HIGH",
                                                              "userInteraction": "NONE"}}]}}}


class FreshDataTests(unittest.TestCase):
    def test_fact_labels_are_balanced_traceable_and_not_in_model_inputs(self):
        record = source_record()
        cases, labels = record_cases(record, record_url("CVE-2025-10001"))
        self.assertEqual(len(cases), 4)
        self.assertEqual(sorted(r["gold_label"] for r in labels), ["F", "F", "T", "T"])
        self.assertTrue(all("gold_label" not in c for c in cases))
        self.assertTrue(all(c["target_cve_id"] == "CVE-2025-10001" for c in cases))
        self.assertTrue(all(c["target_cve_id"] in c["claim"] for c in cases))
        self.assertTrue(all("verified" not in c["arms"] for c in cases))
        self.assertTrue(all(r["basis"].startswith("/containers/cna/metrics/0/cvssV3_1/") for r in labels))

    def test_invalid_conflicting_and_outdated_records_are_rejected(self):
        base = source_record()
        conflicting = copy.deepcopy(base)
        conflicting["containers"]["cna"]["metrics"].append({"cvssV3_1": {
            "version": "3.1", "baseSeverity": "LOW", "userInteraction": "REQUIRED"}})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            fact_values(conflicting)
        for change in [{"state": "REJECTED"}, {"datePublished": "2024-12-01"}]:
            record = copy.deepcopy(base)
            record["cveMetadata"].update(change)
            with self.assertRaisesRegex(ValueError, "published 2025"):
                fact_values(record)
        with self.assertRaisesRegex(ValueError, "2025 CVE"):
            record_url("../../secret")

    def test_target_identity_and_nonempty_explanation_constraint_are_explicit(self):
        case = record_cases(source_record(), record_url("CVE-2025-10001"))[0][0]
        request = make_request(case, "cited", None, {"model": "qwen", "seed": 7, "context_window": 32768})
        self.assertIn("Target CVE: CVE-2025-10001", request["messages"][0]["content"])
        self.assertEqual(request["format"]["properties"]["comparison"]["minLength"], 1)
        self.assertNotIn("minLength", CITED_SCHEMA["properties"]["comparison"])

    def test_correct_decision_with_empty_explanation_has_separate_metrics(self):
        record = {"stage": "cited", "response": {"done_reason": "stop", "message": {
            "content": json.dumps({"verdict": "X", "passage_id": "", "quote": "",
                                   "comparison": "", "confidence": 70})}}}
        self.assertEqual(declared_verdict(record), "X")
        self.assertIsNone(parse_response(record))

    @patch("secure_rag.claim_study.model_identity", return_value={"model": "fake", "digest": "fixture"})
    def test_fresh_run_uses_three_single_pass_arms_and_reports_both_endpoints(self, identity):
        def response(*args):
            request = args[3]
            if "format" not in request:
                content = "T"
            elif request["format"] == FORMAT_SCHEMA:
                content = '{"verdict":"T"}'
            else:
                content = json.dumps({"verdict": "T", "quote": "", "passage_id": "",
                                      "comparison": "", "confidence": 80})
            return {"message": {"content": content}, "done": True, "done_reason": "stop"}

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            cases, labels = [], []
            for n in (10001, 10002):
                generated_cases, generated_labels = record_cases(source_record(n), record_url(f"CVE-2025-{n}"))
                cases.extend(generated_cases)
                labels.extend(generated_labels)
            write_jsonl(root / "data/inputs.jsonl", cases)
            write_jsonl(root / "data/labels.jsonl", labels)
            write_metrics(root / "data/manifest.json", {
                "study": "cited-claims-fresh-v3", "split": "external_structured_fact_test",
                "expected_calls": 24, "condition_counts": {"fresh_cve": 8},
                "inputs_sha256": digest(cases), "labels_sha256": digest(labels),
            })
            with patch("secure_rag.claim_study.request_json", side_effect=response) as request:
                status = run_study(root / "data", root / "run", "fake", max_new_pairs=1)
                self.assertFalse(status["complete"])
                self.assertEqual(request.call_count, 12)
                status = run_study(root / "data", root / "run", "fake")
                self.assertTrue(status["complete"])
                self.assertEqual(request.call_count, 24)
            metrics = json.loads((root / "run/metrics.json").read_text())
            cited = metrics["by_condition"]["fresh_cve"]["cited"]
            self.assertEqual(cited["verdict_accuracy"], .5)
            self.assertEqual(cited["accuracy"], 0)
            self.assertEqual(cited["complete_response_rate"], 0)
            self.assertIsNone(metrics["verifier_total_cost"])
            self.assertNotIn("verified", metrics["stage_cost"])


if __name__ == "__main__":
    unittest.main()
