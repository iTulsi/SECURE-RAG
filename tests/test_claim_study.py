from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from secure_rag.benchmark_data import digest
from secure_rag.claim_study import (
    CITED_SCHEMA, FORMAT_SCHEMA, citation_matches, evaluate_study, make_request,
    parse_response, prepare_study, read_study, run_study, stages,
)
from secure_rag.dataset import sha256_file
from secure_rag.evaluation import _read_jsonl, write_metrics
from secure_rag.pilot import write_jsonl
from secure_rag.retrieval import CONTEXT_PREFIX, QUESTION_SUFFIX
from secure_rag.runner import BaselineError


def fixture(root: Path) -> tuple[Path, Path]:
    official, prepared = root / "official", root / "prepared"
    official.mkdir()
    rows, labels = [], []
    for task in ("KCV", "VOOD"):
        with (official / f"SECURE - {task}.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["URL", "Prompt", "Question", "Correct Answer"])
            for i in range(3):
                cve = f"CVE-2024-{i+1:04d}"
                question = f"User interaction is required to exploit {cve}."
                context = {"cveMetadata": {"cveId": cve},
                           "descriptions": [{"value": "The product requires user interaction."}],
                           "metrics": {"userInteraction": "REQUIRED"}}
                prompt = CONTEXT_PREFIX + json.dumps(context) + QUESTION_SUFFIX + question
                if task == "VOOD":
                    prompt = "No record supplied. " + question
                label = "X" if task == "VOOD" else "T" if i == 1 else "F"
                writer.writerow([f"https://example.com/{cve}", prompt, question, label])
                identifier = f"{task.lower()}-{i:04d}"
                rows.append({"id": identifier, "task": task, "group_id": f"g{i}",
                             "split": "development" if i == 0 else "validation", "prompt": prompt})
                labels.append({"id": identifier, "gold_label": label})
    write_jsonl(prepared / "inputs.jsonl", rows)
    write_jsonl(prepared / "labels.jsonl", labels)
    write_metrics(prepared / "manifest.json", {
        "inputs_sha256": digest(rows), "labels_sha256": digest(labels),
        "audit": {t: {"sha256": sha256_file(official / f"SECURE - {t}.tsv")}
                  for t in ("KCV", "VOOD")},
    })
    return prepared, official


IDENTITY = {"model": "fake:test", "digest": "frozen-fixture"}


def fake_response(*args) -> dict:
    request = args[3]
    schema = request.get("format")
    if schema is None:
        content = "T"
    elif schema == FORMAT_SCHEMA:
        content = json.dumps({"verdict": "T"})
    else:
        assert schema == CITED_SCHEMA
        text = request["messages"][0]["content"].split("Passages:\n", 1)[1]
        passages, _ = json.JSONDecoder().raw_decode(text)
        first = passages[0] if passages else {"id": "", "text": ""}
        content = json.dumps({"verdict": "T", "passage_id": first["id"],
                              "quote": first["text"][:35], "comparison": "Compared claim and record.",
                              "confidence": 80})
    return {"message": {"content": content}, "done": True, "done_reason": "stop",
            "prompt_eval_count": 20, "eval_count": 10}


class ClaimStudyTests(unittest.TestCase):
    def test_prepares_validation_pairs_and_controlled_conditions_without_gold_in_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, official = fixture(root)
            manifest = prepare_study(prepared, official, root / "study")
            cases, labels, _ = read_study(root / "study")
            self.assertEqual(manifest["paired_claims"], 2)
            self.assertEqual(manifest["case_count"], 10)
            self.assertEqual(manifest["expected_calls"], 28)
            self.assertTrue(all("gold_label" not in c for c in cases))
            self.assertTrue(all(c["pair_id"] != "kcv-0000" for c in cases))
            for case in cases:
                if case["condition"] in {"unrelated", "missing"}:
                    self.assertEqual(labels[case["id"]], "X")
                if case["condition"] == "unrelated":
                    self.assertTrue(all(p["cve_id"] == "CVE-2024-0001" for p in case["passages"]))
                if case["condition"] == "bm25_top3":
                    full = next(c for c in cases if c["pair_id"] == case["pair_id"] and c["condition"] == "full_record")
                    self.assertTrue({p["id"] for p in case["passages"]} <= {p["id"] for p in full["passages"]})

    def test_citation_matching_checks_text_not_entailment_or_correctness(self):
        case = {"passages": [{"id": "p1", "text": "Product Delta is affected."}]}
        answer = {"verdict": "T", "passage_id": "p1", "quote": "Delta is affected."}
        self.assertTrue(citation_matches(case, answer))
        self.assertFalse(citation_matches(case, {**answer, "passage_id": "p2"}))
        self.assertFalse(citation_matches(case, {**answer, "quote": "Beta is unaffected."}))
        self.assertFalse(citation_matches(case, {**answer, "quote": ""}))
        self.assertEqual(answer["verdict"], "T")

    def test_parse_rejects_invalid_shapes_confidence_and_truncated_outputs(self):
        good = {"verdict": "T", "quote": "fact", "passage_id": "p1", "comparison": "Supports claim.", "confidence": 80}
        record = {"stage": "cited", "response": {"done_reason": "stop", "message": {"content": json.dumps(good)}}}
        self.assertEqual(parse_response(record), good)
        for value in ({**good, "confidence": True}, {**good, "confidence": 101},
                      {**good, "comparison": ""}, {**good, "verdict": ["T"]}, {**good, "extra": "F"}):
            changed = {**record, "response": {"done_reason": "stop", "message": {"content": json.dumps(value)}}}
            self.assertIsNone(parse_response(changed))
        self.assertIsNone(parse_response({**record, "response": {**record["response"], "done_reason": "length"}}))

    def test_official_prompt_is_exact_and_no_stage_receives_gold(self):
        case = {"id": "c", "condition": "missing", "official_prompt": "Exact original",
                "claim": "A claim", "passages": [], "gold_label": "SECRET_GOLD"}
        config = {"model": "fake:test", "seed": 1, "context_window": 32768}
        self.assertEqual(make_request(case, "official", None, config)["messages"][0]["content"], "Exact original")
        for stage in stages(case):
            request = make_request(case, stage, "prior" if stage == "verified" else None, config)
            self.assertNotIn("SECRET_GOLD", json.dumps(request))
        with self.assertRaisesRegex(ValueError, "saved cited"):
            make_request(case, "verified", None, config)
        with self.assertRaisesRegex(ValueError, "not defined"):
            make_request({**case, "condition": "unrelated"}, "official", None, config)

    @patch("secure_rag.claim_study.model_identity", return_value=IDENTITY)
    @patch("secure_rag.claim_study.request_json", side_effect=fake_response)
    def test_resumes_all_conditions_and_scores_unsupported_model_answers_without_overrides(self, request, identity):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            prepared, official = fixture(root)
            data, out = root / "study", root / "run"
            prepare_study(prepared, official, data)
            status = run_study(data, out, "fake:test", max_new_pairs=1)
            self.assertFalse(status["complete"])
            self.assertEqual(request.call_count, 14)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                evaluate_study(data, out)
            status = run_study(data, out, "fake:test")
            self.assertTrue(status["complete"])
            self.assertEqual(request.call_count, 28)
            metrics = json.loads((out / "metrics.json").read_text())
            for condition in ("missing", "unrelated"):
                self.assertEqual(metrics["by_condition"][condition]["cited"]["accuracy"], 0)
                self.assertEqual(metrics["by_condition"][condition]["cited"]["binary_answer_rate"], 1)
            self.assertEqual(metrics["verifier_total_cost"]["calls"], 20)
            samples = _read_jsonl(out / "citation_audit_samples.jsonl")
            self.assertTrue(all(s["quote_entails_verdict"] is None for s in samples))
            self.assertTrue(all("gold_label" not in s for s in samples))
            run_study(data, out, "fake:test")
            self.assertEqual(request.call_count, 28)
            with self.assertRaisesRegex(ValueError, "Frozen validation"):
                run_study(data, out, "fake:test", seed=999)

    @patch("secure_rag.claim_study.model_identity", return_value=IDENTITY)
    def test_network_failure_keeps_completed_call_and_rejects_tampered_cache(self, identity):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            prepared, official = fixture(root)
            data, out = root / "study", root / "run"
            prepare_study(prepared, official, data)
            with patch("secure_rag.claim_study.request_json", side_effect=[
                {"message": {"content": "T"}, "done": True, "done_reason": "stop"},
                BaselineError("offline"),
            ]):
                with self.assertRaisesRegex(BaselineError, "offline"):
                    run_study(data, out, "fake:test")
            records = _read_jsonl(out / "responses.jsonl")
            self.assertEqual(len(records), 1)
            records[0]["response"]["message"]["content"] = "F"
            write_jsonl(out / "responses.jsonl", records)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_study(data, out, "fake:test")

    def test_prepared_data_tampering_and_nonvalidation_split_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, official = fixture(root)
            data = root / "study"
            prepare_study(prepared, official, data)
            path = data / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["split"] = "test"
            write_metrics(path, manifest)
            with self.assertRaisesRegex(ValueError, "integrity"):
                read_study(data)


if __name__ == "__main__":
    unittest.main()
