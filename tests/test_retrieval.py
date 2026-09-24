from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from secure_rag.retrieval import (
    bm25_scores,
    chunk_context,
    extract_context,
    prepare_dense_examples,
    prepare_hybrid_examples,
    rank_hybrid,
)


class RetrievalTests(unittest.TestCase):
    def test_bm25_rewards_exact_lexical_evidence(self) -> None:
        documents = [
            "attack vector local access only",
            "the attack can be initiated remotely over the network",
        ]

        scores = bm25_scores("Can the attack be initiated remotely?", documents)

        self.assertGreater(scores[1], scores[0])

    def test_hybrid_ranking_records_both_rankers(self) -> None:
        question = "remote network attack"
        chunks = ["local access", "remote network attack is possible"]
        embeddings = {
            question: [1.0, 0.0],
            chunks[0]: [1.0, 0.0],
            chunks[1]: [0.0, 1.0],
        }

        ranked = rank_hybrid(question, chunks, embeddings, top_k=2, rrf_k=60)

        self.assertEqual(len(ranked), 2)
        self.assertEqual({row["text"] for row in ranked}, set(chunks))
        self.assertTrue(all("dense_rank" in row and "bm25_rank" in row for row in ranked))

    def test_extracts_and_chunks_official_style_context(self) -> None:
        prompt = (
            'You are given the following JSON data as context: {"name":"widget",'
            '"affected":true}  Based on the context, you have to analyze the '
            'following statement: Widget is affected.'
        )
        context = extract_context(prompt)

        self.assertEqual(context, {"name": "widget", "affected": True})
        self.assertEqual(
            chunk_context(context, max_chars=100),
            ['root.name = "widget"\nroot.affected = true'],
        )

    def test_returns_none_when_context_is_missing(self) -> None:
        self.assertIsNone(extract_context("No context. Question only."))

    def test_prepares_dense_kcv_and_preserves_empty_vood(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "examples.jsonl"
            examples.write_text(
                json.dumps(
                    {
                        "id": "kcv-0000",
                        "task": "KCV",
                        "question": "Is it affected?",
                        "gold_label": "T",
                        "prompt": (
                            'You are given the following JSON data as context: '
                            '{"affected":true,"severity":"high"}  Based on the '
                            "context, you have to analyze the following statement: x"
                        ),
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "id": "vood-0000",
                        "task": "VOOD",
                        "question": "Is it affected?",
                        "gold_label": "X",
                        "prompt": "Question without JSON evidence",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_embeddings(*args, **kwargs):
                texts = args[3]
                return [[1.0, float(index)] for index, _ in enumerate(texts)]

            with patch(
                "secure_rag.retrieval._request_embeddings", side_effect=fake_embeddings
            ):
                count, _ = prepare_dense_examples(
                    examples_path=examples,
                    output_path=root / "dense.jsonl",
                    retrieval_path=root / "retrieval.jsonl",
                    embedding_cache_path=root / "embeddings.jsonl",
                    base_url="http://localhost:1234/v1",
                    model="test-embedding",
                    timeout_seconds=1,
                    batch_size=32,
                    top_k=1,
                    max_chunk_chars=100,
                )

            dense = [
                json.loads(line)
                for line in (root / "dense.jsonl").read_text().splitlines()
            ]
            retrieval = [
                json.loads(line)
                for line in (root / "retrieval.jsonl").read_text().splitlines()
            ]
            self.assertEqual(count, 1)
            self.assertEqual([row["gold_label"] for row in dense], ["T", "X"])
            self.assertIn("[1]", dense[0]["prompt"])
            self.assertIn("No evidence was provided", dense[1]["prompt"])
            self.assertEqual(retrieval[0]["candidate_count"], 1)
            self.assertEqual(retrieval[1]["retrieved"], [])

    def test_hybrid_keeps_extra_candidates_out_of_generator_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "examples.jsonl"
            examples.write_text(json.dumps({
                "id": "kcv-1", "question": "affected?", "prompt": (
                    'You are given the following JSON data as context: '
                    '{"a":"firstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirstfirst",'
                    '"b":"secondsecondsecondsecondsecondsecondsecondsecondsecondsecondsecondsecondsecondsecondsecondsecond"}'
                    '  Based on the context, '
                    'you have to analyze the following statement: affected?'
                ),
            }) + "\n", encoding="utf-8")
            with patch("secure_rag.retrieval._request_embeddings",
                       side_effect=lambda *args, **kwargs: [[1.0, 1.0] for _ in args[3]]):
                prepare_hybrid_examples(
                    examples, root / "hybrid.jsonl", root / "retrieval.jsonl",
                    root / "embeddings.jsonl", "http://localhost/v1", "local", 5,
                    32, 1, 100, 60,
                )
            record = json.loads((root / "retrieval.jsonl").read_text().splitlines()[0])
            prompt = json.loads((root / "hybrid.jsonl").read_text().splitlines()[0])["prompt"]
            self.assertEqual(len(record["retrieved"]), 1)
            self.assertGreater(len(record["candidate_pool"]), 1)
            self.assertEqual(prompt.count("[1]"), 1)


if __name__ == "__main__":
    unittest.main()
