from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import compare_runs, write_comparison
from .corpus import build_corpus
from .dataset import TASK_FILENAMES, load_and_validate, sha256_file, summarize
from .evaluation import evaluate, write_metrics
from .kcv_facts import apply_cvss_facts
from .pilot import make_prediction_template, prepare_pilot
from .retrieval import (
    prepare_dense_examples,
    prepare_hybrid_examples,
    prepare_multisource_examples,
)
from .runner import run_baseline
from .stages import (
    apply_verification,
    prepare_claim_evidence,
    prepare_context_prompts,
    prepare_reranked_examples,
    prepare_verification,
    route_by_context,
)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="secure-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate official KCV/VOOD data")
    validate.add_argument("--data-dir", required=True, type=_path)

    pilot = subparsers.add_parser(
        "prepare-pilot", help="create a paired source-grouped pilot"
    )
    pilot.add_argument("--data-dir", required=True, type=_path)
    pilot.add_argument("--output-dir", required=True, type=_path)
    pilot.add_argument("--sources", type=int, default=20)
    pilot.add_argument("--seed", type=int, default=20260920)
    pilot.add_argument("--upstream-commit")
    pilot.add_argument(
        "--exclude-manifest",
        type=_path,
        help="exclude all CVE sources selected in an earlier pilot",
    )

    corpus = subparsers.add_parser(
        "build-corpus", help="normalize official evidence snapshots"
    )
    corpus.add_argument("--cve-dir", type=_path, help="directory of CVE JSON 5 records")
    for source in ("nvd", "kev", "cwe", "attack"):
        corpus.add_argument(
            f"--{source}", type=_path, help="native source file (CWE XML or ZIP)"
        )
    corpus.add_argument("--output", required=True, type=_path)
    corpus.add_argument("--max-chunk-chars", type=int, default=1200)

    multi = subparsers.add_parser(
        "prepare-multisource", help="entity-routed multi-dataset evidence retrieval"
    )
    for name in ("examples", "corpus", "output", "retrieval-output"):
        multi.add_argument(f"--{name}", required=True, type=_path)
    multi.add_argument("--strategy", choices=("bm25", "hybrid"), default="bm25")
    multi.add_argument("--top-k", type=int, default=5)
    multi.add_argument("--max-chunk-chars", type=int, default=1200)
    multi.add_argument("--embedding-cache", type=_path)
    multi.add_argument("--base-url")
    multi.add_argument("--model")
    multi.add_argument("--timeout", type=float, default=120)
    multi.add_argument("--batch-size", type=int, default=32)
    multi.add_argument("--rrf-k", type=int, default=60)

    template = subparsers.add_parser(
        "make-template", help="create an X-only sanity baseline"
    )
    template.add_argument("--examples", required=True, type=_path)
    template.add_argument("--output", required=True, type=_path)

    score = subparsers.add_parser("evaluate", help="strictly score cached predictions")
    score.add_argument("--examples", required=True, type=_path)
    score.add_argument("--predictions", required=True, type=_path)
    score.add_argument("--output", required=True, type=_path)

    run = subparsers.add_parser(
        "run-baseline", help="run an OpenAI-compatible endpoint"
    )
    run.add_argument("--examples", required=True, type=_path)
    run.add_argument("--output", required=True, type=_path)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--seed", type=int)
    run.add_argument("--limit", type=int)

    reasoned = subparsers.add_parser(
        "run-reasoned",
        help="run evidence-first prompts with an auditable final verdict",
    )
    reasoned.add_argument("--examples", required=True, type=_path)
    reasoned.add_argument("--output", required=True, type=_path)
    reasoned.add_argument("--base-url", required=True)
    reasoned.add_argument("--model", required=True)
    reasoned.add_argument("--timeout", type=float, default=120.0)
    reasoned.add_argument("--seed", type=int)
    reasoned.add_argument("--limit", type=int)

    dense = subparsers.add_parser(
        "prepare-dense", help="retrieve evidence for the E1 dense-only experiment"
    )
    dense.add_argument("--examples", required=True, type=_path)
    dense.add_argument("--output", required=True, type=_path)
    dense.add_argument("--retrieval-output", required=True, type=_path)
    dense.add_argument("--embedding-cache", required=True, type=_path)
    dense.add_argument("--base-url", required=True)
    dense.add_argument("--model", required=True)
    dense.add_argument("--timeout", type=float, default=120.0)
    dense.add_argument("--batch-size", type=int, default=32)
    dense.add_argument("--top-k", type=int, default=5)
    dense.add_argument("--max-chunk-chars", type=int, default=1200)

    hybrid = subparsers.add_parser(
        "prepare-hybrid", help="retrieve evidence for the E2 BM25+dense experiment"
    )
    hybrid.add_argument("--examples", required=True, type=_path)
    hybrid.add_argument("--output", required=True, type=_path)
    hybrid.add_argument("--retrieval-output", required=True, type=_path)
    hybrid.add_argument("--embedding-cache", required=True, type=_path)
    hybrid.add_argument("--base-url", required=True)
    hybrid.add_argument("--model", required=True)
    hybrid.add_argument("--timeout", type=float, default=120.0)
    hybrid.add_argument("--batch-size", type=int, default=32)
    hybrid.add_argument("--top-k", type=int, default=5)
    hybrid.add_argument("--max-chunk-chars", type=int, default=1200)
    hybrid.add_argument("--rrf-k", type=int, default=60)

    compare = subparsers.add_parser(
        "compare-runs", help="compare paired predictions and retrieval outcomes"
    )
    compare.add_argument("--examples", required=True, type=_path)
    compare.add_argument("--baseline", required=True, type=_path)
    compare.add_argument("--candidate", required=True, type=_path)
    compare.add_argument("--retrieval", type=_path)
    compare.add_argument("--output", required=True, type=_path)

    rerank = subparsers.add_parser(
        "prepare-reranked", help="E3 local-model passage reranking"
    )
    rerank.add_argument("--examples", required=True, type=_path)
    rerank.add_argument("--retrieval", required=True, type=_path)
    rerank.add_argument("--output", required=True, type=_path)
    rerank.add_argument("--reranking-output", required=True, type=_path)
    rerank.add_argument("--base-url", required=True)
    rerank.add_argument("--model", required=True)
    rerank.add_argument("--timeout", type=float, default=120.0)
    rerank.add_argument("--seed", type=int, default=20260920)
    rerank.add_argument("--top-k", type=int, default=5)

    verification = subparsers.add_parser(
        "prepare-verification", help="E4 evidence check prompts"
    )
    verification.add_argument("--examples", required=True, type=_path)
    verification.add_argument("--output", required=True, type=_path)

    gate = subparsers.add_parser("apply-verification", help="E4 abstention gate")
    gate.add_argument("--examples", required=True, type=_path)
    gate.add_argument("--generator", required=True, type=_path)
    gate.add_argument("--verifier", required=True, type=_path)
    gate.add_argument("--output", required=True, type=_path)

    route = subparsers.add_parser(
        "route-by-context",
        help="route supported examples to E0 and missing-context examples to E1",
    )
    route.add_argument("--examples", required=True, type=_path)
    route.add_argument("--dense-examples", required=True, type=_path)
    route.add_argument("--original-predictions", required=True, type=_path)
    route.add_argument("--dense-predictions", required=True, type=_path)
    route.add_argument("--output-examples", required=True, type=_path)
    route.add_argument("--output-predictions", required=True, type=_path)

    routed_prompts = subparsers.add_parser(
        "prepare-context-prompts",
        help="prepare E0/E1 routed prompts for one new inference run",
    )
    routed_prompts.add_argument("--examples", required=True, type=_path)
    routed_prompts.add_argument("--output", required=True, type=_path)

    claim = subparsers.add_parser(
        "prepare-claim-evidence", help="compact CVE context for claim verification"
    )
    claim.add_argument("--examples", required=True, type=_path)
    claim.add_argument("--output", required=True, type=_path)

    facts = subparsers.add_parser(
        "apply-cvss-facts", help="resolve direct KCV CVSS claims from supplied JSON"
    )
    facts.add_argument("--examples", required=True, type=_path)
    facts.add_argument("--predictions", required=True, type=_path)
    facts.add_argument("--output", required=True, type=_path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "apply-cvss-facts":
        count = apply_cvss_facts(args.examples, args.predictions, args.output)
        print(f"Checked CVSS facts for {count} KCV predictions; wrote {args.output}")
        return 0
    if args.command == "validate":
        kcv, vood = load_and_validate(args.data_dir)
        report = {
            "kcv": summarize(kcv),
            "vood": summarize(vood),
            "paired_rows": len(kcv),
            "dataset_sha256": {
                task: sha256_file(args.data_dir / filename)
                for task, filename in TASK_FILENAMES.items()
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-pilot":
        kcv, vood = load_and_validate(args.data_dir)
        excluded: frozenset[str] = frozenset()
        if args.exclude_manifest:
            manifest = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
            sources = manifest.get("selected_sources")
            if (
                not isinstance(sources, list)
                or not sources
                or any(not isinstance(source, str) for source in sources)
            ):
                raise ValueError("Excluded manifest lacks selected_sources")
            excluded = frozenset(sources)
        manifest = prepare_pilot(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            kcv=kcv,
            vood=vood,
            source_count=args.sources,
            seed=args.seed,
            upstream_commit=args.upstream_commit,
            excluded_sources=excluded,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.command == "build-corpus":
        inputs = (
            [("cve", path) for path in sorted(args.cve_dir.rglob("CVE-*.json"))]
            if args.cve_dir
            else []
        )
        inputs.extend(
            (source, getattr(args, source))
            for source in ("nvd", "kev", "cwe", "attack")
            if getattr(args, source)
        )
        report = build_corpus(inputs, args.output, args.max_chunk_chars)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-multisource":
        count, elapsed = prepare_multisource_examples(
            args.examples,
            args.corpus,
            args.output,
            args.retrieval_output,
            args.strategy,
            args.top_k,
            args.max_chunk_chars,
            args.embedding_cache,
            args.base_url,
            args.model,
            args.timeout,
            args.batch_size,
            args.rrf_k,
        )
        print(f"Wrote multi-source prompts with {count} passages in {elapsed:.3f}s")
        return 0

    if args.command == "make-template":
        count = make_prediction_template(args.examples, args.output)
        print(f"Wrote {count} predictions to {args.output}")
        return 0

    if args.command == "evaluate":
        metrics = evaluate(args.examples, args.predictions)
        write_metrics(args.output, metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0

    if args.command in {"run-baseline", "run-reasoned"}:
        new_count, total_count = run_baseline(
            examples_path=args.examples,
            output_path=args.output,
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
            seed=args.seed,
            limit=args.limit,
            reasoned=args.command == "run-reasoned",
        )
        print(f"Added {new_count} predictions; cache now contains {total_count}")
        return 0

    if args.command == "prepare-claim-evidence":
        counts = prepare_claim_evidence(args.examples, args.output)
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-dense":
        retrieved_count, elapsed = prepare_dense_examples(
            examples_path=args.examples,
            output_path=args.output,
            retrieval_path=args.retrieval_output,
            embedding_cache_path=args.embedding_cache,
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
            batch_size=args.batch_size,
            top_k=args.top_k,
            max_chunk_chars=args.max_chunk_chars,
        )
        print(
            f"Wrote dense prompts with {retrieved_count} retrieved passages "
            f"in {elapsed:.3f}s"
        )
        return 0

    if args.command == "prepare-hybrid":
        retrieved_count, elapsed = prepare_hybrid_examples(
            examples_path=args.examples,
            output_path=args.output,
            retrieval_path=args.retrieval_output,
            embedding_cache_path=args.embedding_cache,
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
            batch_size=args.batch_size,
            top_k=args.top_k,
            max_chunk_chars=args.max_chunk_chars,
            rrf_k=args.rrf_k,
        )
        print(
            f"Wrote hybrid prompts with {retrieved_count} retrieved passages "
            f"in {elapsed:.3f}s"
        )
        return 0

    if args.command == "compare-runs":
        comparison = compare_runs(
            examples_path=args.examples,
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            retrieval_path=args.retrieval,
        )
        write_comparison(args.output, comparison)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-reranked":
        new_count = prepare_reranked_examples(
            args.examples,
            args.retrieval,
            args.output,
            args.reranking_output,
            args.base_url,
            args.model,
            args.timeout,
            args.seed,
            args.top_k,
        )
        print(f"Ranked {new_count} new examples; wrote {args.output}")
        return 0

    if args.command == "prepare-verification":
        count = prepare_verification(args.examples, args.output)
        print(f"Wrote {count} verification prompts to {args.output}")
        return 0

    if args.command == "apply-verification":
        counts = apply_verification(
            args.examples, args.generator, args.verifier, args.output
        )
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0

    if args.command == "route-by-context":
        counts = route_by_context(
            args.examples,
            args.dense_examples,
            args.original_predictions,
            args.dense_predictions,
            args.output_examples,
            args.output_predictions,
        )
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-context-prompts":
        counts = prepare_context_prompts(args.examples, args.output)
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
