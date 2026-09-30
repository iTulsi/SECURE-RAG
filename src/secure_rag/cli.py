from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import compare_runs, write_comparison
from .corpus import build_corpus
from .dataset import TASK_FILENAMES, load_and_validate, sha256_file, summarize
from .evaluation import evaluate, write_metrics
from .kcv_facts import apply_cvss_facts
from .pilot import make_prediction_template, prepare_pilot, write_jsonl
from .retrieval import (
    prepare_dense_examples,
    prepare_hybrid_examples,
    prepare_multisource_examples,
)
from .runner import BaselineError, repair_reasoned_predictions, run_baseline
from .stages import (
    apply_verification,
    combine_decision_review,
    prepare_claim_evidence,
    prepare_context_prompts,
    prepare_decision_review,
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
    run.add_argument("--progress", action="store_true")

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
    reasoned.add_argument("--progress", action="store_true")

    review = subparsers.add_parser(
        "run-review", help="retrieve, resume model inference, score, and write a report"
    )
    review.add_argument("--examples", type=_path, default="data/pilot/examples.jsonl")
    review.add_argument("--corpus", type=_path, default="data/corpus/evidence.jsonl")
    review.add_argument("--output-dir", type=_path, default="outputs/submission")
    review.add_argument("--baseline-predictions", type=_path)
    review.add_argument("--base-url", default="http://localhost:11434/v1")
    review.add_argument("--model", default="qwen2.5:3b-instruct")
    review.add_argument("--seed", type=int, default=20260920)
    review.add_argument("--timeout", type=float, default=180)

    improve = subparsers.add_parser(
        "run-kcv-review", help="review uncertain claims with structured evidence decisions"
    )
    improve.add_argument("--examples", type=_path, default="data/pilot/examples.jsonl")
    improve.add_argument("--predictions", type=_path,
                         help="prior predictions to preserve valid binary decisions")
    improve.add_argument("--output-dir", type=_path, default="outputs/kcv_review")
    improve.add_argument("--base-url", default="http://localhost:11434/v1")
    improve.add_argument("--model", default="qwen2.5:3b-instruct")
    improve.add_argument("--seed", type=int, default=20260920)
    improve.add_argument("--timeout", type=float, default=180)

    repair = subparsers.add_parser(
        "repair-reasoned", help="recover explicit verdicts from saved responses"
    )
    for name in ("examples", "predictions", "output"):
        repair.add_argument(f"--{name}", required=True, type=_path)

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
    if args.command == "repair-reasoned":
        counts = repair_reasoned_predictions(args.examples, args.predictions, args.output)
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0
    if args.command == "run-kcv-review":
        names = ("review_examples.jsonl", "fixed_predictions.jsonl",
                 "review_predictions.jsonl", "predictions.jsonl", "metrics.json",
                 "comparison.json", "run_manifest.json", "RESULTS.md")
        if {args.examples, args.predictions}.intersection(
            (args.output_dir / name).resolve() for name in names
        ):
            raise ValueError("Choose an output directory that does not overwrite inputs")
        baseline = evaluate(args.examples, args.predictions) if args.predictions else None
        prompts, fixed, reviewed, combined = [args.output_dir / name for name in names[:4]]
        counts = prepare_decision_review(args.examples, args.predictions, prompts, fixed)
        print(json.dumps(counts, indent=2, sort_keys=True), flush=True)
        try:
            run_baseline(prompts, reviewed, args.base_url, args.model, args.timeout,
                         args.seed, None, progress=True, structured=True)
        except BaselineError as error:
            raise SystemExit(
                f"Review incomplete: {error}\nEnsure Ollama is running and supports "
                "JSON-schema output. Repeat this command to resume. No new metrics written."
            ) from error
        if not reviewed.exists():
            write_jsonl(reviewed, [])
        combine_decision_review(args.examples, prompts, fixed, reviewed, combined)
        metrics = evaluate(args.examples, combined)
        write_metrics(args.output_dir / "metrics.json", metrics)
        if args.predictions:
            write_comparison(args.output_dir / "comparison.json",
                             compare_runs(args.examples, args.predictions, combined))
        manifest = {
            "pipeline": "supplied-context routing, explicit CVSS facts, structured decision review",
            "model": args.model, "seed": args.seed, "counts": counts,
            "examples_sha256": sha256_file(args.examples),
            "prior_predictions_sha256": sha256_file(args.predictions) if args.predictions else None,
            "prompts_sha256": sha256_file(prompts),
            "reviewed_predictions_sha256": sha256_file(reviewed),
            "predictions_sha256": sha256_file(combined),
            "model_request": {"temperature": 0, "max_tokens": 512, "format": "json_schema"},
        }
        write_metrics(args.output_dir / "run_manifest.json", manifest)
        lines = ["# SECURE RAG measured decision-review results", "",
                 f"Review model: `{args.model}`. Seed: `{args.seed}`.", "",
                 "| Run | KCV accuracy | KCV coverage | VOOD accuracy | Overall accuracy |",
                 "|---|---:|---:|---:|---:|"]
        runs = [("Decision review", metrics)]
        if baseline is not None:
            runs.insert(0, ("Provided prior predictions", baseline))
        for label, result in runs:
            lines.append(f"| {label} | {result['kcv']['accuracy']:.2%} | "
                         f"{result['kcv']['coverage']:.2%} | {result['vood']['accuracy']:.2%} | "
                         f"{result['overall']['accuracy']:.2%} |")
        lines.extend(["", f"Routing counts: `{json.dumps(counts, sort_keys=True)}`.",
                      f"Invalid predictions: {metrics['overall']['invalid_predictions']}.",
                      f"Overall macro F1: {metrics['overall']['macro_f1']:.6f}.",
                      f"Abstention F1: {metrics['abstention']['f1']:.6f}.", "",
                      "Decisions use original supplied CVE evidence. No external retrieval "
                      "or model training was performed in this run. Missing context produces "
                      "X. Explicit supported CVSS claims use deterministic checks. Other "
                      "prior T/F decisions are retained; uncertain/invalid decisions get a "
                      "fresh model review. Without prior predictions all remaining "
                      "context-bearing claims get model review.", "",
                      "Gold labels and task names are used only for scoring. Prompts, saved "
                      "model responses, prior predictions, hashes, and paired changes are "
                      "available in this directory. Pilot results are exploratory; evaluate "
                      "the source-disjoint holdout without tuning against its labels.", ""])
        report = args.output_dir / "RESULTS.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        print(f"Complete. KCV: {metrics['kcv']['accuracy']:.2%}; "
              f"overall: {metrics['overall']['accuracy']:.2%}. Report: {report}")
        return 0
    if args.command == "run-review":
        names = (
            "examples.jsonl", "retrieval.jsonl", "predictions.jsonl", "metrics.json",
            "comparison.json", "run_manifest.json", "RESULTS.md",
        )
        inputs = {args.examples, args.corpus, args.baseline_predictions}
        if inputs.intersection((args.output_dir / name).resolve() for name in names):
            raise ValueError("Choose an output directory that does not overwrite inputs")
        # Reject a mismatched comparison set before making model requests.
        baseline_metrics = (
            evaluate(args.examples, args.baseline_predictions)
            if args.baseline_predictions else None
        )
        prompts = args.output_dir / "examples.jsonl"
        retrieval = args.output_dir / "retrieval.jsonl"
        predictions = args.output_dir / "predictions.jsonl"
        count, elapsed = prepare_multisource_examples(
            args.examples, args.corpus, prompts, retrieval
        )
        print(f"Retrieved {count} passages in {elapsed:.2f}s. Running {args.model}...", flush=True)
        try:
            run_baseline(
                prompts, predictions, args.base_url, args.model, args.timeout,
                args.seed, None, progress=True,
            )
        except BaselineError as error:
            raise SystemExit(
                f"Inference incomplete: {error}\n"
                "Start Ollama with 'ollama serve' and ensure the requested model is pulled. "
                "Repeat this command to resume saved predictions. No new metrics were written."
            ) from error
        metrics = evaluate(args.examples, predictions)
        write_metrics(args.output_dir / "metrics.json", metrics)
        if args.baseline_predictions:
            write_comparison(
                args.output_dir / "comparison.json",
                compare_runs(args.examples, args.baseline_predictions, predictions, retrieval),
            )
        manifest = {
            "pipeline": "context-available entity-routed BM25, top_k=5",
            "model": args.model,
            "seed": args.seed,
            "temperature": 0,
            "max_tokens": 4,
            "examples_sha256": sha256_file(args.examples),
            "corpus_sha256": sha256_file(args.corpus),
            "prompts_sha256": sha256_file(prompts),
            "predictions_sha256": sha256_file(predictions),
            "baseline_predictions_sha256": (
                sha256_file(args.baseline_predictions) if args.baseline_predictions else None
            ),
            "retrieved_passages": count,
            "examples": metrics["overall"]["examples"],
        }
        write_metrics(args.output_dir / "run_manifest.json", manifest)
        lines = [
            "# SECURE-RAG multi-source measured results", "",
            f"Model: `{args.model}`. Seed: `{args.seed}`. Examples: {manifest['examples']}.",
            "Protocol: context-available augmentation; five BM25-ranked passages.", "",
            "| Run | KCV accuracy | KCV coverage | VOOD accuracy | Overall accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
        runs = [("Multi-source model", metrics)]
        if baseline_metrics is not None:
            runs.insert(0, ("Provided baseline predictions", baseline_metrics))
        for label, result in runs:
            lines.append(
                f"| {label} | {result['kcv']['accuracy']:.2%} | "
                f"{result['kcv']['coverage']:.2%} | {result['vood']['accuracy']:.2%} | "
                f"{result['overall']['accuracy']:.2%} |"
            )
        lines.extend([
            "", f"Invalid predictions: {metrics['overall']['invalid_predictions']}.",
            f"Abstention F1: {metrics['abstention']['f1']:.2%}.", "",
            "These scores use complete prediction files. The provided baseline, if any, "
            "was scored again; it was not rerun by this command. See run_manifest.json "
            "for input/output hashes and comparison.json for paired changes.", "",
            "Pilot data was used during development; these scores do not establish "
            "generalization. Use the packaged source-disjoint holdout for that check. "
            "Current external records can disagree with historical benchmark labels. "
            "VOOD retains empty evidence by protocol; high VOOD accuracy does not "
            "demonstrate high factual KCV accuracy.", "",
        ])
        report = args.output_dir / "RESULTS.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        print(f"Complete. KCV: {metrics['kcv']['accuracy']:.2%}; "
              f"overall: {metrics['overall']['accuracy']:.2%}. Report: {report}")
        return 0
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
            progress=args.progress,
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
