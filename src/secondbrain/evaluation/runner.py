"""Run a small golden-set evaluation against a running Second Brain API.

This intentionally measures retrieval separately from generated-answer signals.
NLI support and citation validity are model/system indicators, not human judgments
of factual correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

DEFAULT_GOLDEN = Path("eval/golden.json")
DEFAULT_API = "http://localhost:8000"
DEFAULT_REFUSAL = "I can't find this in your documents."


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate golden cases from a JSON file."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Golden set must be a non-empty JSON array.")

    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Golden case {index} must be an object.")
        for field in ("id", "question", "expected_sources"):
            if field not in case:
                raise ValueError(f"Golden case {index} is missing required field {field!r}.")
        if not isinstance(case["id"], str) or not case["id"].strip():
            raise ValueError(f"Golden case {index} has an invalid id.")
        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate golden case id: {case['id']!r}.")
        seen_ids.add(case["id"])
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"Golden case {case['id']!r} has an invalid question.")
        if not isinstance(case["expected_sources"], list) or not all(
            isinstance(title, str) for title in case["expected_sources"]
        ):
            raise ValueError(f"Golden case {case['id']!r} has invalid expected_sources.")
    return cases


def retrieval_metrics(ranked_titles: Sequence[str], expected: set[str]) -> dict[str, float | None]:
    """Compute source-level hit rate, reciprocal rank, and binary nDCG."""
    if not expected:
        return {"recall": None, "mrr": None, "ndcg": None}

    # Repeated chunks from one document count once: this set measures source
    # discovery, not how many chunks from the same source occupy the ranking.
    unique_titles = list(dict.fromkeys(ranked_titles))
    relevant = [title in expected for title in unique_titles]
    reciprocal_rank = next(
        (1.0 / rank for rank, is_relevant in enumerate(relevant, start=1) if is_relevant),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevant, start=1)
        if is_relevant
    )
    ideal_relevant = min(len(expected), len(unique_titles))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return {
        "recall": sum(relevant) / len(expected),
        "mrr": reciprocal_rank,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _request_json(client: httpx.Client, method: str, url: str, **kwargs: Any) -> Any:
    response = client.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


def run_evaluation(
    cases: list[dict[str, Any]], api_url: str, top_k: int, repeats: int, api_key: str = ""
) -> dict[str, Any]:
    """Call /search and strict /ask repeatedly, aggregating measured signals."""
    headers = {"X-API-Key": api_key} if api_key else {}
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=240.0, headers=headers) as client:
        for case in cases:
            expected = set(case["expected_sources"])
            answerable = bool(case.get("answerable", bool(expected)))
            run_rows = []
            for repeat in range(1, repeats + 1):
                search_data = _request_json(
                    client,
                    "POST",
                    f"{api_url.rstrip('/')}/search",
                    json={"query": case["question"], "top_k": top_k},
                )
                if not isinstance(search_data, list):
                    raise ValueError("/search returned an unexpected response shape.")
                ranked_titles = [str(hit["document_title"]) for hit in search_data]
                metrics = retrieval_metrics(ranked_titles, expected)

                ask_data = _request_json(
                    client,
                    "POST",
                    f"{api_url.rstrip('/')}/ask",
                    json={"question": case["question"], "top_k": top_k, "strict": True},
                )
                if not isinstance(ask_data, dict):
                    raise ValueError("/ask returned an unexpected response shape.")
                answer = str(ask_data.get("answer", ""))
                refusal = answer.strip().casefold() == DEFAULT_REFUSAL.casefold()
                run_rows.append(
                    {
                        "repeat": repeat,
                        "retrieved_sources": ranked_titles,
                        "retrieval": metrics,
                        "refused": refusal,
                        "refusal_correct": refusal == (not answerable),
                        "citations_valid": bool(ask_data.get("citations_valid", False)),
                        "supported_ratio": float(ask_data.get("supported_ratio", 0.0)),
                        "latency_ms": int(ask_data.get("latency_ms", 0)),
                        "answer": answer,
                    }
                )

            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "answerable": answerable,
                    "expected_sources": sorted(expected),
                    "unique_answer_count": len({run["answer"] for run in run_rows}),
                    "answers_consistent": len({run["answer"] for run in run_rows}) == 1,
                    "runs": run_rows,
                }
            )

    answerable_runs = [run for row in rows if row["answerable"] for run in row["runs"]]
    unanswerable_runs = [run for row in rows if not row["answerable"] for run in row["runs"]]
    all_runs = [run for row in rows for run in row["runs"]]

    def average(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    retrieval_values = [
        run["retrieval"] for row in rows if row["answerable"] for run in row["runs"]
    ]
    summary = {
        "answerable_retrieval_recall_at_k": average(
            [float(metrics["recall"] or 0.0) for metrics in retrieval_values]
        ),
        "answerable_mrr": average([float(metrics["mrr"] or 0.0) for metrics in retrieval_values]),
        "answerable_ndcg": average([float(metrics["ndcg"] or 0.0) for metrics in retrieval_values]),
        "answerable_citation_valid_rate": average(
            [float(run["citations_valid"]) for run in answerable_runs]
        ),
        "answerable_mean_nli_supported_ratio": average(
            [run["supported_ratio"] for run in answerable_runs]
        ),
        "answerable_refusal_rate": average([float(run["refused"]) for run in answerable_runs]),
        "unanswerable_correct_refusal_rate": average(
            [float(run["refusal_correct"]) for run in unanswerable_runs]
        ),
        "exact_answer_consistency": average([float(row["answers_consistent"]) for row in rows]),
        "mean_ask_latency_ms": average([float(run["latency_ms"]) for run in all_runs]),
    }
    return {
        "api_url": api_url.rstrip("/"),
        "top_k": top_k,
        "repeats": repeats,
        "case_count": len(cases),
        "summary": summary,
        "cases": rows,
        "metric_notes": {
            "retrieval": (
                "Source-title relevance; recall is the fraction of expected titles found in top-k."
            ),
            "supported_ratio": (
                "The API's NLI estimate, not a human judgment or proof of factuality."
            ),
            "refusal": "Exact match to the configured strict refusal sentence.",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--api", default=os.getenv("SECONDBRAIN_API", DEFAULT_API))
    parser.add_argument("--api-key", default=os.getenv("SECONDBRAIN_API_KEY", ""))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional path for the full JSON report.")
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.top_k > 20 or args.repeats < 1:
        parser.error("--top-k must be from 1 to 20 and --repeats must be positive.")

    try:
        report = run_evaluation(
            load_cases(args.golden), args.api, args.top_k, args.repeats, args.api_key
        )
    except (OSError, ValueError, httpx.HTTPError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        print(f"Full report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
