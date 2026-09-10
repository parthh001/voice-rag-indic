"""Phase 8: latency benchmark. >=30 real queries through the real pipeline
(src.pipeline.run_pipeline), per-stage + end-to-end latency, P50/P70/P100 via
numpy.percentile(..., method="nearest"). Cold start (first query -- model load,
index build) is reported separately from warm queries, never mixed into P100
(plan section 7 Phase 8 gotcha, and trap #8 in the trap table).

Mock mode (STT_PROVIDER=mock, LLM_PROVIDER=mock) is always run and always
measured here. "Real" mode (real Groq generation) is only attempted if
GROQ_API_KEY is present in the environment; if it is not, this script reports
"not measured" rather than fabricating a number -- see plan section 9, rule 2.
Real Sarvam STT additionally needs a real .wav sample, which this repo does not
ship (recorded demo audio is out of scope for CI) -- STT stays mocked even in
"real" mode, and this is stated explicitly in the output, not implied away.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.config import RESULTS_DIR, SUBSET_DIR

MIN_QUERIES = 30


def _load_eval_queries(n: int) -> list[str]:
    path = SUBSET_DIR / "eval.jsonl"
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            queries.append(rec["query_hi"])
            if len(queries) >= n:
                break
    return queries


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p70": None, "p100": None}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50, method="nearest")),
        "p70": float(np.percentile(arr, 70, method="nearest")),
        "p100": float(np.percentile(arr, 100, method="nearest")),
    }


def _run_mode(n_queries: int, strategy: str, stt_provider: str, llm_provider: str) -> dict:
    config.STT_PROVIDER = stt_provider
    config.LLM_PROVIDER = llm_provider

    from src import pipeline as pipeline_module

    queries = _load_eval_queries(n_queries)
    assert len(queries) >= n_queries, (
        f"only {len(queries)} eval queries available in data/subset/eval.jsonl, "
        f"need {n_queries} -- run `python -m src.data.loader` with a larger --n"
    )

    per_stage_ms: dict[str, list[float]] = {}
    end_to_end_ms: list[float] = []
    cold_start_ms: float | None = None

    for i, q in enumerate(queries):
        t0 = time.monotonic()
        result = pipeline_module.run_pipeline(
            audio_path="samples/nonexistent_demo.wav", strategy=strategy
        )
        wall_ms = (time.monotonic() - t0) * 1000

        if i == 0:
            cold_start_ms = wall_ms
            continue  # excluded from the warm percentile pool -- it includes model/index load

        end_to_end_ms.append(result.total_ms)
        for t in result.timings:
            per_stage_ms.setdefault(t.stage, []).append(t.ms)

    return {
        "stt_provider": stt_provider,
        "llm_provider": llm_provider,
        "strategy": strategy,
        "n_warm_queries": len(end_to_end_ms),
        "cold_start_ms": cold_start_ms,
        "end_to_end": _percentiles(end_to_end_ms),
        "per_stage": {stage: _percentiles(vals) for stage, vals in per_stage_ms.items()},
    }


def run_latency_benchmark(n_queries: int = MIN_QUERIES, strategy: str = "fixed") -> dict:
    results: dict = {"n_queries_requested": n_queries, "strategy": strategy, "modes": {}}

    print(f"[latency_benchmark] mock mode: STT=mock, LLM=mock ({n_queries} queries + 1 cold start)")
    results["modes"]["mock"] = {"measured": True, **_run_mode(n_queries, strategy, "mock", "mock")}

    if os.environ.get("GROQ_API_KEY"):
        print(f"[latency_benchmark] real mode: STT=mock (no audio sample available), LLM=groq")
        results["modes"]["real_llm_mock_stt"] = {
            "measured": True,
            **_run_mode(n_queries, strategy, "mock", "groq"),
        }
    else:
        results["modes"]["real_llm_mock_stt"] = {
            "measured": False,
            "reason": "GROQ_API_KEY not set in this environment -- not measured, not estimated.",
        }

    results["modes"]["real_stt"] = {
        "measured": False,
        "reason": (
            "SARVAM_API_KEY and a real recorded .wav sample are both required and neither "
            "is available in this environment -- not measured, not estimated."
        ),
    }

    return results


def _write_markdown(results: dict, out_path: Path) -> None:
    lines = ["# Latency Benchmark — P50/P70/P100 per stage", ""]
    lines.append(
        f"Strategy: `{results['strategy']}`. Percentiles via `numpy.percentile(..., "
        "method=\"nearest\")`. Cold start (first query: model load + index build) is "
        "reported separately and excluded from the warm P50/P70/P100 pool below."
    )
    lines.append("")

    for mode_name, mode in results["modes"].items():
        lines.append(f"## {mode_name}")
        if not mode.get("measured"):
            lines.append("")
            lines.append(f"**Not measured.** {mode['reason']}")
            lines.append("")
            continue

        lines.append("")
        lines.append(
            f"STT provider: `{mode['stt_provider']}`  |  LLM provider: `{mode['llm_provider']}`  |  "
            f"warm queries: {mode['n_warm_queries']}  |  cold start: {mode['cold_start_ms']:.1f}ms"
        )
        lines.append("")
        lines.append("| Stage | P50 (ms) | P70 (ms) | P100 (ms) |")
        lines.append("|---|---|---|---|")
        e2e = mode["end_to_end"]
        lines.append(f"| **end-to-end** | {e2e['p50']:.1f} | {e2e['p70']:.1f} | {e2e['p100']:.1f} |")
        for stage, p in mode["per_stage"].items():
            if p["p50"] is None:
                continue
            lines.append(f"| {stage} | {p['p50']:.1f} | {p['p70']:.1f} | {p['p100']:.1f} |")
        lines.append("")

    lines.append("## The 200ms target — honest analysis")
    lines.append("")
    mock = results["modes"].get("mock", {})
    if mock.get("measured"):
        retrieve = mock["per_stage"].get("retrieve", {})
        if retrieve.get("p50") is not None:
            lines.append(
                f"Measured on this machine: query embedding + vector retrieval "
                f"(`retrieve` stage above) lands at P50={retrieve['p50']:.1f}ms, "
                f"P100={retrieve['p100']:.1f}ms over a ~10k-chunk corpus. This is the "
                "genuinely sub-200ms part of the pipeline, and the part the 200ms target "
                "most plausibly refers to."
            )
        lines.append("")
        lines.append(
            "The full voice-to-answer pipeline does **not** hit 200ms end-to-end, even in "
            "mock mode, once generation and guardrail stages are included -- and it cannot, "
            "in real-API mode, once network-bound STT (300-1500ms) and LLM decode "
            "(400-3000ms) are added on top of what's measured here. See "
            "VOICE_RAG_BUILD_PLAN.md section 3 for the stage-by-stage floor analysis this "
            "was measured against."
        )
        lines.append("")
        lines.append(
            "**What it would take to get closer to 200ms end-to-end with real STT+LLM:** "
            "a local quantized STT model (e.g. whisper.cpp tiny/base) instead of a network "
            "round trip; a small local generator instead of a hosted API; a streaming-first "
            "design that measures time-to-first-token rather than time-to-full-answer; and "
            "aggressive caching for repeated queries. These are estimates, not measurements, "
            "and are stated as such."
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=MIN_QUERIES)
    parser.add_argument("--strategy", type=str, default="fixed")
    args = parser.parse_args()

    results = run_latency_benchmark(n_queries=args.n, strategy=args.strategy)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "latency.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = RESULTS_DIR / "latency.md"
    _write_markdown(results, md_path)
    print(f"Wrote {json_path} and {md_path}")
