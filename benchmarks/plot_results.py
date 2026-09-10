"""Generates results/chunking_comparison.png from the real numbers in
results/chunking_comparison.json -- a README hero chart, not a separate
measurement. Run after benchmarks/chunking_comparison.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import RESULTS_DIR

STRATEGIES = ["fixed", "semantic", "metadata_aware"]
MODES = ["hi_to_hi", "hi_to_en", "en_to_en"]
MODE_LABELS = {"hi_to_hi": "hi→hi", "hi_to_en": "hi→en (cross-lingual)", "en_to_en": "en→en"}
COLORS = {"fixed": "#4C72B0", "semantic": "#DD8452", "metadata_aware": "#55A868"}


def plot_chunking_comparison() -> Path:
    data = json.loads((RESULTS_DIR / "chunking_comparison.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(MODES))
    width = 0.25

    for i, strategy in enumerate(STRATEGIES):
        values = [data["strategies"][strategy]["modes"][mode]["overall"]["recall@5"] for mode in MODES]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=strategy, color=COLORS[strategy])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_ylabel("Recall@5")
    ax.set_title("Chunking strategy comparison — Recall@5 by corpus mode\n(1,000 real queries, real relevance labels, MSMARCO-XI Hindi validation subset)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES])
    ax.set_ylim(0, 1.05)
    ax.legend(title="strategy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    out_path = RESULTS_DIR / "chunking_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_guardrail_comparison() -> Path:
    data = json.loads((RESULTS_DIR / "guardrail_eval.json").read_text(encoding="utf-8"))
    cheap = data["cheap_checks_only"]
    judge = data.get("with_llm_judge")

    labels = ["cheap checks only"] + (["LLM judge, standalone"] if judge else [])
    precisions = [cheap["precision"]] + ([judge["precision"]] if judge else [])
    recalls = [cheap["recall"]] + ([judge["recall"]] if judge else [])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(labels))
    width = 0.32
    ax.bar(x - width / 2, precisions, width, label="precision", color="#4C72B0")
    ax.bar(x + width / 2, recalls, width, label="recall", color="#DD8452")
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        ax.text(i - width / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Off-topic guardrail: cheap check vs. real LLM judge\n(40-query hand-labelled set)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    out_path = RESULTS_DIR / "guardrail_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p1 = plot_chunking_comparison()
    print(f"wrote {p1}")
    p2 = plot_guardrail_comparison()
    print(f"wrote {p2}")
