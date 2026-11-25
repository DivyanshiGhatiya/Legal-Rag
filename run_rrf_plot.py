"""
RRF score analysis per chunk size (25 questions).

Reuses the Chroma indexes built by run_chunk_eval.py (evaluation_results/chroma_<size>),
the same hybrid retrieval (dense + BM25 + RRF, NO reranker, top-5) and the same questions.
Only the query embedding needs Ollama; chunks are not re-embedded.

Usage (project root):  python run_rrf_plot.py

Outputs (evaluation_results/):
  rrf_scores_25q.csv         one row per (chunk size, question, rank) with the chunk's RRF score
  rrf_scores_25q.png         two panels, see below
  rrf_summary_25q.csv        mean RRF per chunk size for rank 1, relevant chunks, non-relevant chunks

Note: RRF = sum of 1/(60 + rank) over the dense and BM25 lists, so it is rank-based. The maximum
possible score is 2/61 = 0.0328 (chunk ranked #1 by BOTH dense and BM25). It measures how strongly
the two retrievers agree, not similarity.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_chunk_eval import CHUNK_SIZES, QUESTIONS, TOP_K, norm
from utils import retrieval

OUT_DIR = "evaluation_results"
SUFFIX = "_25q"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    rows = []
    for size in CHUNK_SIZES:
        persist = os.path.join(OUT_DIR, f"chroma_{size}")
        if not os.path.isdir(persist):
            raise SystemExit(f"{persist} not found. Run run_chunk_eval.py first.")
        for q, phrase in QUESTIONS:
            res = retrieval.retrieve(q, collection_name="aoa_eval", persist_dir=persist,
                                     top_n_final=TOP_K, use_reranker=False)
            for rank, r in enumerate(res, 1):
                rows.append({
                    "chunk_size": size, "question": q, "rank": rank, "chunk_id": r["chunk_id"],
                    "rrf_score": r["rrf_score"],
                    "relevant": norm(phrase) in norm(r["text"]),
                })

    with open(os.path.join(OUT_DIR, f"rrf_scores{SUFFIX}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for size in CHUNK_SIZES:
        sub = [r for r in rows if r["chunk_size"] == size]
        summary.append({
            "chunk_size": size,
            "mean_rrf_rank1": round(mean([r["rrf_score"] for r in sub if r["rank"] == 1]), 5),
            "mean_rrf_relevant": round(mean([r["rrf_score"] for r in sub if r["relevant"]]), 5),
            "mean_rrf_not_relevant": round(mean([r["rrf_score"] for r in sub if not r["relevant"]]), 5),
        })
    with open(os.path.join(OUT_DIR, f"rrf_summary{SUFFIX}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    # Panel 1: mean RRF score at each rank position, one line per chunk size
    for size in CHUNK_SIZES:
        ys = [mean([r["rrf_score"] for r in rows if r["chunk_size"] == size and r["rank"] == k])
              for k in range(1, TOP_K + 1)]
        ax[0].plot(range(1, TOP_K + 1), ys, marker="o", label=f"{size} chars")
    ax[0].set_xticks(range(1, TOP_K + 1))
    ax[0].set_xlabel("Rank in top 5")
    ax[0].set_ylabel("Mean RRF score")
    ax[0].set_title("Mean RRF score by rank")
    ax[0].axhline(2 / 61, ls=":", color="gray")
    ax[0].text(1.05, 2 / 61, "max = 2/61", va="bottom", fontsize=8, color="gray")
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    # Panel 2: relevant vs non-relevant chunks, by chunk size
    xs = range(len(CHUNK_SIZES))
    ax[1].bar([x - 0.2 for x in xs], [s["mean_rrf_relevant"] for s in summary], width=0.4, label="Relevant chunks")
    ax[1].bar([x + 0.2 for x in xs], [s["mean_rrf_not_relevant"] for s in summary], width=0.4, label="Other retrieved chunks")
    ax[1].set_xticks(list(xs))
    ax[1].set_xticklabels(CHUNK_SIZES)
    ax[1].set_xlabel("Chunk size (characters, overlap=100)")
    ax[1].set_ylabel("Mean RRF score")
    ax[1].set_title("RRF score: relevant vs other chunks")
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].legend()
    fig.suptitle("RRF scores by chunk size (25 questions, hybrid dense+BM25, no reranker)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"rrf_scores{SUFFIX}.png"), dpi=200)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()