"""
Answer + citation-verification table for 5 example queries (Info Edge AoA).

Uses the Chroma index built by run_chunk_eval.py (collection "aoa_eval"), the project's
hybrid retrieval (dense + BM25 + RRF, NO reranker, top-5), the project's generate_answer()
and the project's verify_citations().

Usage (project root; Ollama running; GROQ_API_KEY set in .env):
    python run_answer_table.py                  # uses the 1000-char index
    python run_answer_table.py --chunk-size 1500

Pass/Fail rule (all three must hold):
  1. citations_valid    - project's verify_citations() says the citations are valid
                          (known chunk IDs, no uncited claim lines)
  2. citation_supports  - at least one CITED chunk literally contains the Articles' key phrase
  3. answer_correct     - the answer text contains the expected key facts (regex checks below)
"""
import argparse
import csv
import os
import re

from utils import retrieval
from utils.generation import generate_answer

OUT_DIR = "evaluation_results"
COLLECTION = "aoa_eval"

# (question, key phrase that must be in a cited chunk, expected answer, [regexes the answer must match])
CASES = [
    ("What is the minimum and maximum number of Directors?",
     "number of Directors of the Company shall not be less than 6",
     "Minimum 6, maximum 15 (Article 33(a))",
     [r"\b6\b|\bsix\b", r"\b15\b|fifteen"]),
    ("How many Board Meetings must be held in a calendar year?",
     "At least 4 (four) Board Meetings",
     "At least 4 (Article 52(a))",
     [r"\b4\b|\bfour\b"]),
    ("How much notice is required for a Board Meeting?",
     "At least 7 (seven) days",
     "At least 7 days' written notice (Article 52(d))",
     [r"\b7\b|\bseven\b"]),
    ("Who appoints the Chairperson of the Board?",
     "members of the Board may elect",
     "The members of the Board elect one of themselves (Article 34(a))",
     [r"elect"]),
    ("What is the quorum for a Board Meeting?",
     "quorum for each Board Meeting shall be",
     "One-third of total strength or two directors, whichever is higher (Article 53(a))",
     [r"one[- ]third", r"\btwo\b|\b2\b"]),
]


def norm(s):
    return " ".join(s.lower().replace("’", "'").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=1000)
    args = ap.parse_args()
    persist = os.path.join(OUT_DIR, f"chroma_{args.chunk_size}")
    if not os.path.isdir(persist):
        raise SystemExit(f"{persist} not found. Run run_chunk_eval.py first.")

    rows = []
    for q, phrase, expected, patterns in CASES:
        chunks = retrieval.retrieve(q, collection_name=COLLECTION, persist_dir=persist,
                                    top_n_final=5, use_reranker=False)
        gen = generate_answer(q, chunks)
        answer = gen["answer"]
        report = gen["citation_report"] or {}
        cited = set(report.get("citations", []))

        citations_valid = bool(report.get("is_valid"))
        citation_supports = any(norm(phrase) in norm(c["text"]) for c in chunks if c["chunk_id"] in cited)
        # LLMs often emit non-breaking/Unicode hyphens ("one‑third"); fold them to "-" before matching.
        answer_for_match = re.sub(r"[‐-―−]", "-", answer)
        answer_correct = all(re.search(p, answer_for_match, re.I) for p in patterns)
        passed = citations_valid and citation_supports and answer_correct
        problems = "; ".join(report.get("problems", []))

        rows.append({
            "chunk_size": args.chunk_size,
            "query": q,
            "expected_answer": expected,
            "retrieved_answer": answer,
            "cited_chunks": ", ".join(sorted(cited)),
            "key_phrase_in_top5": any(norm(phrase) in norm(c["text"]) for c in chunks),
            "citations_valid": citations_valid,
            "citation_supports_answer": citation_supports,
            "answer_correct": answer_correct,
            "citation_problems": problems,
            "result": "PASS" if passed else "FAIL",
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"answer_verification_{args.chunk_size}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def cell(s):
        return str(s).replace("\n", " ").replace("|", "/")

    md = ["| # | Query | Retrieved answer | Cited chunks | Answer chunk in top 5 | Citations valid | Cited chunk supports answer | Answer correct | Citation problems | Result |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {cell(r['query'])} | {cell(r['retrieved_answer'])} | {cell(r['cited_chunks'])} | "
                  f"{'Yes' if r['key_phrase_in_top5'] else 'No'} | "
                  f"{'Yes' if r['citations_valid'] else 'No'} | {'Yes' if r['citation_supports_answer'] else 'No'} | "
                  f"{'Yes' if r['answer_correct'] else 'No'} | {cell(r['citation_problems']) or '-'} | **{r['result']}** |")
    md_path = os.path.join(OUT_DIR, f"answer_verification_{args.chunk_size}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nPassed {sum(r['result'] == 'PASS' for r in rows)}/{len(rows)}. Saved {csv_path} and {md_path}")


if __name__ == "__main__":
    main()