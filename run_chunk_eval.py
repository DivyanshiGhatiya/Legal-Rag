"""
Chunk-size retrieval evaluation for the Legal RAG project (Info Edge AoA).

Only chunk_size varies (500/1000/1500/2000). Everything else is held constant:
document, cleaning, overlap=100, embedding model (Ollama nomic-embed-text),
hybrid dense+BM25 retrieval (RRF), NO cross-encoder reranker, top-5.

Usage (from the project root, with Ollama running and `ollama pull nomic-embed-text` done):
    python run_chunk_eval.py --pdf path/to/aoa.pdf             # full experiment
    python run_chunk_eval.py --pdf path/to/aoa.pdf --prep-only # extraction/chunk counts/relevance only
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys
import types

CHUNK_SIZES = [500, 1000, 1500, 2000]
OVERLAP = 100
TOP_K = 5
OUT_DIR = "evaluation_results"

# (question, exact phrase from the Articles that marks a chunk as relevant)
BASE_QUESTIONS = [
    ("What is the minimum and maximum number of Directors?", "number of Directors of the Company shall not be less than 6"),
    ("How many Board Meetings must be held in a calendar year?", "At least 4 (four) Board Meetings"),
    ("How much notice is required for a Board Meeting?", "At least 7 (seven) days"),
    ("Who appoints the Chairperson of the Board?", "members of the Board may elect"),
    ("What is the quorum for a Board Meeting?", "quorum for each Board Meeting shall be"),
]

EXTRA_QUESTIONS = [
    ('Can the Managing Director also be the Chairperson?', 'Managing Director of the Company can also be appointed as the Chairperson'),
    ('Does the Chairperson have a casting vote if there is a tie?', 'Chairperson shall have a casting vote in the event of a tie'),
    ('When can an alternate director be nominated for an absent director?', 'absence for a period of not less than 3 (three) months'),
    ('How many woman directors must the Board have?', 'at least 1 (one) woman Director'),
    ('Can the Company appoint more than 15 directors?', 'appoint more than 15 (fifteen) directors after passing Special Resolution'),
    ('Do directors have to hold qualification shares?', 'shall not be required to hold any qualification shares'),
    ('What fraction of directors retire by rotation at each AGM?', 'one third of such of the Directors as are liable to retire by rotation'),
    ('How many votes does each director have at a Board Meeting?', 'each Director may exercise 1 (one) vote'),
    ('What vote is needed to adopt a Board resolution?', 'affirmative vote of a majority of the Directors present'),
    ('What happens if a Board meeting lacks a quorum?', 'automatically stand adjourned'),
    ('Who can remove a Debenture Director?', 'A Debenture Director may be removed from office at any time'),
    ("When does a director's resignation take effect?", 'resignation of a director shall take effect from the date on which the notice is received'),
    ('Can a Board meeting be called on shorter notice?', 'shorter notice in case of any urgent matters'),
    ('What is the quorum when interested directors are two-thirds or more?', 'number of interested Directors exceeds or is equal to two-thirds of the total strength'),
    ('When must an alternate director vacate office?', 'shall vacate office if and when the Original Director returns to India'),
    ('Which powers need a special resolution in general meeting before the Board can use them?', 'shall not exercise the powers given under Section 180'),
    ('Can continuing directors act despite a vacancy?', 'continuing Directors may act notwithstanding any vacancy'),
    ('Under which sections is director remuneration paid?', 'shall be paid in accordance with the provisions of the Section 197'),
    ('Can directors be reimbursed travelling and hotel expenses?', 'Directors may be paid all travelling, hotel and other expenses'),
    ('Can a director be paid special remuneration for extra services?', 'such special remuneration for such extra services'),
]
QUESTIONS = BASE_QUESTIONS + EXTRA_QUESTIONS  # 25 questions; first 5 are the original set
SUFFIX = "_25q"  # keeps the earlier 5-question result files from being overwritten


def norm(s):
    """Whitespace/case/quote-insensitive form used ONLY for matching the relevance phrases."""
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.lower().split())


def extract_and_clean(pdf_path):
    data = open(pdf_path, "rb").read()
    try:
        import fitz  # noqa: F401
        from utils.pdf_processor import extract_text_and_metadata, clean_text
        extraction = extract_text_and_metadata(data, ocr_enabled=False)
        method = "utils.pdf_processor (PyMuPDF)"
    except ImportError:
        # Fallback: PyMuPDF absent. Reuse the project's real clean_text by stubbing the
        # unused imports, and read text with pypdf using the same "[PAGE n]" layout.
        sys.modules.setdefault("fitz", types.ModuleType("fitz"))
        ocr = types.ModuleType("utils.ocr_processor")
        ocr.extract_page_text_with_ocr = ocr.page_needs_ocr = lambda *a, **k: None
        sys.modules["utils.ocr_processor"] = ocr
        from utils.pdf_processor import clean_text
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(data))
        parts = [f"[PAGE {i}]\n{(p.extract_text() or '').strip()}" for i, p in enumerate(reader.pages, 1)]
        extraction = {"text": "\n".join(parts), "num_pages": len(parts), "file_size": len(data),
                      "ocr_used": False, "ocr_pages": [], "text_pages": list(range(1, len(parts) + 1)),
                      "extraction_method": "text"}
        method = "pypdf fallback + project clean_text (PyMuPDF not installed)"
    extraction["file_name"] = os.path.basename(pdf_path)
    return extraction, clean_text(extraction["text"]), method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--prep-only", action="store_true")
    args = ap.parse_args()

    from utils.chunker import create_chunks_hierarchical
    extraction, cleaned, method = extract_and_clean(args.pdf)
    print(f"Extraction: {method}; pages={extraction['num_pages']}; cleaned chars={len(cleaned)}")

    nclean = norm(cleaned)
    for q, phrase in QUESTIONS:
        print(f"  phrase in cleaned text ({norm(phrase) in nclean}): {phrase!r}")

    all_chunks = {}
    for size in CHUNK_SIZES:
        chunks = create_chunks_hierarchical(doc_meta=extraction, cleaned_text=cleaned,
                                            chunk_size=size, overlap=OVERLAP)
        all_chunks[size] = chunks
        ids = [c["chunk_id"] for c in chunks]
        rel = {p: [c["chunk_id"] for c in chunks if norm(p) in norm(c["text"])] for _, p in QUESTIONS}
        print(f"size={size}: chunks={len(chunks)} unique_ids={len(set(ids)) == len(ids)}")
        for p, r in rel.items():
            print(f"    {len(r)} relevant chunk(s) for {p!r}: {r}")
    if args.prep_only:
        return

    # ---- full experiment (needs Ollama + chromadb + rank_bm25) ----
    from utils.embedding import embed_chunks
    from utils import retrieval
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    doc_hash = hashlib.sha256(open(args.pdf, "rb").read()).hexdigest()
    rows, detail = [], []
    for size in CHUNK_SIZES:
        chunks = all_chunks[size]
        persist = os.path.join(OUT_DIR, f"chroma_{size}")
        shutil.rmtree(persist, ignore_errors=True)
        embed_chunks(chunks, collection_name="aoa_eval", persist_dir=persist,
                     document_name=extraction["file_name"], document_hash=doc_hash)
        relevant_ids = {p: {c["chunk_id"] for c in chunks if norm(p) in norm(c["text"])} for _, p in QUESTIONS}
        recalls, precs, rrs, h1s = [], [], [], []
        for q, phrase in QUESTIONS:
            res = retrieval.retrieve(q, collection_name="aoa_eval", persist_dir=persist,
                                     top_n_final=TOP_K, use_reranker=False)
            got = [r["chunk_id"] for r in res]
            hits = sum(1 for g in got if g in relevant_ids[phrase])
            total_rel = len(relevant_ids[phrase])
            # Recall@5: fraction of relevant chunks (those containing the phrase) found in top 5.
            # Precision@5: relevant chunks in top 5 / 5.
            recalls.append(hits / total_rel if total_rel else 0.0)
            precs.append(hits / TOP_K)
            rank = next((i for i, g in enumerate(got, 1) if g in relevant_ids[phrase]), None)
            rrs.append(1.0 / rank if rank else 0.0)   # reciprocal rank of first relevant chunk
            h1s.append(1.0 if rank == 1 else 0.0)     # relevant chunk ranked first
            detail.append((size, q, total_rel, hits, got, rank))
        rows.append({"chunk_size": size, "overlap": OVERLAP, "num_chunks": len(chunks),
                     "recall_at_5": round(sum(recalls) / len(recalls), 4),
                     "precision_at_5": round(sum(precs) / len(precs), 4),
                     "mrr": round(sum(rrs) / len(rrs), 4),
                     "hit_at_1": round(sum(h1s) / len(h1s), 4)})

    with open(os.path.join(OUT_DIR, f"chunk_results{SUFFIX}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, f"chunk_results_per_question{SUFFIX}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_size", "question", "relevant_chunks_in_index", "relevant_in_top5", "top5_chunk_ids", "rank_of_first_relevant"])
        for d in detail:
            w.writerow([d[0], d[1], d[2], d[3], ";".join(d[4]), d[5] if d[5] else "not_in_top5"])

    xs = [r["chunk_size"] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [r["recall_at_5"] for r in rows], marker="o", label="Recall@5")
    plt.plot(xs, [r["precision_at_5"] for r in rows], marker="s", label="Precision@5")
    plt.xticks(xs)
    plt.xlabel("Chunk size (characters, overlap=100)")
    plt.ylabel("Score")
    plt.title("Chunk size vs retrieval accuracy (hybrid dense+BM25, no reranker)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"chunk_vs_retrieval_accuracy{SUFFIX}.png"), dpi=200)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()