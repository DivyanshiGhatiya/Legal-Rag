import time
import statistics
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fitz import open as open_pdf

from utils.chunker import create_chunks_hierarchical
from utils.embedding import (
    compute_file_hash,
    embed_chunks,
)
from utils.retrieval import retrieve
from utils.pdf_processor import clean_text


# ===== SETTINGS =====
PDF_PATH = Path("data/uploads/aoa.pdf")
QUESTIONS_PATH = Path("test_questions.txt")

CHUNK_SIZES = [500, 1000, 1500, 2000]
OVERLAP = 100
TOP_K = 5
RUN_COUNT = 3
OUTPUT_DIR = Path("response_time_results")


def extract_pdf_text(pdf_path):
    document = open_pdf(pdf_path)

    text_parts = []
    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text").strip()
        text_parts.append(f"[PAGE {page_number}]\n{page_text}")

    document.close()
    return "\n".join(text_parts)


def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not questions:
        raise RuntimeError("test_questions.txt does not contain questions.")

    return questions


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}. "
            "Upload the PDF to data/uploads/."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_bytes = PDF_PATH.read_bytes()
    document_hash = compute_file_hash(pdf_bytes)
    raw_text = extract_pdf_text(PDF_PATH)
    cleaned_text = clean_text(raw_text)
    questions = load_questions()

    document_metadata = {
        "file_name": PDF_PATH.name,
        "document_hash": document_hash,
        "num_pages": open_pdf(PDF_PATH).page_count,
    }

    all_results = []

    for chunk_size in CHUNK_SIZES:
        print(f"\nTesting chunk size: {chunk_size}")

        collection_name = f"aoa_response_time_{chunk_size}"
        persist_dir = str(OUTPUT_DIR / f"chroma_{chunk_size}")

        chunks = create_chunks_hierarchical(
            doc_meta=document_metadata,
            cleaned_text=cleaned_text,
            chunk_size=chunk_size,
            overlap=OVERLAP,
        )

        print(f"Created chunks: {len(chunks)}")

        embed_chunks(
            chunks,
            collection_name=collection_name,
            persist_dir=persist_dir,
            document_name=PDF_PATH.name,
            document_hash=document_hash,
        )

        question_times = []

        for question in questions:
            # Warm-up run: not included in final timing.
            retrieve(
                question,
                collection_name=collection_name,
                persist_dir=persist_dir,
                top_n_final=TOP_K,
                use_reranker=False,
            )

            measured_times = []

            for _ in range(RUN_COUNT):
                start_time = time.perf_counter()

                retrieve(
                    question,
                    collection_name=collection_name,
                    persist_dir=persist_dir,
                    top_n_final=TOP_K,
                    use_reranker=False,
                )

                elapsed_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                measured_times.append(elapsed_ms)

            average_ms = statistics.mean(measured_times)

            question_times.append(average_ms)

            print(
                f"  Question: {question}"
                f"\n  Average: {average_ms:.2f} ms"
            )

        average_response_time = statistics.mean(question_times)

        all_results.append({
            "chunk_size": chunk_size,
            "average_response_time_ms": round(
                average_response_time,
                2,
            ),
            "chunk_count": len(chunks),
        })

        print(
            f"Chunk size {chunk_size} average "
            f"response time: {average_response_time:.2f} ms"
        )

    results_file = OUTPUT_DIR / "chunk_vs_response_time.csv"

    with results_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "chunk_size",
                "average_response_time_ms",
                "chunk_count",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    data = pd.read_csv(results_file)

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        data["chunk_size"].astype(str),
        data["average_response_time_ms"],
        color="#4C78A8",
    )

    plt.xlabel("Chunk Size (Characters)")
    plt.ylabel("Average Response Time (ms)")
    plt.title("Chunk Size vs Response Time")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    chart_file = OUTPUT_DIR / "chunk_vs_response_time.png"
    plt.savefig(chart_file, dpi=300)
    plt.close()

    print("\nFINAL RESULTS")
    print(data.to_string(index=False))
    print(f"\nCSV: {results_file.resolve()}")
    print(f"Graph: {chart_file.resolve()}")


if __name__ == "__main__":
    main()