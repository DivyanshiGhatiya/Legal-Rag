# utils/pdf_processor.py
"""
PDF ingestion utilities for the Legal RAG project (Week 2).

Functions:
- validate_pdf: quick sanity check before processing.
- extract_text_and_metadata: pull raw text + basic stats via PyMuPDF.
- clean_text: normalize whitespace while preserving paragraph breaks.
- save_processed: write metadata + cleaned text to a JSON file.
"""

import os
import json
from typing import Tuple, Dict, Any

import fitz  # PyMuPDF


def validate_pdf(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Verify that the supplied bytes represent a readable PDF file.

    Parameters
    ----------
    file_bytes: bytes
        The raw upload content.

    Returns
    -------
    (is_valid, error_message)
        is_valid – True if the file passes all checks.
        error_message – empty string if valid, otherwise a human‑readable reason.
    """
    if not file_bytes:
        return False, "Uploaded file is empty."

    # Quick signature check – PDF files start with "%PDF"
    if not file_bytes.startswith(b"%PDF"):
        return False, "File does not appear to be a PDF (missing %PDF header)."

    try:
        # Attempt to open with PyMuPDF; this will raise if the file is corrupt.
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if doc.page_count == 0:
            doc.close()
            return False, "PDF contains zero pages."
        doc.close()
    except Exception as exc:
        return False, f"Unable to open PDF: {exc}"

    return True, ""  # No error


def extract_text_and_metadata(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extract raw text and collect basic metadata from a PDF.

    Parameters
    ----------
    file_bytes: bytes
        Validated PDF content.

    Returns
    -------
    dict with keys:
        - "text": concatenated raw text from all pages.
        - "num_pages": integer page count.
        - "file_size": size in bytes of the original upload.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text_parts = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # zero‑based indexing
        text = page.get_text("text")   # plain text extraction
        full_text_parts.append(text)

    doc.close()

    raw_text = "\n".join(full_text_parts)
    return {
        "text": raw_text,
        "num_pages": len(full_text_parts),
        "file_size": len(file_bytes),
    }


def clean_text(raw_text: str) -> str:
    """
    Normalize whitespace while preserving logical paragraph breaks.

    Steps:
    1. Split the raw text into lines.
    2. Strip each line; if the stripped line is empty → treat as a paragraph break.
    3. Join non‑empty lines with a single space.
    4. Finally, join the resulting "paragraphs" with two newlines.

    This removes extra spaces/tabs, collapses multiple blank lines to a single
    paragraph separator, and keeps the document readable.
    """
    lines = raw_text.splitlines()
    paragraphs = []
    current = []

    for line in lines:
        stripped = line.strip()
        if stripped:  # line with content
            # Collapse internal whitespace
            collapsed = " ".join(stripped.split())
            current.append(collapsed)
        else:  # blank line → possible paragraph break
            if current:
                paragraphs.append(" ".join(current))
                current = []

    # Append any leftover content
    if current:
        paragraphs.append(" ".join(current))

    # Join paragraphs with a blank line between them
    cleaned = "\n\n".join(paragraphs)
    return cleaned


def save_processed(
    metadata: Dict[str, Any],
    cleaned_text: str,
    output_dir: str = "processed_data",
) -> str:
    """
    Save metadata + cleaned text as a JSON file.

    Parameters
    ----------
    metadata: dict
        Output of ``extract_text_and_metadata`` (includes file size, page count, etc.).
    cleaned_text: str
        The cleaned text from ``clean_text``.
    output_dir: str, optional
        Directory where the JSON will be written. Created if missing.

    Returns
    -------
    str
        Absolute path to the written JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build a safe filename: original name (without extension) + _processed.json
    # The caller should have supplied the original filename in metadata["file_name"].
    original_name = metadata.get("file_name", "uploaded")
    # Strip any path components and extension
    base = os.path.splitext(os.path.basename(original_name))[0]
    out_filename = f"{base}_processed.json"
    out_path = os.path.join(output_dir, out_filename)

    payload = {
        "metadata": metadata,
        "cleaned_text": cleaned_text,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return os.path.abspath(out_path)
