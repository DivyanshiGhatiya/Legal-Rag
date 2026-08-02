# utils/chunker.py
"""
Hierarchy‑aware chunking utilities for the Legal RAG project (Week 3).

Main entry point:
    create_chunks_hierarchical(doc_meta, cleaned_text, chunk_size=1000, overlap=200)
        → List[dict]  # each dict matches the example schema

Helper functions:
    _detect_headings(text)                → List[Tuple[int, str, int]]
        Returns (start_index, heading_text, level) for every heading line.
    _split_by_headings(text, headings)   → List[Tuple[str, List[str]]]
        Returns (block_text, hierarchy_path) for each heading‑delimited block.
    _sliding_window(text, size, overlap) → List[str]
        Simple character‑based sliding window (used as a fallback inside large blocks).
"""

from typing import List, Tuple, Dict, Any
import os
import re
import json


# ----------------------------------------------------------------------
# 1️⃣ Heading detection (regex‑based, no external NLP libs)
# ----------------------------------------------------------------------
_HEADING_PATTERNS = [
    # 1. Numbered sections: "1.", "2.1", "3.4.5", possibly followed by a title
    re.compile(r'^(\d+(?:\.\d+)+)\s+(.+)$', re.MULTILINE),
    # 2. All‑caps lines that are at least 3 chars long and not just numbers
    re.compile(r'^([A-Z][A-Z\s\d\-]{2,})$', re.MULTILINE),
    # 3. Lines ending with a colon that look like a heading (e.g., "Definitions:")
    re.compile(r'^([A-Z][a-zA-Z\s\d\-]+:)\s*$', re.MULTILINE),
]


def _detect_headings(text: str) -> List[Tuple[int, str, int]]:
    """
    Scan ``text`` line‑by‑line and return a list of tuples:
        (start_index_in_text, heading_text, heading_level)

    ``heading_level`` is an integer where 1 = top‑level (e.g., "1." or "PART I"),
    2 = sub‑level (e.g., "1.1" or a subsection under a PART), etc.
    The heuristic is deliberately simple – it works well for most numbered
    legal documents and for ALL‑CAPS headings that are common in contracts.
    """
    headings: List[Tuple[int, str, int]] = []

    # We need the absolute character offset of each line start.
    # ``re.finditer`` on the whole string gives us the start position of each match.
    for pattern in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            start = m.start()
            # Normalise heading text – strip surrounding whitespace
            heading_text = m.group(0).strip()

            # Determine level:
            #   * If the match starts with a number like "1.2.3", level = number of dots + 1
            #   * Otherwise (ALL‑CAPS or "...:" ) we treat it as level 1.
            num_match = re.match(r'^(\d+(?:\.\d+)*)', heading_text)
            if num_match:
                level = num_match.group(1).count('.') + 1
            else:
                level = 1   # treat caps/colon headings as top‑level for simplicity

            headings.append((start, heading_text, level))

    # Sort by position and remove duplicates that may have been caught by multiple patterns
    headings.sort(key=lambda x: x[0])
    uniq: List[Tuple[int, str, int]] = []
    prev_start = -1
    for start, txt, lvl in headings:
        if start != prev_start:          # avoid duplicate matches at same index
            uniq.append((start, txt, lvl))
            prev_start = start
    return uniq


# ----------------------------------------------------------------------
# 2️⃣ Split the text on detected headings, preserving the hierarchy path
# ----------------------------------------------------------------------
def _split_by_headings(text: str,
                       headings: List[Tuple[int, str, int]]) -> List[Tuple[str, List[str]]]:
    """
    Break ``text`` into chunks that start at each heading.
    Returns a list of tuples:
        (block_text, hierarchy_path)

    * ``block_text`` – the raw text from this heading up to (but not including)
      the next heading, or the end of the document.
    * ``hierarchy_path`` – list of heading strings that precede this block,
      ordered from top‑level to the most recent heading.
    """
    if not headings:
        # No headings found → treat the whole doc as a single block with empty hierarchy
        return [(text, [])]

    blocks: List[Tuple[str, List[str]]] = []
    for i, (start, heading_text, level) in enumerate(headings):
        # Determine where this block ends
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)

        block_text = text[start:end].strip()

        # Build the hierarchy path: all headings of a *lower* or *equal* level that
        # appear before the current heading.
        hierarchy: List[str] = []
        for j in range(i):
            prev_start, prev_heading, prev_level = headings[j]
            if prev_level <= level:          # keep ancestors (including same level from earlier sections)
                hierarchy.append(prev_heading)

        blocks.append((block_text, hierarchy))

    return blocks


# ----------------------------------------------------------------------
# 3️⃣ Simple sliding‑window fallback (used only when a heading block is too big)
# ----------------------------------------------------------------------
def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0 <= overlap < chunk_size):
        raise ValueError("must have 0 <= overlap < chunk_size")

    slices: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        slices.append(text[start:end])
        start += chunk_size - overlap
        if start < 0:   # safeguard against pathological overlap values
            start = 0
    return slices


# ----------------------------------------------------------------------
# 4️⃣ Public API – hierarchical chunk creator
# ----------------------------------------------------------------------
def create_chunks_hierarchical(
    doc_meta: Dict[str, Any],
    cleaned_text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Convert a cleaned document into a list of chunk dictionaries that respect
    the document’s heading hierarchy.

    Parameters
    ----------
    doc_meta : dict
        Metadata produced by the ingestion step (must contain ``file_name``;
        other keys like ``num_pages``, ``file_size`` are copied unchanged).
    cleaned_text : str
        Output of ``utils.pdf_processor.clean_text``.
    chunk_size : int, optional
        Target maximum size (in characters) for each chunk. Default = 1000.
    overlap : int, optional
        Number of characters that consecutive chunks overlap (used only when
        a heading‑based block exceeds ``chunk_size``). Default = 200.

    Returns
    -------
    List[dict]
        Each dict follows the example schema:

        {
            "chunk_id": "DOC001_CHUNK_001",
            "document_id": "DOC001",
            "file_name": "contract.pdf",
            "section": "4.2",               # optional – derived from hierarchy
            "subsection": "(a)",            # optional – derived from hierarchy
            "chunk_index": 1,
            "chunk_length": 950,
            "text": "...the quick brown fox...",
            "metadata": {
                "file_name": "...",
                "num_pages": ...,
                "file_size": ...,
                "hierarchy": ["Part II", "Section 4.2", "(a)"]
            }
        }
    """
    if not cleaned_text:
        return []

    # ------------------------------------------------------------------
    # Step 1 – detect headings and split the document into heading‑based blocks
    # ------------------------------------------------------------------
    headings = _detect_headings(cleaned_text)
    blocks = _split_by_headings(cleaned_text, headings)

    # ------------------------------------------------------------------
    # Step 2 – generate a stable document ID (filename without extension)
    # ------------------------------------------------------------------
    file_name = doc_meta.get("file_name", "unknown")
    document_id = os.path.splitext(file_name)[0]

    # ------------------------------------------------------------------
    # Step 3 – walk through each block, possibly chunk it further with sliding window
    # ------------------------------------------------------------------
    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 1   # global chunk index for this document

    for block_text, hierarchy in blocks:
        # If the block is small enough, keep it as‑one chunk
        if len(block_text) <= chunk_size:
            chunks_to_add = [block_text]
        else:
            # Otherwise fall back to sliding window inside this block
            chunks_to_add = _sliding_window(block_text, chunk_size=chunk_size, overlap=overlap)

        for chunk_text in chunks_to_add:
            # Build a readable chunk_id: DOCNAME_CHUNK_001, DOCNAME_CHUNK_002, …
            chunk_id = f"{document_id}_CHUNK_{chunk_counter:03d}"
            chunk_counter += 1

            # Derive optional “section” / “subsection” fields from hierarchy
            # (just the last two levels if they exist – you can adapt this)
            section = hierarchy[-2] if len(hierarchy) >= 2 else ""
            subsection = hierarchy[-1] if len(hierarchy) >= 1 else ""

            chunk_obj: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "file_name": file_name,
                "section": section,
                "subsection": subsection,
                "chunk_index": len([c for c in all_chunks if c["document_id"] == document_id]) + 1,
                "chunk_length": len(chunk_text),
                "text": chunk_text,
                "metadata": {
                    **doc_meta,                     # copy original metadata (file size, pages, etc.)
                    "hierarchy": hierarchy,        # full path of headings that precede this chunk
                },
            }
            all_chunks.append(chunk_obj)

    return all_chunks


# ----------------------------------------------------------------------
# 5️⃣ Persist the chunk list as JSON (same signature as the old helper)
# ----------------------------------------------------------------------
def save_chunked(
    chunks: List[Dict[str, Any]],
    output_dir: str = "chunked_data",
) -> str:
    """
    Write ``chunks`` to a JSON file named "<document_id>_chunks.json`` inside
    ``output_dir``.  Returns the absolute path to the written file.
    """
    if not chunks:
        raise ValueError("No chunks to save.")

    os.makedirs(output_dir, exist_ok=True)

    # All chunks belong to the same document – use the first one's document_id
    doc_id = chunks[0]["document_id"]
    out_filename = f"{doc_id}_chunks.json"
    out_path = os.path.join(output_dir, out_filename)

    payload = {"chunks": chunks}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return os.path.abspath(out_path)
