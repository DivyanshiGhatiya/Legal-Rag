"""Explainability helpers for the Advanced RAG application.

This module keeps attribution and uncertainty presentation independent from the
retrieval and generation implementations. It does not import Streamlit; the
UI wrapper accepts a Streamlit-like module object so the functions remain easy
to test and reuse.
"""

from __future__ import annotations

import html
<<<<<<< HEAD
import hashlib
import importlib
import json
import logging
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
=======
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
>>>>>>> ba4cfaf (Add explainability helpers)


_TOKEN_PATTERN = re.compile(r"\S+")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
_UNCERTAINTY_TERMS = {
    "may",
    "might",
    "possibly",
    "perhaps",
    "unclear",
    "unknown",
    "cannot",
    "insufficient",
    "uncertain",
}
<<<<<<< HEAD
_EXTRACTED_PAGE_PATTERN = re.compile(r"\[\s*PAGE\s+(\d+)\s*\]", re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+)\]")
_INTERNAL_STATUS_TAG_PATTERN = re.compile(
    r"\[(?:SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|CONFLICTING|NEEDS_REVIEW)\]\s*",
    re.IGNORECASE,
)
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
_CLAIM_STATUSES = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONFLICTING",
    "NEEDS_REVIEW",
}
_EVIDENCE_CANDIDATE_THRESHOLD = 0.20
_LEGAL_SECTION_HEADINGS = {
    "overview",
    "applicable legal provisions",
    "eligibility or requirements",
    "eligibility",
    "requirements",
    "procedure",
    "documents required",
    "exceptions and limitations",
    "exceptions",
    "limitations",
    "conclusion",
    "sources and citations",
}
logger = logging.getLogger(__name__)


def format_legal_answer(answer: Any) -> str:
    """Normalize presentation spacing without changing answer facts or citations."""
    if answer is None:
        return ""

    text = _INTERNAL_STATUS_TAG_PATTERN.sub(
        "",
        str(answer).replace("\r\n", "\n").replace("\r", "\n"),
    ).strip()
    if not text:
        return ""

    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    formatted: List[str] = []

    for line in lines:
        stripped = line.strip()
        heading_match = re.match(r"^#{1,6}\s*(.+?)\s*#*$", stripped)
        if heading_match:
            heading = heading_match.group(1).strip()
            normalized = heading.lower().rstrip(":")
            if normalized in _LEGAL_SECTION_HEADINGS:
                formatted.append(f"### {heading}")
                continue

        plain_heading = re.match(r"^(.{1,80}?):\s*$", stripped)
        if plain_heading:
            heading = plain_heading.group(1).strip()
            if heading.lower() in _LEGAL_SECTION_HEADINGS:
                formatted.append(f"### {heading}")
                continue

        formatted.append(line)

    output: List[str] = []
    for line in formatted:
        if not line.strip():
            if output and output[-1] != "" :
                output.append("")
            continue

        previous = output[-1].strip() if output else ""
        current_is_heading = bool(re.match(r"^#{1,6}\s", line))
        previous_is_heading = bool(re.match(r"^#{1,6}\s", previous))
        current_is_list = bool(re.match(r"^(?:[-*+]\s|\d+[.)]\s)", line))
        previous_is_list = bool(re.match(r"^(?:[-*+]\s|\d+[.)]\s)", previous))

        if output and (current_is_heading or previous_is_heading) and output[-1] != "":
            output.append("")
        elif output and current_is_list and not previous_is_list and previous:
            output.append("")
        elif output and previous_is_list and not current_is_list:
            output.append("")

        output.append(line)

    while output and not output[-1]:
        output.pop()
    return "\n".join(output).strip()


def extract_page_reference(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Return page data with provenance; estimated pages are never verified."""
    metadata = chunk.get("metadata") or {}
    exact_candidates = (
        chunk.get("page"),
        chunk.get("page_number"),
        chunk.get("page_num"),
        chunk.get("page_label"),
        metadata.get("page"),
        metadata.get("page_number"),
        metadata.get("page_num"),
        metadata.get("page_label"),
        (chunk.get("fitz") or {}).get("page_number")
        if isinstance(chunk.get("fitz"), dict)
        else None,
    )

    for candidate in exact_candidates:
        if candidate is None or str(candidate).strip() == "":
            continue
        match = re.search(r"\d+", str(candidate))
        if match and int(match.group(0)) > 0:
            page = int(match.group(0))
            return {
                "page_number": page,
                "page_label": f"Page {page}",
                "provenance": "EXACT_METADATA",
                "verified": True,
            }

    text = str(chunk.get("text") or "")
    match = _EXTRACTED_PAGE_PATTERN.search(text)
    if match:
        page = int(match.group(1))
        return {
            "page_number": page,
            "page_label": f"Page {page}",
            "provenance": "EXACT_TEXT_MARKER",
            "verified": True,
        }

    offset = next(
        (
            value
            for value in (
                chunk.get("char_offset"),
                chunk.get("start_char"),
                metadata.get("char_offset"),
                metadata.get("start_char"),
            )
            if value is not None and str(value).strip() != ""
        ),
        None,
    )
    total_length = next(
        (
            value
            for value in (
                chunk.get("document_char_length"),
                chunk.get("total_characters"),
                metadata.get("document_char_length"),
                metadata.get("total_characters"),
            )
            if value is not None and str(value).strip() != ""
        ),
        None,
    )
    try:
        if offset is not None and int(offset) >= 0:
            offset_value = int(offset)
            if total_length is not None and int(total_length) > 0:
                estimated_pages = max(1, (int(total_length) + 1999) // 2000)
                page = min(estimated_pages, (offset_value // 2000) + 1)
            else:
                page = (offset_value // 2000) + 1
            return {
                "page_number": page,
                "page_label": f"Estimated page {page}",
                "provenance": "ESTIMATED_CHAR_OFFSET",
                "verified": False,
            }
    except (TypeError, ValueError):
        pass

    return {
        "page_number": None,
        "page_label": "Page N/A",
        "provenance": "UNAVAILABLE",
        "verified": False,
    }
=======
_PAGE_PATTERNS = (
    re.compile(r"\bpage\s*[:#-]?\s*(\d+)\s*(?:of\s*\d+)?\b", re.IGNORECASE),
    re.compile(r"\bp\.\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"[-–—]\s*(\d+)\s*[-–—]"),
)
>>>>>>> ba4cfaf (Add explainability helpers)


def _normalize_token(token: str) -> str:
    """Return a lowercase alphanumeric representation of a token."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _query_terms(query: str) -> set[str]:
    """Extract meaningful normalized terms from a user query."""
    return {
        normalized
        for token in _WORD_PATTERN.findall(query or "")
        if len(normalized := _normalize_token(token)) > 1
    }


def extract_and_map_page_number(chunk: Dict[str, Any]) -> Tuple[int, str]:
    """Extract ``(page_number, formatted_label)`` from a chunk.

    The resolver checks common parser metadata first, then structural markers
    in the chunk text, and finally estimates the page from a character offset
    using the document's total length or a 2,000-character average page size.
    ``0`` is used when no reliable or estimable position exists.
    """
<<<<<<< HEAD
    reference = extract_page_reference(chunk)
    page_number = reference["page_number"] or 0
    return page_number, reference["page_label"]
=======
    metadata = chunk.get("metadata") or {}
    page_candidates = (
        chunk.get("page"),
        chunk.get("page_number"),
        chunk.get("page_label"),
        metadata.get("page"),
        metadata.get("page_number"),
        metadata.get("page_label"),
        metadata.get("page_num"),
        (chunk.get("fitz") or {}).get("page_number")
        if isinstance(chunk.get("fitz"), dict)
        else None,
    )

    for candidate in page_candidates:
        if candidate is None or str(candidate).strip() == "":
            continue
        value = str(candidate).strip()
        match = re.search(r"\d+", value)
        if match and int(match.group(0)) > 0:
            page = int(match.group(0))
            return page, f"Page {page}"

    text = str(chunk.get("text") or "")
    for pattern in _PAGE_PATTERNS:
        match = pattern.search(text)
        if match:
            page = int(match.group(1))
            return page, f"Page {page}"

    offset = (
        chunk.get("char_offset")
        or chunk.get("start_char")
        or metadata.get("char_offset")
        or metadata.get("start_char")
    )
    total_length = (
        chunk.get("document_char_length")
        or chunk.get("total_characters")
        or metadata.get("document_char_length")
        or metadata.get("total_characters")
    )
    try:
        if offset is not None and int(offset) >= 0:
            offset_value = int(offset)
            if total_length and int(total_length) > 0:
                estimated_pages = max(1, (int(total_length) + 1999) // 2000)
                page = min(estimated_pages, (offset_value // 2000) + 1)
            else:
                page = (offset_value // 2000) + 1
            return page, f"Page {page}"
    except (TypeError, ValueError):
        pass

    return 0, "Page N/A"
>>>>>>> ba4cfaf (Add explainability helpers)


def extract_and_map_page(chunk: Dict[str, Any]) -> str:
    """Backward-compatible string wrapper around page-number extraction."""
    return extract_and_map_page_number(chunk)[1]


def get_document_page_stats(
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Group retrieved chunks by source and return sorted referenced pages."""
    document_pages: Dict[str, set[int]] = {}
    for chunk in retrieved_chunks:
        source = _chunk_source_name(chunk)
<<<<<<< HEAD
        page_reference = extract_page_reference(chunk)
        document_pages.setdefault(source, set())
        if page_reference["verified"]:
            document_pages[source].add(page_reference["page_number"])
=======
        page_number, _ = extract_and_map_page_number(chunk)
        document_pages.setdefault(source, set())
        if page_number > 0:
            document_pages[source].add(page_number)
>>>>>>> ba4cfaf (Add explainability helpers)

    return {
        source: sorted(pages)
        for source, pages in sorted(document_pages.items())
    }


def _chunk_source_name(chunk: Dict[str, Any]) -> str:
    """Return the best available source/document name for a chunk."""
    metadata = chunk.get("metadata") or {}
    return str(
        chunk.get("file_name")
        or chunk.get("document_name")
        or metadata.get("source_name")
        or "Unknown document"
    )


def _grounding_score(claim: str, context: str) -> float:
    """Calculate a bounded lexical grounding score for a claim/context pair."""
    claim_terms = _query_terms(claim)
    context_terms = _query_terms(context)
    if not claim_terms:
        return 0.0
    exact_overlap = len(claim_terms & context_terms) / len(claim_terms)
    fuzzy_matches = sum(
        1
        for claim_term in claim_terms
        if any(
            claim_term.startswith(context_term) or context_term.startswith(claim_term)
            for context_term in context_terms
        )
    ) / len(claim_terms)
    return min(1.0, (0.75 * exact_overlap) + (0.25 * fuzzy_matches))


<<<<<<< HEAD
def split_answer_claims(answer: Any) -> List[str]:
    """Split answer text into claims, or preserve a caller's pre-split list."""
    if isinstance(answer, (list, tuple)):
        return [
            _INTERNAL_STATUS_TAG_PATTERN.sub("", str(claim)).strip()
            for claim in answer
            if _INTERNAL_STATUS_TAG_PATTERN.sub("", str(claim)).strip()
        ]
    if not isinstance(answer, str) or not answer.strip():
        return []

    answer = _INTERNAL_STATUS_TAG_PATTERN.sub("", answer)
    claims = []
    paragraphs = re.split(r"\n\s*\n", answer)
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        units = []
        if any(_BULLET_PATTERN.match(line) for line in lines):
            current = []
            for line in lines:
                if _BULLET_PATTERN.match(line) and current:
                    units.append(" ".join(current))
                    current = []
                current.append(_BULLET_PATTERN.sub("", line, count=1).strip())
            if current:
                units.append(" ".join(current))
        else:
            units = [" ".join(lines)]

        for unit in units:
            citations = _CITATION_PATTERN.findall(unit)
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", unit)
                if sentence.strip()
            ]
            for sentence in sentences:
                if citations and not _CITATION_PATTERN.search(sentence):
                    sentence += " " + " ".join(f"[{citation}]" for citation in citations)
                claims.append(sentence)
    return claims


def _normalize_chunk_id(chunk_id: Any) -> str:
    normalized = str(chunk_id).strip().lower()
    match = re.match(r"^(.*?)(\d+)$", normalized)
    return f"{match.group(1)}{int(match.group(2))}" if match else normalized


def _claim_text(claim: str) -> str:
    return _CITATION_PATTERN.sub("", claim).strip()


def _build_chunk_id_resolver(retrieved_chunks: List[Dict[str, Any]]) -> Callable[[Any], Optional[str]]:
    known_ids = {}
    suffix_ids: Dict[str, set[str]] = {}
    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id is None or not str(chunk_id).strip():
            continue
        chunk_id = str(chunk_id).strip()
        known_ids.setdefault(_normalize_chunk_id(chunk_id), chunk_id)
        suffix = re.search(r"(?:^|_)chunk_(\d+)$", chunk_id, re.IGNORECASE)
        if suffix:
            suffix_ids.setdefault(f"chunk_{int(suffix.group(1))}", set()).add(chunk_id)
    aliases = {
        suffix: next(iter(ids))
        for suffix, ids in suffix_ids.items()
        if len(ids) == 1
    }

    def resolve(chunk_id: Any) -> Optional[str]:
        normalized = _normalize_chunk_id(chunk_id)
        exact = known_ids.get(normalized)
        if exact:
            return exact
        suffix = re.search(r"(?:^|_)chunk_(\d+)$", normalized, re.IGNORECASE)
        return aliases.get(f"chunk_{int(suffix.group(1))}") if suffix else None

    return resolve


def _evidence_excerpt(text: str, claim: str, limit: int = 700) -> str:
    if len(text) <= limit:
        return text
    terms = _query_terms(claim)
    normalized_text = text.lower()
    positions = [normalized_text.find(term) for term in terms if normalized_text.find(term) >= 0]
    start = max(0, min(positions) - (limit // 4)) if positions else 0
    end = min(len(text), start + limit)
    excerpt = text[start:end]
    if start:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt += "..."
    return excerpt


def _evidence_record(
    claim: str,
    chunk: Dict[str, Any],
    lexical_score: float,
    verification_status: str,
    evidence_support_score: Optional[float] = None,
) -> Dict[str, Any]:
    page = extract_page_reference(chunk)
    return {
        "document_name": _chunk_source_name(chunk),
        "page_number": page["page_number"],
        "page_provenance": page["provenance"],
        "page_verified": page["verified"],
        "chunk_id": chunk.get("chunk_id"),
        "evidence_excerpt": _evidence_excerpt(str(chunk.get("text") or ""), claim),
        "lexical_relevance_score": round(lexical_score, 4),
        "evidence_support_score": evidence_support_score,
        "verification_status": verification_status,
        "dense_score": chunk.get("dense_score"),
        "bm25_score": chunk.get("bm25_score"),
        "rrf_score": chunk.get("rrf_score"),
        "rerank_score": chunk.get("rerank_score"),
    }


def verify_claim_evidence(
    answer_claims: Any,
    retrieved_chunks: List[Dict[str, Any]],
    verifier: Optional[Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Assess claims conservatively and validate every cited chunk ID.

    A verifier receives ``(claim, retrieved_chunks)`` and returns a mapping
    with ``status`` plus ``supporting_chunk_ids`` (and optionally
    ``evidence_scores``). Without a verifier, lexical matching only identifies
    review candidates; it never establishes support.
    """
    claims = split_answer_claims(answer_claims)
    chunks = list(retrieved_chunks or [])
    resolve_id = _build_chunk_id_resolver(chunks)
    chunks_by_id = {
        str(chunk.get("chunk_id")): chunk
        for chunk in chunks
        if chunk.get("chunk_id") is not None
    }
    output = []

    for claim in claims:
        claim_body = _claim_text(claim)
        referenced_ids = _CITATION_PATTERN.findall(claim)
        valid_citations = []
        invalid_citations = []
        for referenced_id in referenced_ids:
            resolved = resolve_id(ref     erenced_id)
            target = valid_citations if resolved else invalid_citations
            value = resolved or referenced_id.strip()
            if value not in target:
                target.append(value)

        scored_chunks = []
        for chunk in chunks:
            score = _grounding_score(claim_body, str(chunk.get("text") or ""))
            chunk_id = str(chunk.get("chunk_id", ""))
            if score >= _EVIDENCE_CANDIDATE_THRESHOLD or chunk_id in valid_citations:
                scored_chunks.append((chunk, score))

        verifier_result = None
        verifier_error = False
        if verifier is not None:
            try:
                result = verifier(claim_body, chunks)
                verifier_result = result if isinstance(result, dict) else None
                verifier_error = verifier_result is None
            except Exception as exc:
                logger.exception(
                    "pipeline_stage=claim_evidence_verification "
                    "status=failed error_type=%s",
                    type(exc).__name__,
                )
                verifier_error = True

        candidate_status = "NEEDS_REVIEW" if scored_chunks else "UNSUPPORTED"
        status = candidate_status
        verified_ids: List[str] = []
        conflicting_ids: List[str] = []
        support_scores: Dict[str, float] = {}
        verification_note = "No semantic verifier configured."

        if verifier_result is not None:
            requested_status = str(verifier_result.get("status", "NEEDS_REVIEW")).upper()
            status = requested_status if requested_status in _CLAIM_STATUSES else "NEEDS_REVIEW"
            raw_ids = verifier_result.get(
                "supporting_chunk_ids",
                verifier_result.get("supported_chunk_ids", []),
            )
            if isinstance(raw_ids, (str, int)):
                raw_ids = [raw_ids]
            for raw_id in raw_ids or []:
                resolved = resolve_id(raw_id)
                if resolved and resolved not in verified_ids:
                    verified_ids.append(resolved)
            raw_conflicting_ids = verifier_result.get("conflicting_chunk_ids", [])
            if isinstance(raw_conflicting_ids, (str, int)):
                raw_conflicting_ids = [raw_conflicting_ids]
            for raw_id in raw_conflicting_ids or []:
                resolved = resolve_id(raw_id)
                if resolved and resolved not in conflicting_ids:
                    conflicting_ids.append(resolved)
            raw_scores = verifier_result.get("evidence_scores") or {}
            if isinstance(raw_scores, dict):
                for raw_id, raw_score in raw_scores.items():
                    resolved = resolve_id(raw_id)
                    try:
                        if resolved:
                            support_scores[resolved] = float(raw_score)
                    except (TypeError, ValueError):
                        continue
            verification_note = str(verifier_result.get("rationale") or "Verifier supplied a verdict.")
            if status in {"SUPPORTED", "PARTIALLY_SUPPORTED"} and not verified_ids:
                status = "NEEDS_REVIEW"
                verification_note = "Verifier returned support without a valid retrieved chunk ID."
            if status == "CONFLICTING" and not conflicting_ids:
                status = "NEEDS_REVIEW"
                verification_note = "Verifier reported conflict without a valid retrieved chunk ID."
        elif verifier_error:
            status = "NEEDS_REVIEW"
            verification_note = "Verifier failed; evidence was not verified."

        score_by_id = {
            str(chunk.get("chunk_id")): score
            for chunk, score in scored_chunks
            if chunk.get("chunk_id") is not None
        }
        chunk_by_id = {
            str(chunk.get("chunk_id")): chunk
            for chunk in chunks
            if chunk.get("chunk_id") is not None
        }
        evidence_ids = verified_ids if status in {"SUPPORTED", "PARTIALLY_SUPPORTED"} else []
        supporting_evidence = [
            _evidence_record(
                claim_body,
                chunk_by_id[chunk_id],
                score_by_id.get(chunk_id, _grounding_score(claim_body, str(chunk_by_id[chunk_id].get("text") or ""))),
                status,
                support_scores.get(chunk_id),
            )
            for chunk_id in evidence_ids
        ]
        candidate_evidence = [
            _evidence_record(claim_body, chunk, score, status if verifier_result else "NEEDS_REVIEW")
            for chunk, score in scored_chunks
        ]
        conflicting_evidence = [
            _evidence_record(
                claim_body,
                chunk_by_id[chunk_id],
                score_by_id.get(
                    chunk_id,
                    _grounding_score(
                        claim_body,
                        str(chunk_by_id[chunk_id].get("text") or ""),
                    ),
                ),
                "CONFLICTING",
                support_scores.get(chunk_id),
            )
            for chunk_id in conflicting_ids
            if chunk_id in chunk_by_id
        ]

        source_records = supporting_evidence or conflicting_evidence or candidate_evidence
        source_labels = []
        for evidence in source_records:
            page_label = (
                f"Page {evidence['page_number']}"
                if evidence["page_verified"]
                else "Page N/A"
            )
            label = f"{evidence['document_name']} • {page_label}"
            if label not in source_labels:
                source_labels.append(label)
        mapped_source = (
            "; ".join(source_labels)
            if supporting_evidence
            else f"Needs review: {'; '.join(source_labels)}" if source_labels
            else "No source found"
        )
        lexical_score = max((score for _, score in scored_chunks), default=0.0)
        output.append({
            "claim": claim,
            "status": status,
            "verification_note": verification_note,
            "conflict_type": (
                str(verifier_result.get("conflict_type") or "DIRECT_OR_UNSPECIFIED")
                if verifier_result and status == "CONFLICTING"
                else None
            ),
            "attribution_score": round(lexical_score, 4),
            "lexical_relevance_score": round(lexical_score, 4),
            "mapped_source": mapped_source,
            "cited_chunk_id": (
                valid_citations[0]
                if valid_citations
                else verified_ids[0] if verified_ids else None
            ),
            "citation_ids": [str(item).strip() for item in referenced_ids],
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "supporting_evidence": supporting_evidence,
            "candidate_evidence": candidate_evidence,
            "conflicting_evidence": conflicting_evidence,
        })
    return output


def compute_claim_attribution(
    answer_claims: Any,
    retrieved_chunks: List[Dict[str, Any]],
    verifier: Optional[Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Backward-compatible entry point for multi-citation claim attribution.

    Without a semantic verifier, lexical matches are marked ``NEEDS_REVIEW``
    rather than being represented as verified support.
    """
    return verify_claim_evidence(answer_claims, retrieved_chunks, verifier)


def build_retrieval_trace(
    retrieved_chunks: List[Dict[str, Any]],
    retrieval_metadata: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Normalize actual retrieval output without synthesizing missing scores."""
    chunks = list(retrieved_chunks or [])
    metadata_by_id = {}
    if isinstance(retrieval_metadata, dict):
        metadata_by_id = retrieval_metadata
    elif isinstance(retrieval_metadata, list):
        metadata_by_id = {
            str(item.get("chunk_id")): item
            for item in retrieval_metadata
            if isinstance(item, dict) and item.get("chunk_id") is not None
        }

    component_names = {
        "dense_score": "dense_search()",
        "bm25_score": "bm25_search()",
        "rrf_score": "rrf_fuse()",
        "rerank_score": "rerank() / Cross-Encoder",
    }
    trace = []
    for index, original in enumerate(chunks, start=1):
        chunk = dict(original)
        extra = metadata_by_id.get(str(chunk.get("chunk_id")), {})
        if isinstance(extra, dict):
            for key, value in extra.items():
                if chunk.get(key) is None and key not in {"metadata", "text"}:
                    chunk[key] = value
        page = extract_page_reference(chunk)
        stages = chunk.get("retrieval_stages")
        if not stages:
            stages = [
                stage
                for key, stage in (
                    ("dense_score", "DENSE"),
                    ("bm25_score", "BM25"),
                    ("rrf_score", "RRF"),
                    ("rerank_score", "CROSS_ENCODER"),
                )
                if chunk.get(key) is not None
            ]
        elif isinstance(stages, str):
            stages = [stages]
        missing_scores = {
            key: component_names[key]
            for key in component_names
            if chunk.get(key) is None
        }
        trace.append({
            "final_rank": chunk.get("final_rank") or index,
            "rank_source": "pipeline" if chunk.get("final_rank") is not None else "retrieval_order",
            "chunk_id": chunk.get("chunk_id"),
            "source_document": _chunk_source_name(chunk),
            "page_number": page["page_number"],
            "page_provenance": page["provenance"],
            "dense_rank": chunk.get("dense_rank"),
            "dense_score": chunk.get("dense_score"),
            "bm25_rank": chunk.get("bm25_rank"),
            "bm25_score": chunk.get("bm25_score"),
            "rrf_rank": chunk.get("rrf_rank"),
            "rrf_score": chunk.get("rrf_score"),
            "cross_encoder_score": chunk.get("rerank_score"),
            "retrieval_stages": list(stages),
            "missing_score_sources": missing_scores,
        })
    return trace


def generate_evidence_report(
    answer_claims: Any,
    retrieved_chunks: List[Dict[str, Any]],
    claim_attributions: Optional[List[Dict[str, Any]]] = None,
    retrieval_metadata: Optional[Any] = None,
    contradiction_verifier: Optional[Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]] = None,
    citation_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize evidence coverage and optional contradiction-verifier results."""
    claims = split_answer_claims(answer_claims)
    attributions = list(
        claim_attributions
        if claim_attributions is not None
        else compute_claim_attribution(claims, retrieved_chunks)
    )
    chunks = list(retrieved_chunks or [])
    resolve_id = _build_chunk_id_resolver(chunks)
    chunks_by_id = {
        str(chunk.get("chunk_id")): chunk
        for chunk in chunks
        if chunk.get("chunk_id") is not None
    }
    contradictions = []
    contradiction_reviews = []
    verifier_failures = 0
    evaluated_claims = 0

    for attribution in attributions:
        if attribution.get("status") != "CONFLICTING":
            continue
        for evidence in attribution.get("conflicting_evidence", []):
            contradictions.append({
                "claim": attribution["claim"],
                "status": "CONFLICTING",
                "conflict_type": attribution.get("conflict_type") or "VERIFIER_REPORTED",
                "evidence_excerpt": evidence["evidence_excerpt"],
                "document_name": evidence["document_name"],
                "chunk_id": evidence["chunk_id"],
                "page_number": evidence["page_number"],
                "page_provenance": evidence["page_provenance"],
                "rationale": attribution["verification_note"],
            })

    if contradiction_verifier is not None:
        for attribution in attributions:
            try:
                verdict = contradiction_verifier(_claim_text(attribution["claim"]), chunks)
                if not isinstance(verdict, dict):
                    raise TypeError("Verifier response must be a mapping")
            except Exception as exc:
                logger.exception(
                    "pipeline_stage=contradiction_verification "
                    "status=failed error_type=%s",
                    type(exc).__name__,
                )
                verifier_failures += 1
                continue
            evaluated_claims += 1
            verdict_status = str(verdict.get("status", "NEEDS_REVIEW")).upper()
            raw_ids = verdict.get("conflicting_chunk_ids", [])
            if isinstance(raw_ids, (str, int)):
                raw_ids = [raw_ids]
            valid_ids = []
            invalid_ids = []
            for raw_id in raw_ids or []:
                resolved = resolve_id(raw_id)
                if resolved:
                    if resolved not in valid_ids:
                        valid_ids.append(resolved)
                else:
                    invalid_ids.append(str(raw_id))

            if verdict_status == "CONFLICTING" and valid_ids:
                attribution["status"] = "CONFLICTING"
                for chunk_id in valid_ids:
                    chunk = chunks_by_id[chunk_id]
                    page = extract_page_reference(chunk)
                    if not any(
                        item["claim"] == attribution["claim"]
                        and item["chunk_id"] == chunk_id
                        for item in contradictions
                    ):
                        contradictions.append({
                            "claim": attribution["claim"],
                            "status": "CONFLICTING",
                            "conflict_type": verdict.get("conflict_type", "DIRECT_OR_UNSPECIFIED"),
                            "evidence_excerpt": _evidence_excerpt(
                                str(chunk.get("text") or ""),
                                _claim_text(attribution["claim"]),
                            ),
                            "document_name": _chunk_source_name(chunk),
                            "chunk_id": chunk_id,
                            "page_number": page["page_number"],
                            "page_provenance": page["provenance"],
                            "rationale": str(verdict.get("rationale") or ""),
                        })
            elif verdict_status in {"CONFLICTING", "NEEDS_REVIEW"} or verdict_status not in {
                "NO_CONTRADICTION",
                "CLEAR",
                "NOT_CONFLICTING",
            }:
                attribution["status"] = "NEEDS_REVIEW"
                review_evidence = []
                for chunk_id in valid_ids:
                    chunk = chunks_by_id[chunk_id]
                    page = extract_page_reference(chunk)
                    review_evidence.append({
                        "document_name": _chunk_source_name(chunk),
                        "chunk_id": chunk_id,
                        "page_number": page["page_number"],
                        "page_provenance": page["provenance"],
                        "evidence_excerpt": _evidence_excerpt(
                            str(chunk.get("text") or ""),
                            _claim_text(attribution["claim"]),
                        ),
                    })
                contradiction_reviews.append({
                    "claim": attribution["claim"],
                    "status": "NEEDS_REVIEW",
                    "conflicting_chunk_ids": valid_ids,
                    "invalid_chunk_ids": invalid_ids,
                    "evidence": review_evidence,
                    "rationale": str(verdict.get("rationale") or "Potential inconsistency needs human review."),
                })

    counts = {status: 0 for status in _CLAIM_STATUSES}
    claims_without_evidence = []
    claims_without_citations = []
    invalid_citations = []
    cited_claim_count = 0
    for attribution in attributions:
        status = attribution.get("status", "NEEDS_REVIEW")
        if status not in counts:
            status = "NEEDS_REVIEW"
        counts[status] += 1
        if status == "UNSUPPORTED":
            claims_without_evidence.append(attribution["claim"])
        if attribution.get("valid_citations"):
            cited_claim_count += 1
        else:
            claims_without_citations.append(attribution["claim"])
        if attribution.get("invalid_citations"):
            invalid_citations.append({
                "claim": attribution["claim"],
                "chunk_ids": list(attribution["invalid_citations"]),
            })
    if citation_report:
        for citation in citation_report.get("unknown_citations", []) or []:
            invalid_citations.append({
                "claim": "Answer-level citation validation",
                "chunk_ids": [str(citation)],
            })

    total = len(attributions)
    sufficient_count = counts["SUPPORTED"]
    coverage = (sufficient_count / total * 100.0) if total else 0.0
    citation_coverage = (cited_claim_count / total * 100.0) if total else 0.0
    if not total:
        overall_status = "NOT_EVALUATED"
    elif counts["CONFLICTING"]:
        overall_status = "CONFLICTING"
    elif counts["UNSUPPORTED"]:
        overall_status = "INSUFFICIENT_EVIDENCE"
    elif counts["NEEDS_REVIEW"] or counts["PARTIALLY_SUPPORTED"]:
        overall_status = "NEEDS_REVIEW"
    else:
        overall_status = "SUPPORTED"

    if contradictions:
        contradiction_status = "CONFLICTS_FOUND"
    elif contradiction_reviews:
        contradiction_status = "NEEDS_REVIEW"
    elif contradiction_verifier is None or verifier_failures == total:
        contradiction_status = "NOT_EVALUATED"
    elif verifier_failures:
        contradiction_status = "PARTIALLY_EVALUATED"
    else:
        contradiction_status = "NO_CONTRADICTION_FOUND"

    if overall_status == "SUPPORTED" and contradiction_status != "NO_CONTRADICTION_FOUND":
        overall_status = "NEEDS_REVIEW"

    suggested_actions = []
    if claims_without_evidence:
        suggested_actions.append("Retrieve additional evidence, qualify, or abstain on unsupported claims.")
    if counts["NEEDS_REVIEW"] or counts["PARTIALLY_SUPPORTED"]:
        suggested_actions.append("Have a reviewer verify claims marked for review or partial support.")
    if contradictions or contradiction_reviews:
        suggested_actions.append("Review potentially conflicting provisions, dates, jurisdiction, and scope; do not infer precedence automatically.")
    if invalid_citations or claims_without_citations:
        suggested_actions.append("Regenerate or edit the answer so each factual claim has a valid retrieved-chunk citation.")
    if not suggested_actions:
        suggested_actions.append("No follow-up is indicated by the configured checks; legal review may still be appropriate.")

    return {
        "overall_status": overall_status,
        "evidence_coverage": {
            "total_claims": total,
            "supported_claims": counts["SUPPORTED"],
            "partially_supported_claims": counts["PARTIALLY_SUPPORTED"],
            "unsupported_claims": counts["UNSUPPORTED"],
            "needs_review_claims": counts["NEEDS_REVIEW"],
            "conflicting_claims": counts["CONFLICTING"],
            "sufficient_evidence_percentage": round(coverage, 2),
            "sufficient_evidence_definition": "Only SUPPORTED claims count; partial, conflicting, unsupported, and review claims do not.",
            "citation_coverage_percentage": round(citation_coverage, 2),
            "citation_coverage_definition": "Claims containing at least one citation ID that resolves to a retrieved chunk; this measures ID validity, not semantic support.",
        },
        "claims_without_supporting_evidence": claims_without_evidence,
        "claims_without_valid_citations": claims_without_citations,
        "invalid_citations": invalid_citations,
        "contradiction_status": contradiction_status,
        "contradiction_verifier_evaluated_claims": evaluated_claims,
        "contradiction_verifier_failures": verifier_failures,
        "contradictory_evidence": contradictions,
        "contradiction_reviews": contradiction_reviews,
        "retrieval_trace": build_retrieval_trace(chunks, retrieval_metadata),
        "claim_attributions": attributions,
        "answer_citation_report": citation_report,
        "suggested_next_actions": suggested_actions,
    }


def build_llm_verifiers(
    model: Optional[str] = None,
    client: Optional[Any] = None,
) -> Tuple[
    Callable[[str, List[Dict[str, Any]]], Dict[str, Any]],
    Callable[[str, List[Dict[str, Any]]], Dict[str, Any]],
]:
    """Build opt-in support and contradiction verifiers using project config.

    This lazily reuses ``utils.generation``'s configured Groq-compatible
    client. The callbacks make provider calls only when invoked; do not pass
    them to the UI unless sending retrieved legal text to that provider is
    acceptable for the application.
    """
    generation_module = importlib.import_module("utils.generation")

    GROQ_MODEL = generation_module.GROQ_MODEL
    client = client or generation_module._get_client()
    create_completion = getattr(generation_module, "create_chat_completion", None)

    selected_model = model or GROQ_MODEL
    verifier_cache: Dict[Tuple[Any, ...], Any] = {}

    def ask_verifier(
        claim: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cache_key = (
            claim,
            tuple(
                (
                    str(chunk.get("chunk_id") or ""),
                    hashlib.sha256(
                        str(chunk.get("text") or "").encode("utf-8")
                    ).hexdigest(),
                )
                for chunk in retrieved_chunks
            ),
        )
        cached_result = verifier_cache.get(cache_key)
        if isinstance(cached_result, Exception):
            raise cached_result
        if cached_result is not None:
            logger.info("pipeline_cache=claim_verification status=hit")
            return cached_result

        evidence = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_name": _chunk_source_name(chunk),
                "text": str(chunk.get("text") or ""),
            }
            for chunk in retrieved_chunks
        ]
        prompt = {
            "task": "claim_support_and_contradiction_detection",
            "claim": claim,
            "retrieved_evidence": evidence,
            "instructions": [
                "Use only the supplied retrieved evidence.",
                "Return one JSON object with status, supporting_chunk_ids, "
                "conflicting_chunk_ids, contradiction_status, conflict_type, "
                "and rationale. Cite only supplied chunk_id values.",
                "Assess whether evidence supports the claim and independently "
                "whether passages directly conflict with it.",
                "Use contradiction_status NO_CONTRADICTION only after checking "
                "the supplied passages; otherwise use NEEDS_REVIEW.",
                "Do not infer that one legal provision overrides another.",
                "Treat dates, jurisdiction, amendments, exceptions, and scope as material.",
                "If evidence is ambiguous or incomplete, choose NEEDS_REVIEW.",
                "Do not return a numeric probability or calibrated confidence score.",
            ],
        }
        completion_kwargs = {
            "model": selected_model,
            "temperature": 0,
            "max_tokens": 700,
            "timeout": 30,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an evidence verifier. Do not answer the legal question. "
                        "Return valid JSON only, with no markdown fences."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
            ],
        }
        started = time.perf_counter()
        try:
            if create_completion is not None:
                response = create_completion(
                    client,
                    stage="explainability_verification",
                    **completion_kwargs,
                )
            else:
                response = client.chat.completions.create(**completion_kwargs)
            content = response.choices[0].message.content or ""
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("Verifier response must be a JSON object.")
            verifier_cache[cache_key] = result
            return result
        except Exception as exc:
            verifier_cache[cache_key] = exc
            raise
        finally:
            logger.info(
                "pipeline_stage=explainability_verification "
                "duration_seconds=%.3f chunks=%d api_calls=1",
                time.perf_counter() - started,
                len(retrieved_chunks),
            )

    def verify_claim(claim: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = ask_verifier(claim, retrieved_chunks)
        status = str(result.get("status", "NEEDS_REVIEW")).upper()
        if status not in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "NEEDS_REVIEW", "CONFLICTING"}:
            status = "NEEDS_REVIEW"
        return {
            "status": status,
            "supporting_chunk_ids": result.get("supporting_chunk_ids", []),
            "conflicting_chunk_ids": result.get("conflicting_chunk_ids", []),
            "rationale": str(result.get("rationale") or ""),
            "evidence_scores": result.get("evidence_scores", {}),
        }

    def verify_contradictions(
        claim: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result = ask_verifier(claim, retrieved_chunks)
        status = str(
            result.get("contradiction_status")
            or (
                "CONFLICTING"
                if result.get("conflicting_chunk_ids")
                else "NEEDS_REVIEW"
            )
        ).upper()
        if status not in {"CONFLICTING", "NEEDS_REVIEW", "NO_CONTRADICTION", "CLEAR", "NOT_CONFLICTING"}:
            status = "NEEDS_REVIEW"
        return {
            "status": status,
            "conflicting_chunk_ids": result.get("conflicting_chunk_ids", []),
            "conflict_type": str(result.get("conflict_type") or "DIRECT_OR_UNSPECIFIED"),
            "rationale": str(result.get("rationale") or ""),
        }

    return verify_claim, verify_contradictions


def render_retrieval_trace_ui(
    st_module: Any,
    retrieved_chunks: List[Dict[str, Any]],
    retrieval_metadata: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Render actual retrieval scores and component availability in a table."""
    trace = build_retrieval_trace(retrieved_chunks, retrieval_metadata)
    with st_module.expander("Hybrid Retrieval Trace", expanded=False):
        st_module.write(
            "Dense Search exposes the pipeline value 1 - Chroma distance; interpret it "
            "according to the collection's distance metric. "
            "BM25 supplies lexical scores. RRF combines component ranks, not their "
            "score scales. The Cross-Encoder score reorders fused candidates; its "
            "scale is not directly comparable to Dense, BM25, or RRF."
        )
        if not trace:
            st_module.info("No retrieved chunks are available for a retrieval trace.")
            return trace

        rows = []
        for item in trace:
            missing = item["missing_score_sources"]
            rows.append({
                "Final rank": item["final_rank"],
                "Chunk ID": item["chunk_id"] or "N/A",
                "Source": item["source_document"],
                "Page": (
                    f"Page {item['page_number']} ({item['page_provenance']})"
                    if item["page_number"] is not None
                    else f"N/A ({item['page_provenance']})"
                ),
                "Dense rank / score": (
                    f"{item['dense_rank'] if item['dense_rank'] is not None else 'N/A'} / "
                    f"{item['dense_score'] if item['dense_score'] is not None else 'N/A'}"
                ),
                "BM25 rank / score": (
                    f"{item['bm25_rank'] if item['bm25_rank'] is not None else 'N/A'} / "
                    f"{item['bm25_score'] if item['bm25_score'] is not None else 'N/A'}"
                ),
                "RRF rank / score": (
                    f"{item['rrf_rank'] if item['rrf_rank'] is not None else 'N/A'} / "
                    f"{item['rrf_score'] if item['rrf_score'] is not None else 'N/A'}"
                ),
                "Cross-Encoder score": (
                    item["cross_encoder_score"]
                    if item["cross_encoder_score"] is not None
                    else "N/A"
                ),
                "Stages": ", ".join(item["retrieval_stages"]) or "N/A",
                "Unavailable score source": "; ".join(
                    f"{component}: {source}"
                    for component, source in missing.items()
                ) or "None",
            })
        st_module.dataframe(rows, use_container_width=True, hide_index=True)
    return trace
=======
def compute_claim_attribution(
    answer_claims: List[str],
    retrieved_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map each answer claim to its strongest retrieved chunk and source page."""
    attributions: List[Dict[str, Any]] = []
    for claim in answer_claims:
        best_chunk: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for chunk in retrieved_chunks:
            score = _grounding_score(claim, str(chunk.get("text") or ""))
            if best_chunk is None or score > best_score:
                best_chunk = chunk
                best_score = score

        if best_chunk is None:
            attributions.append({
                "claim": claim,
                "attribution_score": 0.0,
                "mapped_source": "No source found",
                "cited_chunk_id": None,
            })
            continue

        source = _chunk_source_name(best_chunk)
        _, page = extract_and_map_page_number(best_chunk)
        attributions.append({
            "claim": claim,
            "attribution_score": round(best_score, 4),
            "mapped_source": f"{source} • {page}",
            "cited_chunk_id": best_chunk.get("chunk_id"),
        })
    return attributions
>>>>>>> ba4cfaf (Add explainability helpers)


def compute_token_relevance(
    query: str,
    chunk_text: str,
) -> List[Tuple[str, float]]:
    """Score each chunk token against query terms.

    The score is deterministic and bounded between 0.0 and 1.0. Exact query
    matches receive the strongest score; prefix matches receive a softer score.
    This is an attribution visualization, not a replacement for retrieval or
    model-based semantic ranking.
    """
    terms = _query_terms(query)
    scored_tokens: List[Tuple[str, float]] = []

    for token in _TOKEN_PATTERN.findall(chunk_text or ""):
        normalized = _normalize_token(token)
        score = 0.0
        if normalized and normalized in terms:
            score = 1.0
        elif normalized and any(
            normalized.startswith(term) or term.startswith(normalized)
            for term in terms
        ):
            score = 0.65
        scored_tokens.append((token, score))

    return scored_tokens


def render_explainable_chunk_heatmap(
    word_scores: List[Tuple[str, float]],
) -> str:
    """Convert token scores into escaped light-theme HTML spans."""
    rendered_tokens: List[str] = []

    for word, raw_score in word_scores:
        score = max(0.0, min(1.0, float(raw_score)))
        alpha = 0.08 + (0.55 * score)
        safe_word = html.escape(str(word))
        rendered_tokens.append(
            f'<span style="background-color:rgba(245,158,11,{alpha:.2f});'
            'border-radius:0.2rem;padding:0.05rem 0.12rem;'
            f'color:#121110;">{safe_word}</span>'
        )

    return (
        '<div style="background:#FFFFFF;color:#121110;border:1px solid '
        'rgba(18,17,16,0.10);border-radius:0.75rem;padding:0.85rem 1rem;'
        'line-height:1.75;font-size:0.95rem;">'
        + " ".join(rendered_tokens)
        + "</div>"
    )


def render_uncertainty_highlighted_answer(
    sentence_confidences: List[Tuple[str, float]],
    claim_attributions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render answer claims with confidence and source-grounding annotations.

    Confidence bands:
    - 0.80 to 1.00: normal text.
    - 0.50 to 0.79: yellow medium-confidence highlight.
    - Below 0.50: red dashed uncertain highlight and warning badge.
    """
    rendered_sentences: List[str] = []
    claim_attributions = claim_attributions or []

    for index, (sentence, raw_confidence) in enumerate(sentence_confidences):
        confidence = max(0.0, min(1.0, float(raw_confidence)))
        safe_sentence = html.escape(str(sentence))
        attribution = (
            claim_attributions[index]
            if index < len(claim_attributions)
            else {}
        )
        source = attribution.get("mapped_source")
        source_hint = (
            f' title="Grounded by {html.escape(str(source))}"'
            if source
            else ""
        )

        if confidence >= 0.80:
            rendered_sentences.append(
                f'<span{source_hint}>{safe_sentence}</span>'
            )
        elif confidence >= 0.50:
            rendered_sentences.append(
                f'<span{source_hint} style="background:rgba(254,240,138,0.6);'
                'border-radius:0.25rem;padding:0.08rem 0.18rem;'
                'color:#121110;">'
                f"{safe_sentence} "
                '<sup style="color:#854D0E;font-size:0.7em;font-weight:700;">'
                '[Medium Confidence]</sup></span>'
            )
        else:
            rendered_sentences.append(
                f'<span{source_hint} style="background:rgba(254,202,202,0.7);'
                'border:1px dashed #B91C1C;border-radius:0.25rem;'
                'padding:0.08rem 0.18rem;color:#121110;">'
                f"{safe_sentence} "
                '<sup style="color:#991B1B;font-size:0.7em;font-weight:700;">'
                '[Uncertain]</sup></span>'
            )

    return (
        '<div style="background:#FFFFFF;color:#121110;border:1px solid '
        'rgba(18,17,16,0.10);border-radius:0.75rem;padding:0.95rem 1rem;'
        'line-height:1.65;">'
        + " ".join(rendered_sentences)
        + "</div>"
    )


<<<<<<< HEAD
def render_claim_status_answer(claim_attributions: List[Dict[str, Any]]) -> str:
    """Render escaped answer claims with verifier-status, not score, highlights."""
    styles = {
        "SUPPORTED": "color:#166534;",
        "PARTIALLY_SUPPORTED": "background:rgba(254,240,138,0.6);color:#854D0E;",
        "UNSUPPORTED": "background:rgba(254,202,202,0.7);border:1px dashed #B91C1C;color:#991B1B;",
        "CONFLICTING": "background:rgba(254,202,202,0.7);border:1px dashed #7F1D1D;color:#7F1D1D;",
        "NEEDS_REVIEW": "background:rgba(254,240,138,0.6);border:1px dashed #854D0E;color:#854D0E;",
    }
    rendered = []
    for attribution in claim_attributions:
        status = str(attribution.get("status", "NEEDS_REVIEW")).upper()
        style = styles.get(status, styles["NEEDS_REVIEW"])
        claim = html.escape(str(attribution.get("claim") or ""))
        rendered.append(
            f'<span style="{style}border-radius:0.25rem;padding:0.08rem 0.18rem;">'
            f'<strong>[{html.escape(status)}]</strong> {claim}</span>'
        )
    return "<div style=\"line-height:1.7;\">" + " ".join(rendered) + "</div>"


=======
>>>>>>> ba4cfaf (Add explainability helpers)
def render_generation_self_critique_card(critique_data: Dict[str, Any]) -> str:
    """Render generation faithfulness, risk, abstention, and feedback metrics."""
    faithfulness = float(
        critique_data.get(
            "faithfulness_score",
            critique_data.get("grounding_score", 0.0),
        )
    )
    if faithfulness <= 1.0:
        faithfulness *= 100
    faithfulness = max(0.0, min(100.0, faithfulness))

    risk = str(critique_data.get("hallucination_risk", "MEDIUM")).upper()
    risk_colors = {
        "LOW": ("#166534", "rgba(187,247,208,0.75)"),
        "MEDIUM": ("#854D0E", "rgba(254,240,138,0.75)"),
        "HIGH": ("#991B1B", "rgba(254,202,202,0.75)"),
    }
    risk_text, risk_background = risk_colors.get(risk, risk_colors["MEDIUM"])
    abstention = str(
        critique_data.get("abstention_status", critique_data.get("abstention", "PASSED"))
    ).upper()
    abstention_valid = abstention in {"PASSED", "REJECTED"}
    abstention = abstention if abstention_valid else "REJECTED"
    abstention_color = "#166534" if abstention == "PASSED" else "#991B1B"
    abstention_background = (
        "rgba(187,247,208,0.75)" if abstention == "PASSED"
        else "rgba(254,202,202,0.75)"
    )
    feedback = html.escape(str(critique_data.get("feedback", "No critique feedback provided.")))

    return f"""
    <div style="background:#FFFFFF;color:#121110;border:1px solid rgba(18,17,16,0.10);
                border-radius:0.75rem;padding:0.95rem 1rem;margin:0.75rem 0;
                line-height:1.5;">
        <strong>Generation Self-Critique</strong>
        <div style="margin-top:0.65rem;display:flex;gap:0.75rem;flex-wrap:wrap;">
            <span>Faithfulness / Grounding: <strong>{faithfulness:.0f}%</strong></span>
            <span style="background:{risk_background};color:{risk_text};padding:0.15rem 0.45rem;
                         border-radius:999px;font-size:0.78rem;font-weight:700;">
                Hallucination Risk: {html.escape(risk)}
            </span>
            <span style="background:{abstention_background};color:{abstention_color};padding:0.15rem 0.45rem;
                         border-radius:999px;font-size:0.78rem;font-weight:700;">
                Abstention: {abstention}
            </span>
        </div>
        <div style="margin-top:0.65rem;color:#5A5148;">{feedback}</div>
    </div>
    """


def _chunk_page(chunk: Dict[str, Any]) -> Any:
    """Read page metadata from common flat and nested chunk schemas."""
    metadata = chunk.get("metadata") or {}
    return chunk.get("page") or metadata.get("page", "N/A")


def _chunk_source(chunk: Dict[str, Any]) -> str:
    """Read a displayable source name from common chunk schemas."""
    metadata = chunk.get("metadata") or {}
    return str(
        chunk.get("file_name")
        or chunk.get("document_name")
        or metadata.get("source_name")
        or "Unknown document"
    )


def render_explainability_ui(
    st_module: Any,
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
<<<<<<< HEAD
    answer_claims: Any,
    critique_data: Optional[Dict[str, Any]] = None,
    retrieval_metadata: Optional[Any] = None,
    claim_verifier: Optional[Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]] = None,
    contradiction_verifier: Optional[Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]] = None,
    citation_report: Optional[Dict[str, Any]] = None,
    claim_attributions: Optional[List[Dict[str, Any]]] = None,
    evidence_report: Optional[Dict[str, Any]] = None,
    formatted_answer: Optional[str] = None,
    show_answer: bool = True,
=======
    answer_claims: List[str],
    critique_data: Optional[Dict[str, Any]] = None,
>>>>>>> ba4cfaf (Add explainability helpers)
) -> None:
    """Render claim grounding, uncertainty answer, critique, and chunk heatmaps.

    Parameters
    ----------
    st_module:
        The imported Streamlit module, normally passed as ``st``.
    query:
        The user query used to compute token attribution.
    retrieved_chunks:
        Retrieved chunk dictionaries containing at least ``text``.
    answer_claims:
        Individual answer sentences or claims.
    critique_data:
        Optional generation self-critique metrics for the health card.
<<<<<<< HEAD
    claim_attributions, evidence_report:
        Optional precomputed results for avoiding repeated verifier calls during reruns.
    """
    chunks = list(retrieved_chunks or [])
    claims = split_answer_claims(answer_claims)
    attributions = (
        list(claim_attributions)
        if claim_attributions is not None
        else compute_claim_attribution(claims, chunks, verifier=claim_verifier)
    )
    report = evidence_report or generate_evidence_report(
        claims,
        chunks,
        claim_attributions=attributions,
        retrieval_metadata=retrieval_metadata,
        contradiction_verifier=contradiction_verifier,
        citation_report=citation_report,
    )

    if show_answer:
        st_module.markdown("**Answer**")
        st_module.markdown(formatted_answer or format_legal_answer(answer_claims))
    st_module.markdown("### 1. Answer Claim Verification")
    if attributions:
        st_module.dataframe(
            [
                {
                    "Claim": item["claim"],
                    "Status": item["status"],
                    "Valid Citations": ", ".join(item["valid_citations"]) or "N/A",
                }
                for item in attributions
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st_module.info("No answer claims were supplied.")

    st_module.markdown("### 2. Supporting Citations and Evidence")
    for index, attribution in enumerate(attributions, start=1):
        evidence_items = attribution["supporting_evidence"] or attribution["candidate_evidence"]
        with st_module.expander(
            f"{index}. {attribution['status']} · {attribution['claim']}",
            expanded=attribution["status"] in {"UNSUPPORTED", "CONFLICTING", "NEEDS_REVIEW"},
        ):
            st_module.write(attribution["verification_note"])
            if attribution["valid_citations"]:
                st_module.write("Valid cited chunk IDs:", ", ".join(attribution["valid_citations"]))
            if attribution["invalid_citations"]:
                st_module.warning(
                    "Citation IDs not found in retrieved evidence: "
                    + ", ".join(attribution["invalid_citations"])
                )
            if not evidence_items:
                st_module.write("No retrieved passage met the candidate threshold.")
            for evidence in evidence_items:
                page_label = (
                    f"Page {evidence['page_number']} ({evidence['page_provenance']})"
                    if evidence["page_number"] is not None
                    else "Page N/A"
                )
                st_module.write(
                    f"{evidence['document_name']} · {page_label} · "
                    f"Chunk {evidence['chunk_id'] or 'N/A'}"
                )
                st_module.write(
                    "Evidence support score:",
                    evidence["evidence_support_score"]
                    if evidence["evidence_support_score"] is not None
                    else "N/A (verifier did not provide one)",
                )
                st_module.write("Evidence verification status:", evidence["verification_status"])
                st_module.text(evidence["evidence_excerpt"] or "No source text available.")

    st_module.markdown("### 3. Hybrid Retrieval Trace")
    render_retrieval_trace_ui(st_module, chunks, retrieval_metadata)

    st_module.markdown("### 4. Evidence Coverage and Verification Report")
    coverage = report["evidence_coverage"]
    total_claims = coverage["total_claims"]
    supported_percentage = (
        coverage["supported_claims"] / total_claims * 100.0
        if total_claims
        else 0.0
    )
    metric_rows = (
        (
            ("Total claims", total_claims),
            ("Supported", coverage["supported_claims"]),
            ("Supported claims", f"{supported_percentage:.1f}%"),
        ),
        (
            ("Needs review", coverage["needs_review_claims"]),
            ("Unsupported", coverage["unsupported_claims"]),
            ("Conflicting", coverage["conflicting_claims"]),
        ),
        (
            ("Partially supported", coverage["partially_supported_claims"]),
            ("Evidence coverage", f"{coverage['sufficient_evidence_percentage']:.1f}%"),
            ("Citation coverage", f"{coverage['citation_coverage_percentage']:.1f}%"),
        ),
    )
    for metric_row in metric_rows:
        metric_columns = st_module.columns(3)
        for column, (label, value) in zip(metric_columns, metric_row):
            with column:
                st_module.metric(label, value)
    st_module.caption(
        "Retrieval scores rank passage relevance; citation coverage measures valid "
        "chunk references; neither alone establishes claim-level legal support."
    )
    st_module.write(
        "Partially supported:", coverage["partially_supported_claims"],
        "· Unsupported:", coverage["unsupported_claims"],
        "· Needs review:", coverage["needs_review_claims"],
        "· Conflicting:", coverage["conflicting_claims"],
    )
    st_module.caption(coverage["sufficient_evidence_definition"])
    st_module.caption(coverage["citation_coverage_definition"])
    st_module.write("Overall evidence status:", report["overall_status"])
    st_module.write(
        "Claims without valid citations:",
        len(report["claims_without_valid_citations"]),
    )
    for claim in report["claims_without_valid_citations"]:
        st_module.warning("Claim without a valid citation: " + claim)
    st_module.write(
        "Claims not fully verified:",
        coverage["needs_review_claims"]
        + coverage["unsupported_claims"]
        + coverage["conflicting_claims"],
    )
    for action in report["suggested_next_actions"]:
        st_module.write("Suggested next action:", action)

    st_module.markdown("### 5. Contradiction and Unsupported-Claim Warnings")
    if report["contradiction_status"] == "NOT_EVALUATED":
        if report.get("contradiction_verifier_failures"):
            st_module.warning(
                "Contradiction detection was unavailable because the configured "
                "verifier failed for every claim."
            )
        else:
            st_module.info("Contradiction detection was not evaluated; no verifier is configured.")
    elif report["contradiction_status"] == "NO_CONTRADICTION_FOUND":
        st_module.write("The configured verifier found no contradiction in the retrieved passages.")
    else:
        st_module.warning("Contradiction check status: " + report["contradiction_status"])
    for item in report["contradictory_evidence"]:
        st_module.warning(
            f"{item['claim']} · {item['document_name']} · "
            f"Chunk {item['chunk_id']} · {item['page_provenance']}"
        )
        st_module.text(item["evidence_excerpt"])
        st_module.caption(
            "Differences in dates, jurisdiction, amendments, exceptions, or scope "
            "require human review; no provision precedence is inferred."
        )
    for item in report["contradiction_reviews"]:
        st_module.warning(f"Needs human review: {item['claim']} · {item['rationale']}")
        for evidence in item["evidence"]:
            st_module.write(
                f"{evidence['document_name']} · Chunk {evidence['chunk_id']} · "
                f"Page {evidence['page_number'] or 'N/A'} ({evidence['page_provenance']})"
            )
            st_module.text(evidence["evidence_excerpt"])
    for claim in report["claims_without_supporting_evidence"]:
        st_module.warning("Unsupported claim: " + claim)
    for citation in report["invalid_citations"]:
        st_module.warning(
            "Invalid citation for " + citation["claim"] + ": "
            + ", ".join(citation["chunk_ids"])
        )
    if citation_report:
        for uncited in citation_report.get("uncited_lines", []) or []:
            st_module.warning("Claim line lacks an inline citation: " + str(uncited))

    st_module.markdown("### 6. Token Relevance Heatmap")
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        source = html.escape(_chunk_source(chunk))
        page_reference = extract_page_reference(chunk)
        page_label = (
            f"Page {page_reference['page_number']} ({page_reference['provenance']})"
            if page_reference["page_number"] is not None
            else "Page N/A"
        )
=======
    """
    attributions = compute_claim_attribution(answer_claims, retrieved_chunks)
    sentence_confidences = [
        (
            claim,
            float(
                attribution.get("attribution_score", 0.0)
                if attribution
                else 0.0
            ),
        )
        for claim, attribution in zip(answer_claims, attributions)
    ]

    st_module.markdown("### Answer")
    st_module.markdown(
        render_uncertainty_highlighted_answer(sentence_confidences, attributions),
        unsafe_allow_html=True,
    )

    if critique_data:
        st_module.markdown(
            render_generation_self_critique_card(critique_data),
            unsafe_allow_html=True,
        )

    if not retrieved_chunks:
        return

    page_stats = get_document_page_stats(retrieved_chunks)
    summary_parts = []
    for source, pages in page_stats.items():
        safe_source = html.escape(source)
        page_label = ", ".join(str(page) for page in pages) if pages else "N/A"
        summary_parts.append(f"📄 {safe_source} (Pages {html.escape(page_label)})")
    if summary_parts:
        st_module.markdown(
            '<div style="background:#FFFFFF;color:#121110;border:1px solid '
            'rgba(18,17,16,0.10);border-radius:0.75rem;padding:0.75rem 1rem;'
            'margin:0.75rem 0;">'
            + " | ".join(summary_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

    st_module.markdown("### Explainable Retrieval Heatmap")
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        source = html.escape(_chunk_source(chunk))
        _, page_label = extract_and_map_page_number(chunk)
>>>>>>> ba4cfaf (Add explainability helpers)
        page = html.escape(page_label)
        title = f"#{rank} • {source} • {page}"
        text = str(chunk.get("text") or "")

        with st_module.expander(title, expanded=False):
            st_module.markdown(
                f"<div style=\"background:#F4F1ED;border:1px solid rgba(18,17,16,0.10);"
                f"border-radius:999px;padding:0.35rem 0.65rem;display:inline-block;"
                f"color:#121110;\"><strong>{source}</strong> • {page}</div>",
                unsafe_allow_html=True,
            )
            st_module.markdown(
                render_explainable_chunk_heatmap(
                    compute_token_relevance(query, text)
                ),
                unsafe_allow_html=True,
            )
            st_module.markdown("**Raw context:**")
<<<<<<< HEAD
            st_module.text(text)

    st_module.markdown("### 7. Generation Self-Critique")
    if critique_data:
        st_module.markdown(
            render_generation_self_critique_card(critique_data),
            unsafe_allow_html=True,
        )
    else:
        st_module.info("No generation self-critique was supplied.")
=======
            st_module.markdown(text)
>>>>>>> ba4cfaf (Add explainability helpers)


# Integration example for streamlit_app_v2.py:
#
# from explainability import render_explainability_ui
#
# render_explainability_ui(
#     st_module=st,
#     query=submitted_query,
#     retrieved_chunks=query_result.get("hits", []),
#     answer_claims=["The answer claim goes here."],
#     critique_data={
#         "faithfulness_score": 0.92,
#         "hallucination_risk": "LOW",
#         "abstention_status": "PASSED",
#         "feedback": "All answer claims map to retrieved evidence.",
#     },
# )
