"""Explainability helpers for the Advanced RAG application.

This module keeps attribution and uncertainty presentation independent from the
retrieval and generation implementations. It does not import Streamlit; the
UI wrapper accepts a Streamlit-like module object so the functions remain easy
to test and reuse.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
_PAGE_PATTERNS = (
    re.compile(r"\bpage\s*[:#-]?\s*(\d+)\s*(?:of\s*\d+)?\b", re.IGNORECASE),
    re.compile(r"\bp\.\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"[-–—]\s*(\d+)\s*[-–—]"),
)


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
        page_number, _ = extract_and_map_page_number(chunk)
        document_pages.setdefault(source, set())
        if page_number > 0:
            document_pages[source].add(page_number)

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
    answer_claims: List[str],
    critique_data: Optional[Dict[str, Any]] = None,
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
            st_module.markdown(text)


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
