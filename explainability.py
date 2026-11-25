"""Explainability helpers for the Advanced RAG application.

This module keeps attribution and uncertainty presentation independent from the
retrieval and generation implementations. It does not import Streamlit; the
UI wrapper accepts a Streamlit-like module object so the functions remain easy
to test and reuse.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Tuple


_TOKEN_PATTERN = re.compile(r"\S+")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
_CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+)\]")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
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

    The resolver checks page metadata and the extraction pipeline's explicit
    ``[PAGE n]`` markers. It does not infer a page from chunk length or prose;
    ``0`` is used when the source page is unavailable.
    """
    metadata = chunk.get("metadata") or {}
    source_pages = chunk.get("page_numbers") or metadata.get("page_numbers")
    if source_pages is not None:
        page_values = (
            source_pages
            if isinstance(source_pages, (list, tuple, set))
            else re.findall(r"\d+", str(source_pages))
        )
        page_numbers = sorted(
            {
                int(str(page).strip())
                for page in page_values
                if str(page).strip().isdigit() and int(str(page).strip()) > 0
            }
        )
        if page_numbers:
            page_label = (
                f"Page {page_numbers[0]}"
                if len(page_numbers) == 1
                else f"Pages {', '.join(str(page) for page in page_numbers)}"
            )
            return page_numbers[0], page_label

    page_candidates = (
        chunk.get("page"),
        chunk.get("page_number"),
        chunk.get("page_label"),
        metadata.get("page"),
        metadata.get("page_number"),
        metadata.get("page_label"),
        metadata.get("page_numbers"),
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
    marker_pages = sorted(
        {
            int(page_number)
            for page_number in re.findall(
                r"\[PAGE\s+(\d+)\]", text, re.IGNORECASE
            )
        }
    )
    if marker_pages:
        page_label = (
            f"Page {marker_pages[0]}"
            if len(marker_pages) == 1
            else f"Pages {', '.join(str(page) for page in marker_pages)}"
        )
        return marker_pages[0], page_label

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


def _split_answer_claims(answer: str) -> List[str]:
    """Split an answer into citation-bearing prose sentences and bullet claims."""
    claims: List[str] = []
    for paragraph in re.split(r"\n\s*\n", answer or ""):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if any(_BULLET_PATTERN.match(line) for line in lines):
            current: List[str] = []
            for line in lines:
                if _BULLET_PATTERN.match(line) and current:
                    claims.append(" ".join(current))
                    current = []
                current.append(line)
            if current:
                claims.append(" ".join(current))
            continue

        paragraph_text = " ".join(lines)
        claims.extend(
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+(?=[A-Z0-9*`])", paragraph_text
            )
            if sentence.strip()
        )
    return claims


def _normalized_chunk_id(chunk_id: Any) -> str:
    normalized = str(chunk_id).strip().lower()
    match = re.match(r"^(.*?)(\d+)$", normalized)
    if not match:
        return normalized
    return f"{match.group(1)}{int(match.group(2))}"


def _chunk_suffix(chunk_id: Any) -> Optional[str]:
    match = re.search(r"(?:^|_)chunk_(\d+)$", str(chunk_id).strip().lower())
    return f"chunk_{int(match.group(1))}" if match else None


def build_explainability_data(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    citation_report: Optional[Dict[str, Any]] = None,
    retrieval_rounds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble explainability from the pipeline's citations and retrieved hits.

    Citation validation verifies that references resolve to retrieved chunks;
    lexical attribution is a heuristic and does not prove factual entailment.
    """
    claims = _split_answer_claims(answer)
    attributions = compute_claim_attribution(claims, retrieved_chunks)
    chunks_by_normalized_id = {
        _normalized_chunk_id(chunk.get("chunk_id")): str(chunk.get("chunk_id"))
        for chunk in retrieved_chunks
        if chunk.get("chunk_id") is not None
    }
    suffix_ids: Dict[str, set[str]] = {}
    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id is None:
            continue
        suffix = _chunk_suffix(chunk_id)
        if suffix:
            suffix_ids.setdefault(suffix, set()).add(str(chunk_id))
    aliases = {
        suffix: next(iter(chunk_ids))
        for suffix, chunk_ids in suffix_ids.items()
        if len(chunk_ids) == 1
    }
    validated_ids = {
        str(chunk_id)
        for chunk_id in (citation_report or {}).get("citations", [])
    }

    claim_reports: List[Dict[str, Any]] = []
    for claim, attribution in zip(claims, attributions):
        raw_citations = _CITATION_PATTERN.findall(claim)
        cited_chunk_ids: List[str] = []
        valid_cited_ids: List[str] = []
        unresolved_citations: List[str] = []
        for citation in raw_citations:
            normalized = _normalized_chunk_id(citation)
            resolved_id = chunks_by_normalized_id.get(normalized)
            if resolved_id is None:
                resolved_id = aliases.get(normalized)
            if resolved_id and resolved_id not in cited_chunk_ids:
                cited_chunk_ids.append(resolved_id)
                if resolved_id in validated_ids:
                    valid_cited_ids.append(resolved_id)
            elif resolved_id is None and citation not in unresolved_citations:
                unresolved_citations.append(citation)

        if not retrieved_chunks:
            verification_status = "insufficient_evidence"
        elif citation_report is None:
            verification_status = "unavailable"
        elif (
            valid_cited_ids
            and not unresolved_citations
            and len(valid_cited_ids) == len(cited_chunk_ids)
        ):
            verification_status = "citation_reference_verified"
        elif valid_cited_ids:
            verification_status = "partially_verified"
        elif cited_chunk_ids or raw_citations:
            verification_status = "invalid_citation"
        else:
            verification_status = "missing_citation"

        claim_reports.append({
            **attribution,
            "claim": claim,
            "cited_chunk_ids": cited_chunk_ids,
            "validated_citation_ids": valid_cited_ids,
            "unresolved_citations": unresolved_citations,
            "verification_status": verification_status,
        })

    verified_claim_count = sum(
        claim["verification_status"] == "citation_reference_verified"
        for claim in claim_reports
    )
    coverage = (
        verified_claim_count / len(claim_reports)
        if claim_reports and citation_report is not None
        else None
    )
    assessed_sufficiency = [
        round_data["evidence_sufficient"]
        for round_data in (retrieval_rounds or [])
        if isinstance(round_data.get("evidence_sufficient"), bool)
    ]
    evidence_report = {
        "status": (
            "no_retrieved_evidence"
            if not retrieved_chunks
            else "citation_coverage_complete"
            if claim_reports and verified_claim_count == len(claim_reports)
            else "citation_coverage_incomplete"
        ),
        "claim_count": len(claim_reports),
        "verified_claim_count": verified_claim_count,
        "citation_coverage": coverage,
        "claims": claim_reports,
        "unverified_claims": [
            claim["claim"]
            for claim in claim_reports
            if citation_report is not None
            and claim["verification_status"] != "citation_reference_verified"
        ],
        "citation_validation_available": citation_report is not None,
        "citation_report": citation_report,
        "retrieval_sufficiency": (
            assessed_sufficiency[-1] if assessed_sufficiency else None
        ),
        "contradiction_analysis": {
            "status": "not_available",
            "reason": (
                "The current retrieval and generation pipeline has no "
                "contradiction-analysis component."
            ),
        },
    }
    return {
        "claims": claims,
        "claim_attributions": claim_reports,
        "citation_report": citation_report,
        "evidence_report": evidence_report,
    }


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
        alpha = 0.08 + (0.32 * score)
        safe_word = html.escape(str(word))
        rendered_tokens.append(
            f'<span style="background-color:rgba(245,158,11,{alpha:.2f});'
            'border-radius:0.2rem;padding:0.05rem 0.12rem;'
            f'color:var(--legal-text,#121110);">{safe_word}</span>'
        )

    return (
        '<div style="background:var(--legal-surface,#FFFFFF);'
        'color:var(--legal-text,#121110);border:1px solid '
        'var(--legal-border,rgba(18,17,16,0.10));border-radius:0.75rem;'
        'padding:0.85rem 1rem;'
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
                f'<span{source_hint} style="background:rgba(254,240,138,0.2);'
                'border-radius:0.25rem;padding:0.08rem 0.18rem;'
                'color:var(--legal-text,#121110);">'
                f"{safe_sentence} "
                '<sup style="color:var(--legal-text,#854D0E);'
                'font-size:0.7em;font-weight:700;">'
                '[Medium Confidence]</sup></span>'
            )
        else:
            rendered_sentences.append(
                f'<span{source_hint} style="background:rgba(254,202,202,0.2);'
                'border:1px dashed #B91C1C;border-radius:0.25rem;'
                'padding:0.08rem 0.18rem;color:var(--legal-text,#121110);">'
                f"{safe_sentence} "
                '<sup style="color:var(--legal-text,#991B1B);'
                'font-size:0.7em;font-weight:700;">'
                '[Uncertain]</sup></span>'
            )

    return (
        '<div style="background:var(--legal-surface,#FFFFFF);'
        'color:var(--legal-text,#121110);border:1px solid '
        'var(--legal-border,rgba(18,17,16,0.10));border-radius:0.75rem;'
        'padding:0.95rem 1rem;'
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
        "LOW": ("var(--legal-text,#166534)", "rgba(187,247,208,0.2)"),
        "MEDIUM": ("var(--legal-text,#854D0E)", "rgba(254,240,138,0.2)"),
        "HIGH": ("var(--legal-text,#991B1B)", "rgba(254,202,202,0.2)"),
    }
    risk_text, risk_background = risk_colors.get(risk, risk_colors["MEDIUM"])
    abstention = str(
        critique_data.get("abstention_status", critique_data.get("abstention", "PASSED"))
    ).upper()
    abstention_valid = abstention in {"PASSED", "REJECTED"}
    abstention = abstention if abstention_valid else "REJECTED"
    abstention_color = "var(--legal-text,#166534)"
    abstention_background = (
        "rgba(187,247,208,0.2)" if abstention == "PASSED"
        else "rgba(254,202,202,0.2)"
    )
    feedback = html.escape(str(critique_data.get("feedback", "No critique feedback provided.")))

    return f"""
    <div style="background:var(--legal-surface,#FFFFFF);
                color:var(--legal-text,#121110);border:1px solid
                var(--legal-border,rgba(18,17,16,0.10));
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
        <div style="margin-top:0.65rem;color:var(--legal-text,#121110);">{feedback}</div>
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
    retrieval_metadata: Optional[List[Dict[str, Any]]] = None,
    citation_report: Optional[Dict[str, Any]] = None,
    claim_attributions: Optional[List[Dict[str, Any]]] = None,
    evidence_report: Optional[Dict[str, Any]] = None,
    show_answer: bool = True,
    retrieval_rounds: Optional[List[Dict[str, Any]]] = None,
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
    attributions = claim_attributions or compute_claim_attribution(
        answer_claims, retrieved_chunks
    )
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

    if show_answer:
        st_module.markdown("### Answer")
        st_module.markdown(
            render_uncertainty_highlighted_answer(
                sentence_confidences, attributions
            ),
            unsafe_allow_html=True,
        )

    if critique_data:
        st_module.markdown(
            render_generation_self_critique_card(critique_data),
            unsafe_allow_html=True,
        )

    with st_module.expander("Citation validation", expanded=False):
        if citation_report is None:
            st_module.info("Citation validation was not returned for this answer.")
        elif citation_report.get("is_valid"):
            st_module.success("Citation validation passed.")
        else:
            st_module.warning("Citation validation found issues.")
        if citation_report:
            citations = citation_report.get("citations", [])
            if citations:
                st_module.caption(
                    "Validated citation IDs: "
                    + ", ".join(f"`{chunk_id}`" for chunk_id in citations)
                )
            for problem in citation_report.get("problems", []):
                st_module.warning(str(problem))

    if evidence_report is not None:
        st_module.markdown("### Evidence Verification")
        claim_count = evidence_report.get("claim_count", 0)
        verified_count = evidence_report.get("verified_claim_count", 0)
        coverage = evidence_report.get("citation_coverage")
        if not evidence_report.get("citation_validation_available"):
            st_module.info(
                "Citation verification was not available for this answer."
            )
        elif not retrieved_chunks:
            st_module.warning(
                "Insufficient evidence: no retrieved document chunks are available."
            )
        elif claim_count and verified_count == claim_count:
            st_module.success(
                f"Citation references resolved for all {claim_count} claim(s). "
                "This checks references, not factual entailment."
            )
        elif claim_count:
            st_module.warning(
                f"Citation references resolved for {verified_count} of "
                f"{claim_count} claim(s). Unverified claims are listed below."
            )
        else:
            st_module.info("No answer claims were available to verify.")

        if coverage is not None:
            st_module.caption(
                f"Verified citation coverage: {verified_count}/{claim_count} "
                f"({coverage:.0%})"
            )
        retrieval_sufficiency = evidence_report.get("retrieval_sufficiency")
        if retrieval_sufficiency is None:
            st_module.caption(
                "Retrieval sufficiency was not assessed by the current query path."
            )
        else:
            st_module.caption(
                "Retrieval sufficiency assessment: "
                + ("sufficient" if retrieval_sufficiency else "insufficient")
            )

        for claim in evidence_report.get("claims", []):
            cited_ids = claim.get("validated_citation_ids", [])
            chunks_by_id = {
                str(chunk.get("chunk_id")): chunk for chunk in retrieved_chunks
            }
            reference_details = []
            for chunk_id in cited_ids:
                cited_chunk = chunks_by_id.get(str(chunk_id))
                if cited_chunk:
                    _, page_label = extract_and_map_page_number(cited_chunk)
                    reference_details.append(
                        f"`{chunk_id}` — {_chunk_source(cited_chunk)}, {page_label}"
                    )
                else:
                    reference_details.append(f"`{chunk_id}`")
            reference_label = ", ".join(reference_details) or (
                "Citation validation unavailable"
                if not evidence_report.get("citation_validation_available")
                else "No validated retrieved citation"
            )
            st_module.markdown(
                f"- **{claim.get('verification_status', 'unverified').replace('_', ' ').title()}**: "
                f"{claim.get('claim', '')}  \n"
                f"  Evidence references: {reference_label}"
            )
            attribution_score = claim.get("attribution_score")
            if attribution_score is not None:
                st_module.caption(
                    f"Best lexical match: {claim.get('mapped_source', 'Unknown source')} "
                    f"({float(attribution_score):.0%} heuristic overlap; not semantic verification)"
                )
            if claim.get("unresolved_citations"):
                st_module.caption(
                    "Unresolved citation(s): "
                    + ", ".join(f"`{item}`" for item in claim["unresolved_citations"])
                )

        if evidence_report.get("unverified_claims"):
            st_module.warning(
                "Claims without a validated retrieved citation are not "
                "verified against the documents."
            )

        contradiction = evidence_report.get("contradiction_analysis") or {}
        if contradiction.get("status") != "completed":
            st_module.info(
                "Contradiction analysis was not performed: "
                + str(contradiction.get("reason", "no analysis result is available"))
            )


    if retrieval_rounds:
        st_module.markdown("### Retrieval Verification Passes")
        for round_data in retrieval_rounds:
            round_number = round_data.get("round", "?")
            hit_count = round_data.get("hit_count", 0)
            sufficiency = round_data.get("evidence_sufficient")
            query_label = round_data.get("query") or "search"
            with st_module.expander(
                f"Pass {round_number} · {hit_count} chunks · {query_label}",
                expanded=False,
            ):
                if sufficiency is True:
                    st_module.success("Evidence sufficiency check passed.")
                elif sufficiency is False:
                    st_module.warning("Evidence sufficiency check requested another pass.")
                else:
                    st_module.info("No evidence sufficiency result was recorded for this pass.")
                if round_data.get("query"):
                    st_module.caption(f"Search query: {round_data['query']}")
                if round_data.get("evidence_check_error"):
                    st_module.warning(
                        f"Evidence check error: {round_data['evidence_check_error']}"
                    )

    if not retrieved_chunks:
        return

    st_module.markdown("### Retrieval Trace")
    score_names = (
        ("Dense", "dense_score"),
        ("BM25", "bm25_score"),
        ("RRF", "rrf_score"),
        ("Cross-Encoder", "rerank_score"),
    )
    trace_chunks = retrieval_metadata or retrieved_chunks
    with st_module.expander("Retrieval scores and source pages", expanded=False):
        for rank, chunk in enumerate(trace_chunks, start=1):
            scores = []
            for label, key in score_names:
                value = chunk.get(key)
                try:
                    formatted_value = f"{float(value):.3f}" if value is not None else "n/a"
                except (TypeError, ValueError):
                    formatted_value = "n/a"
                scores.append(f"{label}: {formatted_value}")
            _, page_label = extract_and_map_page_number(chunk)
            st_module.caption(
                f"#{rank} `{chunk.get('chunk_id', 'unknown')}` — "
                f"{_chunk_source(chunk)}, {page_label} — "
                + " | ".join(scores)
            )

    page_stats = get_document_page_stats(retrieved_chunks)
    summary_parts = []
    for source, pages in page_stats.items():
        safe_source = html.escape(source)
        page_label = ", ".join(str(page) for page in pages) if pages else "N/A"
        summary_parts.append(f"📄 {safe_source} (Pages {html.escape(page_label)})")
    if summary_parts:
        st_module.markdown(
            '<div style="background:var(--legal-surface,#FFFFFF);'
            'color:var(--legal-text,#121110);border:1px solid '
            'var(--legal-border,rgba(18,17,16,0.10));'
            'border-radius:0.75rem;padding:0.75rem 1rem;'
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
                f"<div style=\"background:var(--legal-surface,#F4F1ED);"
                f"border:1px solid var(--legal-border,rgba(18,17,16,0.10));"
                f"border-radius:999px;"
                f"padding:0.35rem 0.65rem;display:inline-block;"
                f"color:var(--legal-text,#121110);\"><strong>{source}</strong> • {page}</div>",
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
