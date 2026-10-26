"""Build claim-level explainability data from retrieval and citation results."""

import re
from typing import Any, Dict, List, Optional

from explainability import compute_claim_attribution


_CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+)\]")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")


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

        claims.extend(
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+(?=[A-Z0-9*`])", " ".join(lines)
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
    """Assemble claim attribution and evidence status from existing pipeline data.

    Citation validation verifies references resolve to retrieved chunks. The
    lexical attribution score is a heuristic, not factual entailment.
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
