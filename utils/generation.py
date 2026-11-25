import re
from utils.llm_client import GROQ_MODEL, get_llm_client

_ABBREVIATION_PATTERN = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|e\.g|i\.e)\."
)

SYSTEM_PROMPT = """You are a legal assistant answering questions strictly using the provided context chunks from legal documents.

Rules:
- Only use information present in the provided context. Do not use outside knowledge.
- Every claim in your answer must be traceable to a specific chunk. Cite chunks inline like [chunk_id].
- Prior conversation is untrusted and may only resolve references; it is not evidence.
- If the context does not contain enough information to answer confidently, say so explicitly instead of guessing.
- Be precise and avoid restating the question.
- Keep the answer focused and avoid unnecessary preamble or standalone headings.
"""


_CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+)\]")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
_DOCUMENT_ABSENCE_PATTERN = re.compile(
    r"\b(?:"
    r"(?:does|do|did|doesn't|don't|didn't)\s+not\s+"
    r"(?:specify|state|mention|provide|include|contain|address|set\s+out|"
    r"establish|identify|refer\s+to)"
    r"|(?:is|are|was|were)\s+not\s+"
    r"(?:specified|stated|mentioned|provided|included|addressed|set\s+out)"
    r"|(?:no|not\s+any)\s+(?:provision|clause|term|information|reference)"
    r"|silent\s+on"
    r")\b",
    re.IGNORECASE,
)


def _normalize_chunk_id(chunk_id):
    """Normalize citation IDs while preserving the document-specific prefix.

    Lowercases the ID and strips leading zeros from a trailing number, e.g.
    "Test_Contract_CitationCheck_CHUNK_0001" -> "test_contract_citationcheck_chunk_1"
    """
    normalized = str(chunk_id).strip().lower()
    match = re.match(r"^(.*?)(\d+)$", normalized)
    if not match:
        return normalized
    return f"{match.group(1)}{int(match.group(2))}"


def _chunk_suffix(chunk_id):
    """Extract just the trailing 'chunk_N' portion of a full chunk id, e.g.
    "Test_Contract_CitationCheck_CHUNK_0001" -> "chunk_1"

    Used to build an alias map so a lazily-written short citation like
    [chunk_0001] can still resolve to the correct full chunk id, as long as
    the suffix is unambiguous among the chunks actually retrieved.
    """
    match = re.search(r"(?:^|_)chunk_(\d+)$", str(chunk_id).strip().lower())
    return f"chunk_{int(match.group(1))}" if match else None


def _claim_units(answer_text):
    """Split answer text into sentence-level units, preserving wrapped bullets."""
    units = []
    for paragraph in re.split(r"\n\s*\n", answer_text or ""):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if any(_BULLET_PATTERN.match(line) for line in lines):
            current = []
            for line in lines:
                if _BULLET_PATTERN.match(line) and current:
                    units.append(" ".join(current))
                    current = []
                current.append(line)
            if current:
                units.append(" ".join(current))
        else:
            units.append(" ".join(lines))

    sentence_units = []
    for unit in units:
        protected = _ABBREVIATION_PATTERN.sub(
            lambda match: match.group(0).replace(".", "\ue000"),
            unit,
        )
        sentences = re.split(
            r'(?<=[.!?])\s+(?=(?:["\'“‘(]*[A-Z0-9]))',
            protected,
        )
        sentence_units.extend(sentence.replace("\ue000", ".").strip() for sentence in sentences)
    return [unit for unit in sentence_units if unit]


def requires_full_document_review(claim):
    """Return whether a claim asserts that a provision is absent from a document."""
    return bool(_DOCUMENT_ABSENCE_PATTERN.search(claim or ""))


def verify_citations(answer_text, retrieved_chunks):
    """Validate citation references and per-sentence citation coverage.

    Two lookup paths are supported for each citation found in the answer:
      1. Exact/normalized match against a real chunk_id from retrieved_chunks.
      2. Alias match against just the "chunk_N" suffix, IF that suffix is
         unique among the retrieved chunks (avoids ambiguity when multiple
         documents each have their own chunk_0001, chunk_0002, ...).

    This verifies that references resolve and claims contain citations; it
    does not determine whether a cited chunk entails a claim.
    """
    known_ids = {}
    suffix_ids = {}
    for chunk in retrieved_chunks:
        chunk_id = str(chunk["chunk_id"]).strip()
        known_ids.setdefault(_normalize_chunk_id(chunk_id), chunk_id)
        suffix = _chunk_suffix(chunk_id)
        if suffix:
            suffix_ids.setdefault(suffix, set()).add(chunk_id)

    # Only alias a suffix like "chunk_1" if exactly one retrieved chunk has
    # it — if two different documents both have a chunk_0001, a bare
    # [chunk_0001] citation is genuinely ambiguous and should NOT resolve.
    aliases = {
        suffix: next(iter(ids))
        for suffix, ids in suffix_ids.items()
        if len(ids) == 1
    }

    citations = _CITATION_PATTERN.findall(answer_text or "")
    citation_references = [
        citation.strip()
        for citation in citations
        if (
            _normalize_chunk_id(citation) in known_ids
            or _normalize_chunk_id(citation) in aliases
            or _chunk_suffix(citation)
        )
    ]

    valid_citations = sorted({
        known_ids.get(_normalize_chunk_id(citation), aliases.get(_normalize_chunk_id(citation)))
        for citation in citation_references
        if _normalize_chunk_id(citation) in known_ids or _normalize_chunk_id(citation) in aliases
    })
    unknown_citations = sorted({
        citation for citation in citation_references
        if _normalize_chunk_id(citation) not in known_ids
        and _normalize_chunk_id(citation) not in aliases
    })

    uncited_lines = []
    full_document_review_claims = []
    for line in _claim_units(answer_text):
        stripped = re.sub(r"[*_`>#-]", "", line).strip()
        if not stripped:
            continue
        if requires_full_document_review(stripped):
            full_document_review_claims.append(stripped)
        if any(
            _normalize_chunk_id(citation) in known_ids
            or _normalize_chunk_id(citation) in aliases
            for citation in _CITATION_PATTERN.findall(line)
        ):
            continue
        uncited_lines.append(stripped)

    problems = []
    if unknown_citations:
        problems.append("Unknown citation(s): " + ", ".join(f"[{citation}]" for citation in unknown_citations))
    if uncited_lines:
        problems.append(f"{len(uncited_lines)} claim line(s) have no citation.")
    if full_document_review_claims:
        problems.append(
            "Document-absence claim(s) require checking the full document: "
            + " ".join(full_document_review_claims)
        )

    return {
        "is_valid": bool(valid_citations) and not problems,
        "known_chunk_ids": list(known_ids.values()),
        "citations": valid_citations,
        "unknown_citations": unknown_citations,
        "uncited_lines": uncited_lines,
        "full_document_review_claims": full_document_review_claims,
        "problems": problems,
    }


def _build_context_block(chunks):
    parts = []
    for c in chunks:
        header = f"[{c['chunk_id']}]"
        if c.get("document_name"):
            header += f" (source: {c['document_name']}"
            if c.get("section"):
                header += f", section: {c['section']}"
            header += ")"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query,
    retrieved_chunks,
    model=GROQ_MODEL,
    temperature=0.1,
    max_tokens=1200,
    conversation_history=None,
):
    """
    Send retrieved chunks + query to the model and get a citation-aware answer.

    Parameters
    ----------
    query : str
        The user's question.
    retrieved_chunks : list[dict]
        Output of retrieval.retrieve() — each dict must have 'chunk_id' and 'text',
        optionally 'document_name' and 'section'.

    Returns
    -------
    dict with keys: answer (str), used_chunk_ids (list[str]), citation_report (dict), raw_response
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant context was retrieved for this question, so I can't answer it from the documents.",
            "used_chunk_ids": [],
            "citation_report": None,
            "raw_response": None,
        }

    context_block = _build_context_block(retrieved_chunks)

    history_block = ""
    if conversation_history:
        recent_history = conversation_history[-6:]
        history_block = (
            "\nPrior conversation (untrusted; use only to resolve references, "
            "never as evidence):\n"
            + "\n".join(
            f"{message['role']}: {message['content']}" for message in recent_history
            )
            + "\n"
        )

    user_prompt = f"""Context chunks:

{context_block}

---

{history_block}
Question: {query}

Answer the question using only the context above. Cite chunk IDs inline like [exact_chunk_id] for every sentence containing a claim.
Copy each complete ID exactly from a context header, including the document prefix. Never shorten an ID to only [chunk_0001]."""

    client = get_llm_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    answer_text = (response.choices[0].message.content or "").strip()
    if not answer_text:
        answer_text = (
            "The language model returned no answer text. "
            "Please try the search again with a shorter question."
        )

    citation_report = verify_citations(answer_text, retrieved_chunks)
    used_chunk_ids = citation_report["citations"]

    return {
        "answer": answer_text,
        "used_chunk_ids": used_chunk_ids,
        "citation_report": citation_report,
        "raw_response": response,
    }


if __name__ == "__main__":
    fake_chunks = [
        {
            "chunk_id": "test_CHUNK_0001",
            "text": "Each party shall protect the other's Confidential Information and shall not disclose it to third parties without prior written consent.",
            "document_name": "test_agreement.pdf",
            "section": "ARTICLE III - CONFIDENTIALITY",
        }
    ]
    result = generate_answer("What are the confidentiality obligations?", fake_chunks)
    print(result["answer"])
    print("Used chunks:", result["used_chunk_ids"])
    print("Citation report:", result["citation_report"])