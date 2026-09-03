import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """You are a legal assistant answering questions strictly using the provided context chunks from legal documents.

Rules:
- Only use information present in the provided context. Do not use outside knowledge.
- Every claim in your answer must be traceable to a specific chunk. Cite chunks inline like [chunk_id].
- If the context does not contain enough information to answer confidently, say so explicitly instead of guessing.
- Be precise and avoid restating the question.
- Keep the answer focused and avoid unnecessary preamble.
"""


_CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+)\]")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")

# Set to True to see citation-matching debug output in the console.
_DEBUG_CITATIONS = True


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
    """Group wrapped markdown bullets so a citation on the final line covers them."""
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
    return units


def verify_citations(answer_text, retrieved_chunks):
    """Validate citation references and coverage of claim lines in an answer.

    Two lookup paths are supported for each citation found in the answer:
      1. Exact/normalized match against a real chunk_id from retrieved_chunks.
      2. Alias match against just the "chunk_N" suffix, IF that suffix is
         unique among the retrieved chunks (avoids ambiguity when multiple
         documents each have their own chunk_0001, chunk_0002, ...).
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

    if _DEBUG_CITATIONS:
        print(
            "[citation] extracted_citation_ids=" + repr(citations)
            + " known_chunk_ids=" + repr(list(known_ids.values()))
            + " aliases=" + repr(aliases)
        )

    valid_citations = sorted({
        known_ids.get(_normalize_chunk_id(citation), aliases.get(_normalize_chunk_id(citation)))
        for citation in citations
        if _normalize_chunk_id(citation) in known_ids or _normalize_chunk_id(citation) in aliases
    })
    unknown_citations = sorted({
        citation.strip() for citation in citations
        if _normalize_chunk_id(citation) not in known_ids
        and _normalize_chunk_id(citation) not in aliases
    })

    uncited_lines = []
    for line in _claim_units(answer_text):
        stripped = re.sub(r"[*_`>#-]", "", line).strip()
        if not stripped or stripped.endswith(":"):
            continue
        if _CITATION_PATTERN.search(line):
            continue
        uncited_lines.append(stripped)

    problems = []
    if unknown_citations:
        problems.append("Unknown citation(s): " + ", ".join(f"[{citation}]" for citation in unknown_citations))
    if uncited_lines:
        problems.append(f"{len(uncited_lines)} claim line(s) have no citation.")

    return {
        "is_valid": bool(valid_citations) and not problems,
        "known_chunk_ids": list(known_ids.values()),
        "citations": valid_citations,
        "unknown_citations": unknown_citations,
        "uncited_lines": uncited_lines,
        "problems": problems,
    }


def _get_client():
    api_key = os.environ.get(GROQ_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Missing {GROQ_API_KEY_ENV} environment variable. "
            f"Set it before calling the generation module."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


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


def generate_answer(query, retrieved_chunks, model=GROQ_MODEL, temperature=0.1, max_tokens=1200):
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

    user_prompt = f"""Context chunks:

{context_block}

---

Question: {query}

Answer the question using only the context above. Cite chunk IDs inline like [exact_chunk_id] for every claim.
Copy each complete ID exactly from a context header, including the document prefix. Never shorten an ID to only [chunk_0001]."""

    client = _get_client()

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

    if _DEBUG_CITATIONS:
        known_chunk_ids = [str(chunk["chunk_id"]) for chunk in retrieved_chunks]
        print("[citation] known_chunk_ids=" + repr(known_chunk_ids))

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