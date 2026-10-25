import os
import random
import re
import time
import logging
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from dotenv import dotenv_values, load_dotenv

PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PROJECT_ENV_FILE)

GROQ_API_KEY_ENV = "GROQ_API_KEY"
LEGACY_GROK_API_KEY_ENV = "GROK_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CONTEXT_TOKEN_BUDGET = int(os.getenv("LEGAL_RAG_CONTEXT_TOKEN_BUDGET", "1300"))
MAP_STAGE_TOKEN_BUDGET = int(os.getenv("LEGAL_RAG_MAP_CONTEXT_TOKEN_BUDGET", "650"))
CONTEXT_CHAR_BUDGET = int(
    os.getenv("LEGAL_RAG_CONTEXT_CHAR_BUDGET", str(CONTEXT_TOKEN_BUDGET * 4))
)
MAP_STAGE_CHAR_BUDGET = int(
    os.getenv("LEGAL_RAG_MAP_CHAR_BUDGET", str(MAP_STAGE_TOKEN_BUDGET * 4))
)
MAX_MODEL_RETRIES = int(os.getenv("LEGAL_RAG_MAX_MODEL_RETRIES", "3"))
MAX_OUTPUT_TOKENS = int(os.getenv("LEGAL_RAG_MAX_OUTPUT_TOKENS", "1200"))
MAX_SUMMARY_OUTPUT_TOKENS = int(os.getenv("LEGAL_RAG_MAX_SUMMARY_OUTPUT_TOKENS", "320"))

SYSTEM_PROMPT = """You are a legal assistant answering questions strictly using the provided context chunks from legal documents.

Rules:
- Only use information present in the provided context. Do not use outside knowledge.
- Every claim in your answer must be traceable to a specific chunk. Cite chunks inline like [chunk_id].
- If the context does not contain enough information to answer confidently, say so explicitly instead of guessing.
- Be precise and avoid restating the question.
- Keep the answer focused and avoid unnecessary preamble.
- Use relevant Markdown headings such as Overview, Applicable Legal Provisions, Eligibility or Requirements, Procedure, Documents Required, Exceptions and Limitations, Conclusion, or Sources and Citations.
- Use short paragraphs, bullets for multiple conditions, and numbered lists for sequential steps. Do not repeat a heading or add blank lines between every item.
"""


_CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+)\]")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
_SUMMARY_QUERY_PATTERN = re.compile(
    r"\b(summarize|summary|summarise|overview|key points|key takeaways)\b",
    re.IGNORECASE,
)

# Set to True to see citation-matching debug output in the console.
_DEBUG_CITATIONS = False
logger = logging.getLogger(__name__)


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


@lru_cache(maxsize=1)
def _get_client():
    # Accept the existing GROK_API_KEY spelling while standardizing on GROQ_API_KEY.
    file_values = dotenv_values(PROJECT_ENV_FILE)
    api_key = (
        os.environ.get(GROQ_API_KEY_ENV)
        or os.environ.get(LEGACY_GROK_API_KEY_ENV)
        or file_values.get(GROQ_API_KEY_ENV)
        or file_values.get(LEGACY_GROK_API_KEY_ENV)
    )
    if not api_key:
        raise RuntimeError(
            f"Missing {GROQ_API_KEY_ENV} environment variable. "
            f"Set it before calling the generation module."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _is_summary_query(query: str) -> bool:
    return bool(_SUMMARY_QUERY_PATTERN.search(query or ""))


def _retry_delay(error: BaseException, attempt: int) -> Optional[float]:
    if not isinstance(error, APIStatusError) or error.status_code != 429:
        return None

    error_body = str(getattr(error, "body", "")).lower()
    exhausted_terms = (
        "tokens per day",
        "daily token",
        "token budget exhausted",
        "token limit exhausted",
        "insufficient_quota",
        "billing limit",
    )
    if any(term in error_body for term in exhausted_terms):
        return None

    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                retry_time = parsedate_to_datetime(str(retry_after))
                delay = max(0.0, retry_time.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                delay = None
        if delay is not None:
            return max(0.0, delay) if delay <= 60.0 else None
    return min(8.0, (2 ** attempt) + random.uniform(0.0, 0.75))


def user_friendly_model_error(error: BaseException) -> str:
    """Return an actionable, non-fabricated message for model-provider failures."""
    if isinstance(error, APIStatusError) and error.status_code == 429:
        error_body = str(getattr(error, "body", "")).lower()
        exhausted_terms = (
            "tokens per day",
            "daily token",
            "token budget exhausted",
            "token limit exhausted",
            "insufficient_quota",
            "billing limit",
        )
        if any(term in error_body for term in exhausted_terms):
            return (
                "The configured Groq token or billing budget is exhausted. "
                "Wait for the quota to reset or check the provider account. "
                "No unsupported summary was generated."
            )
        return (
            "Groq rate-limited this request after retries within the configured bounds. "
            "Wait briefly or retry later. "
            "No unsupported summary was generated."
        )
    if isinstance(error, APITimeoutError):
        return "The configured language model timed out. Please retry the question."
    if isinstance(error, APIConnectionError):
        return (
            "The configured language model could not be reached. "
            "Check network connectivity and retry."
        )
    if isinstance(error, APIStatusError):
        if error.status_code in {401, 403}:
            return (
                "The configured Groq credentials or model permissions were rejected. "
                "Check the server configuration."
            )
        if error.status_code >= 500:
            return (
                "The Groq service returned a temporary server error. "
                "Please retry later."
            )
        return (
            "The configured language model rejected the request "
            f"(HTTP {error.status_code}). Check server logs for details."
        )
    if isinstance(error, RuntimeError) and "Missing GROQ_API_KEY" in str(error):
        return (
            "The configured language model is unavailable because GROQ_API_KEY "
            "is not configured on the server."
        )
    return (
        "The configured language model request failed "
        f"({type(error).__name__}). Check server logs for details."
    )


def create_chat_completion(
    client: Any,
    stage: str = "llm_completion",
    **kwargs: Any,
) -> Any:
    """Call Groq with bounded retries for HTTP 429 responses only."""
    for attempt in range(MAX_MODEL_RETRIES + 1):
        started = time.perf_counter()
        try:
            return client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None or attempt >= MAX_MODEL_RETRIES:
                raise
            logger.warning(
                "pipeline_stage=%s rate_limit_retry=%d delay_seconds=%.3f",
                stage,
                attempt + 1,
                delay,
            )
            time.sleep(delay)
        finally:
            logger.info(
                "pipeline_stage=%s api_attempt=%d duration_seconds=%.3f",
                stage,
                attempt + 1,
                time.perf_counter() - started,
            )


def _chunk_context(chunk: Dict[str, Any], char_limit: int) -> Optional[str]:
    chunk_id = str(chunk.get("chunk_id") or "").strip()
    text = str(chunk.get("text") or "").strip()
    if not chunk_id or not text:
        return None

    header = f"[{chunk_id}]"
    source = str(chunk.get("document_name") or chunk.get("file_name") or "").strip()
    if source:
        header += f" (source: {source})"
    metadata = chunk.get("metadata") or {}
    page = (
        chunk.get("page")
        or chunk.get("page_number")
        or metadata.get("page")
        or metadata.get("page_number")
    )
    if page is not None and str(page).strip():
        header += f" (page: {page})"
    available = max(0, char_limit - len(header) - 2)
    if available <= 80:
        return None
    if len(text) > available:
        excerpt = text[:available].rsplit(" ", 1)[0].strip()
        excerpt = excerpt.rstrip(".,;:") + "..."
    else:
        excerpt = text
    return f"{header}\n{excerpt}"


def _build_context_block(
    chunks: Iterable[Dict[str, Any]],
    char_budget: int = CONTEXT_CHAR_BUDGET,
) -> str:
    """Include only budgeted, unique evidence and preserve every citation ID."""
    seen_ids = set()
    seen_text = set()
    parts: List[str] = []
    remaining = char_budget
    separator_length = len("\n\n---\n\n")

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        text = str(chunk.get("text") or "").strip()
        if not chunk_id or chunk_id in seen_ids or not text:
            continue
        normalized_text = re.sub(r"\s+", " ", text)
        if normalized_text in seen_text:
            continue
        block = _chunk_context(chunk, remaining - separator_length)
        if not block:
            continue
        seen_ids.add(chunk_id)
        seen_text.add(normalized_text)
        parts.append(block)
        remaining -= len(block) + separator_length
        if remaining <= 80:
            break
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query,
    retrieved_chunks,
    model=GROQ_MODEL,
    temperature=0.1,
    max_tokens=MAX_OUTPUT_TOKENS,
    conversation_history=None,
):
    started = time.perf_counter()
    try:
        return _generate_answer(
            query,
            retrieved_chunks,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            conversation_history=conversation_history,
        )
    finally:
        logger.info(
            "pipeline_stage=answer_generation_total duration_seconds=%.3f",
            time.perf_counter() - started,
        )


def _generate_answer(
    query,
    retrieved_chunks,
    model,
    temperature,
    max_tokens,
    conversation_history,
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

    context_started = time.perf_counter()
    context_chars = sum(len(str(chunk.get("text") or "")) for chunk in retrieved_chunks)
    summary_required = _is_summary_query(query) and context_chars > CONTEXT_CHAR_BUDGET
    client = _get_client()

    if summary_required:
        map_batches = []
        current_batch = []
        for chunk in retrieved_chunks:
            current_context = _build_context_block(
                current_batch,
                char_budget=MAP_STAGE_CHAR_BUDGET,
            )
            chunk_context = _build_context_block(
                [chunk],
                char_budget=MAP_STAGE_CHAR_BUDGET,
            )
            candidate_batch = current_batch + [chunk]
            candidate_context = _build_context_block(
                candidate_batch,
                char_budget=MAP_STAGE_CHAR_BUDGET,
            )
            expected_length = len(current_context) + (
                len("\n\n---\n\n") if current_context and chunk_context else 0
            ) + len(chunk_context)
            if current_batch and len(candidate_context) < expected_length:
                map_batches.append(current_batch)
                current_batch = [chunk]
            else:
                current_batch = candidate_batch
        if current_batch:
            map_batches.append(current_batch)

        map_outputs = []
        map_started = time.perf_counter()
        map_requests = 0
        map_failures = 0
        for batch in map_batches:
            map_context = _build_context_block(
                batch,
                char_budget=MAP_STAGE_CHAR_BUDGET,
            )
            if not map_context:
                map_failures += 1
                continue
            map_prompt = (
                f"Summarize only this evidence for: {query.strip()}\n\n"
                f"{map_context}\n\n"
                "Return concise factual bullets for each source chunk. Preserve citations "
                "exactly as shown and never add facts not stated in the evidence."
            )
            map_requests += 1
            map_response = create_chat_completion(
                client,
                stage="summary_map",
                model=model,
                messages=[
                    {"role": "system", "content": "Evidence-only summarizer. Keep every source citation."},
                    {"role": "user", "content": map_prompt},
                ],
                temperature=0,
                max_tokens=min(
                    max_tokens,
                    MAX_SUMMARY_OUTPUT_TOKENS * len(batch),
                ),
                timeout=30,
            )
            map_text = (map_response.choices[0].message.content or "").strip()
            if map_text:
                map_outputs.append(map_text)
            else:
                map_failures += 1
        logger.info(
            "pipeline_stage=summary_map_total duration_seconds=%.3f "
            "batches=%d api_calls=%d failures=%d",
            time.perf_counter() - map_started,
            len(map_batches),
            map_requests,
            map_failures,
        )
        if map_failures or len(map_outputs) != len(map_batches):
            raise RuntimeError(
                "The document summary is incomplete because one or more evidence "
                "sections could not be summarized."
            )
        context_block = _build_context_block(
            [{
                "chunk_id": f"SUMMARY_SOURCE_{index}",
                "document_name": "Map-stage evidence",
                "text": summary,
            } for index, summary in enumerate(map_outputs, start=1)],
            char_budget=CONTEXT_CHAR_BUDGET,
        )
    else:
        context_block = _build_context_block(retrieved_chunks)
    logger.info(
        "pipeline_stage=context_preparation duration_seconds=%.3f context_chars=%d summary=%s",
        time.perf_counter() - context_started,
        len(context_block),
        summary_required,
    )

    history_started = time.perf_counter()
    history_block = ""
    if conversation_history and not summary_required:
        recent_history = conversation_history[-2:]
        history_block = "\nConversation context (resolve references only):\n" + "\n".join(
            f"{message['role']}: {str(message.get('content', ''))[:500]}"
            for message in recent_history
        ) + "\n"
    logger.info(
        "pipeline_stage=conversation_history_preparation duration_seconds=%.3f "
        "messages=%d included=%s",
        time.perf_counter() - history_started,
        min(len(conversation_history or []), 2),
        bool(history_block),
    )

    user_prompt = f"""Evidence:
{context_block}

{history_block}Question: {query}

Answer only from the evidence. Preserve every source citation. If the evidence is insufficient, say so.
Use short Markdown paragraphs/lists and relevant headings only."""

    logger.info("pipeline_stage=answer_generation_request api_calls=1")
    response = create_chat_completion(
        client,
        stage="answer_generation",
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
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

    citation_started = time.perf_counter()
    citation_report = verify_citations(answer_text, retrieved_chunks)
    used_chunk_ids = citation_report["citations"]
    logger.info(
        "pipeline_stage=citation_validation duration_seconds=%.3f citations=%d",
        time.perf_counter() - citation_started,
        len(used_chunk_ids),
    )

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