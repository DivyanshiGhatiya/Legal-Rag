import logging
import re
import time

from utils.generation import GROQ_MODEL, _get_client, create_chat_completion

_FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|they|them|this|that|these|those|he|she|above|before|same|also|instead|what about|how about|why)\b",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


def looks_like_follow_up(query, history):
    """Use a cheap heuristic so standalone questions avoid an extra LLM call."""
    return bool(history) and bool(_FOLLOW_UP_PATTERN.search(query or ""))


def rewrite_query(query, history, model=GROQ_MODEL, max_tokens=180):
    """Rewrite a follow-up as a standalone retrieval query.

    Returns the original query on configuration, network, or model failures.
    """
    if not query or not history or not looks_like_follow_up(query, history):
        return query

    recent_history = history[-6:]
    transcript = "\n".join(
        f"{message['role']}: {message['content']}" for message in recent_history
    )
    prompt = f"""Rewrite the latest user question as a standalone legal-document retrieval query.
Keep the legal meaning precise. Resolve references using the conversation when possible.
Return only the rewritten query. Do not answer it and do not add citations.

Conversation:
{transcript}

Latest question:
{query}"""

    started = time.perf_counter()
    try:
        client = _get_client()
        response = create_chat_completion(
            client,
            stage="query_rewrite",
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite questions for retrieval without adding facts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            timeout=30,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        logger.info(
            "pipeline_stage=query_rewrite duration_seconds=%.3f status=%s",
            time.perf_counter() - started,
            "rewritten" if rewritten else "empty_response",
        )
        return rewritten or query
    except Exception as exc:
        logger.exception(
            "pipeline_stage=query_rewrite duration_seconds=%.3f "
            "status=fallback error_type=%s",
            time.perf_counter() - started,
            type(exc).__name__,
        )
        return query
       