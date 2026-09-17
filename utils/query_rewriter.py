import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

_FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|they|them|this|that|these|those|he|she|above|before|same|also|instead|what about|how about|why)\b",
    re.IGNORECASE,
)


def looks_like_follow_up(query, history):
    """Use a cheap heuristic so standalone questions avoid an extra LLM call."""
    return bool(history) and bool(_FOLLOW_UP_PATTERN.search(query or ""))


def rewrite_query(query, history, model=GROQ_MODEL, max_tokens=180):
    """Rewrite a follow-up as a standalone retrieval query.

    Returns the original query on configuration, network, or model failures.
    """
    if not query or not history or not looks_like_follow_up(query, history):
        return query

    api_key = os.environ.get(GROQ_API_KEY_ENV)
    if not api_key:
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

    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
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
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or query
    except Exception:
        return query
       