# utils/grok_client.py
"""
Thin wrapper around the Grok LLM API.
We treat Grok as an OpenAI‑compatible chat endpoint.
"""

import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load .env if present (does nothing if file absent)
load_dotenv()


def _get_client() -> OpenAI:
    """
    Create and return an OpenAI client configured for Grok.
    Raises a clear error if the API key is missing.
    """
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROK_API_KEY not found. Set it in a .env file or the environment."
        )
    # Grok's API is OpenAI‑compatible; adjust base_url if the provider changes.
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def get_grok_response(prompt: str, model: str = "grok-beta") -> str:
    """
    Send a single‑turn prompt to Grok and return the text response.

    Parameters
    ----------
    prompt: str
        The user's message.
    model: str, optional
        Model identifier supported by Grok (default: "grok-beta").

    Returns
    -------
    str
        The assistant's reply.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,   # low temperature for more deterministic legal answers
            max_tokens=500,    # enough for a concise answer; increase later if needed
        )
        # The OpenAI‑compatible response shape:
        # response.choices[0].message.content
        return response.choices[0].message.content.strip()
    except Exception as exc:
        # In a real app we would log this; for Week 1 we surface it to the UI.
        return f"⚠️ Error calling Grok API: {exc}"
