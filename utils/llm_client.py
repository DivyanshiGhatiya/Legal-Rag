import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from openai import OpenAI


PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PROJECT_ENV_FILE)

GROQ_API_KEY_ENV = "GROQ_API_KEY"
LEGACY_GROK_API_KEY_ENV = "GROK_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def get_llm_client() -> OpenAI:
    """Create the shared Groq-compatible client using the project's key aliases."""
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
            "Set it before calling the language model."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
