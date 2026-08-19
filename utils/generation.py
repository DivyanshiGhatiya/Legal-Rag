import os
from openai import OpenAI

GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a legal assistant answering questions strictly using the provided context chunks from legal documents.

Rules:
- Only use information present in the provided context. Do not use outside knowledge.
- Every claim in your answer must be traceable to a specific chunk. Cite chunks inline like [chunk_id].
- If the context does not contain enough information to answer confidently, say so explicitly instead of guessing.
- Be precise and avoid restating the question.
- Keep the answer focused and avoid unnecessary preamble.
"""


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


def generate_answer(query, retrieved_chunks, model=GROQ_MODEL, temperature=0.1, max_tokens=800):
    """
    Send retrieved chunks + query to Grok and get a citation-aware answer.

    Parameters
    ----------
    query : str
        The user's question.
    retrieved_chunks : list[dict]
        Output of retrieval.retrieve() — each dict must have 'chunk_id' and 'text',
        optionally 'document_name' and 'section'.

    Returns
    -------
    dict with keys: answer (str), used_chunk_ids (list[str]), raw_response
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant context was retrieved for this question, so I can't answer it from the documents.",
            "used_chunk_ids": [],
            "raw_response": None,
        }

    context_block = _build_context_block(retrieved_chunks)

    user_prompt = f"""Context chunks:

{context_block}

---

Question: {query}

Answer the question using only the context above. Cite chunk IDs inline like [chunk_id] for every claim."""

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

    answer_text = response.choices[0].message.content

    used_chunk_ids = [c["chunk_id"] for c in retrieved_chunks if c["chunk_id"] in answer_text]

    return {
        "answer": answer_text,
        "used_chunk_ids": used_chunk_ids,
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