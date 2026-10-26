import json
import logging
import re

from utils.llm_client import GROQ_MODEL, get_llm_client


logger = logging.getLogger(__name__)

INTENTS = (
    "definition",
    "obligation",
    "rights",
    "procedure",
    "comparison",
    "exception",
    "document_fact",
    "other",
)
DIFFICULTIES = ("EASY", "MEDIUM", "HARD")

_INTENT_PATTERNS = (
    ("comparison", r"\b(compare|comparison|difference between|versus|vs\.?)\b"),
    ("exception", r"\b(exception|except|carve[- ]out|notwithstanding)\b"),
    ("procedure", r"\b(how to|how do|steps|process|procedure|file a|enforce)\b"),
    ("obligation", r"\b(obligation|must|shall|required|duty|responsibilit(?:y|ies))\b"),
    ("rights", r"\b(rights?|entitled|permitted|allowed)\b"),
    ("definition", r"\b(define|definition|meaning of|what does .+ mean)\b"),
)

_COMPLEXITY_PATTERNS = {
    "conditional": r"\b(if|unless|provided that|subject to|depending on)\b",
    "multiple_part": r"\b(and also|as well as|both .{1,60} and|in addition)\b",
    "multi_step": r"\b(step[- ]by[- ]step|in what order|all steps|each stage)\b",
    "cross_reference": r"\b(article|section|clause)\s+\d+.{0,80}\b(and|or|versus)\b",
}


def analyze_query_rules(query: str) -> dict:
    """Produce transparent intent and difficulty signals without an ML model."""
    normalized = " ".join((query or "").lower().split())
    token_count = len(re.findall(r"\b[\w'-]+\b", normalized))
    question_count = normalized.count("?")
    matched_intent = next(
        (
            intent
            for intent, pattern in _INTENT_PATTERNS
            if re.search(pattern, normalized)
        ),
        "document_fact",
    )
    signals = {
        name: bool(re.search(pattern, normalized))
        for name, pattern in _COMPLEXITY_PATTERNS.items()
    }
    has_comparison = matched_intent == "comparison"
    legal_concepts = len(
        re.findall(
            r"\b(contract|party|parties|claim|damages|liability|termination|"
            r"notice|consent|confidentiality|indemnity|jurisdiction|remedy)\b",
            normalized,
        )
    )

    if (
        has_comparison
        or question_count > 1
        or token_count >= 32
        or sum(signals.values()) >= 2
        or legal_concepts >= 3
    ):
        proposed_difficulty = "HARD"
    elif (
        token_count >= 16
        or any(signals.values())
        or matched_intent == "procedure"
        or legal_concepts >= 2
    ):
        proposed_difficulty = "MEDIUM"
    else:
        proposed_difficulty = "EASY"

    return {
        "token_count": token_count,
        "question_count": question_count,
        "legal_concept_count": legal_concepts,
        "signals": {
            **signals,
            "comparison": has_comparison,
        },
        "proposed_intent": matched_intent,
        "proposed_difficulty": proposed_difficulty,
    }


def max_retrieval_rounds(difficulty: str) -> int:
    """Enforce retrieval limits in application code, not from model output."""
    return 3 if (difficulty or "").upper() == "HARD" else 1


def _parse_json_object(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")
    return result


def classify_query(query: str, model: str = GROQ_MODEL) -> dict:
    """Combine rule signals with a general-purpose LLM's final classification."""
    rule_analysis = analyze_query_rules(query)
    fallback = {
        "intent": rule_analysis["proposed_intent"],
        "difficulty": rule_analysis["proposed_difficulty"],
        "rule_analysis": rule_analysis,
        "source": "rules",
    }
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's legal-document query. The rule analysis is "
                        "advisory; make the final decision yourself using both it and the "
                        "original query. Return only a JSON object with string fields "
                        "'intent' and 'difficulty'. Allowed intents: "
                        + ", ".join(INTENTS)
                        + ". Allowed difficulties: EASY, MEDIUM, HARD. Use HARD for "
                        "multi-issue, comparative, or otherwise demanding questions."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_query": query,
                            "rule_based_analysis": rule_analysis,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0,
            max_tokens=120,
        )
        parsed = _parse_json_object(response.choices[0].message.content)
        intent = parsed.get("intent")
        difficulty = parsed.get("difficulty")
        if not isinstance(intent, str) or intent.lower() not in INTENTS:
            raise ValueError("The model returned an unsupported query intent.")
        if not isinstance(difficulty, str) or difficulty.upper() not in DIFFICULTIES:
            raise ValueError("The model returned an unsupported query difficulty.")
        return {
            "intent": intent.lower(),
            "difficulty": difficulty.upper(),
            "rule_analysis": rule_analysis,
            "source": "llm",
        }
    except Exception as exc:
        logger.exception(
            "Query classification fell back to rule analysis: %s", exc
        )
        return {**fallback, "fallback_reason": str(exc)}


def assess_evidence(query: str, search_query: str, hits: list[dict], model: str = GROQ_MODEL) -> dict:
    """Check whether retrieved passages cover a HARD query and propose a focused search."""
    evidence = [
        {
            "chunk_id": hit.get("chunk_id", ""),
            "document_name": hit.get("document_name", ""),
            "section": hit.get("section", ""),
            "text": (hit.get("text") or "")[:1800],
        }
        for hit in hits[:10]
    ]
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Assess only whether the supplied legal-document passages contain "
                        "enough evidence to answer every material part of the original "
                        "question. Do not use outside knowledge. Return only JSON with "
                        "'sufficient' (boolean) and 'next_query' (string). If evidence is "
                        "sufficient, set next_query to an empty string. Otherwise provide "
                        "one concise, specific retrieval query targeting the missing evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_query": query,
                            "current_search_query": search_query,
                            "retrieved_evidence": evidence,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0,
            max_tokens=180,
        )
        parsed = _parse_json_object(response.choices[0].message.content)
        sufficient = parsed.get("sufficient")
        next_query = parsed.get("next_query", "")
        if not isinstance(sufficient, bool) or not isinstance(next_query, str):
            raise ValueError("The evidence assessment returned invalid fields.")
        return {
            "sufficient": sufficient,
            "next_query": next_query.strip(),
            "source": "llm",
        }
    except Exception as exc:
        logger.exception(
            "Evidence assessment failed; stopping further retrieval: %s", exc
        )
        return {
            "sufficient": False,
            "next_query": "",
            "source": "error",
            "error": str(exc),
        }
