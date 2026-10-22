import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.query_classifier import (
    analyze_query_rules,
    assess_evidence,
    classify_query,
    max_retrieval_rounds,
)


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class QueryClassifierTests(unittest.TestCase):
    def test_simple_definition_is_easy(self):
        analysis = analyze_query_rules("What does indemnity mean?")

        self.assertEqual(analysis["proposed_intent"], "definition")
        self.assertEqual(analysis["proposed_difficulty"], "EASY")

    def test_comparison_is_hard(self):
        analysis = analyze_query_rules(
            "Compare the parties' liability and indemnity obligations."
        )

        self.assertEqual(analysis["proposed_intent"], "comparison")
        self.assertEqual(analysis["proposed_difficulty"], "HARD")

    def test_round_limits_are_fixed_in_code(self):
        self.assertEqual(max_retrieval_rounds("EASY"), 1)
        self.assertEqual(max_retrieval_rounds("MEDIUM"), 1)
        self.assertEqual(max_retrieval_rounds("HARD"), 3)
        self.assertEqual(max_retrieval_rounds("unknown"), 1)

    def test_llm_receives_query_and_rule_analysis_and_can_override(self):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return _response('{"intent":"procedure","difficulty":"MEDIUM"}')

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        with patch("utils.query_classifier.get_llm_client", return_value=client):
            result = classify_query("What does indemnity mean?")

        self.assertEqual(result["intent"], "procedure")
        self.assertEqual(result["difficulty"], "MEDIUM")
        self.assertEqual(result["source"], "llm")
        user_message = next(
            message["content"]
            for message in requests[0]["messages"]
            if message["role"] == "user"
        )
        payload = json.loads(user_message)
        self.assertEqual(payload["original_query"], "What does indemnity mean?")
        self.assertEqual(payload["rule_based_analysis"], result["rule_analysis"])

    def test_invalid_llm_classification_uses_rule_fallback(self):
        def create(**kwargs):
            self.assertIn("model", kwargs)
            return _response('{"intent":"unknown","difficulty":"HARD"}')

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        with patch("utils.query_classifier.get_llm_client", return_value=client):
            result = classify_query("What does indemnity mean?")

        self.assertEqual(result["intent"], "definition")
        self.assertEqual(result["difficulty"], "EASY")
        self.assertEqual(result["source"], "rules")
        self.assertIn("fallback_reason", result)

    def test_evidence_assessment_returns_refinement(self):
        def create(**kwargs):
            self.assertIn("model", kwargs)
            return _response(
                '{"sufficient":false,"next_query":"notice deadline after breach"}'
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        with patch("utils.query_classifier.get_llm_client", return_value=client):
            result = assess_evidence("What happens after breach?", "breach", [])

        self.assertFalse(result["sufficient"])
        self.assertEqual(result["next_query"], "notice deadline after breach")


if __name__ == "__main__":
    unittest.main()
