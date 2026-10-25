import unittest
import re
from email.utils import formatdate
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APIStatusError

from utils import generation


def response(status_code=200, headers=None):
    request = httpx.Request("POST", "https://api.groq.test/chat/completions")
    return httpx.Response(status_code, headers=headers or {}, request=request)


def rate_limit_error(retry_after="0"):
    return APIStatusError(
        "rate limited",
        response=response(429, {"retry-after": retry_after}),
        body={"error": {"code": "rate_limit_exceeded"}},
    )


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class GenerationTests(unittest.TestCase):
    def test_context_budget_deduplicates_and_preserves_citation_ids(self):
        chunks = [
            {"chunk_id": "Doc_CHUNK_1", "document_name": "Doc.pdf", "text": "A" * 4000},
            {"chunk_id": "Doc_CHUNK_1", "document_name": "Doc.pdf", "text": "duplicate"},
            {"chunk_id": "Doc_CHUNK_2", "document_name": "Doc.pdf", "text": "B" * 4000},
        ]

        block = generation._build_context_block(chunks, char_budget=1000)

        self.assertLessEqual(len(block), 1000)
        self.assertIn("[Doc_CHUNK_1]", block)
        self.assertNotIn("[Doc_CHUNK_2]", block)

    def test_context_keeps_final_citation_when_chunk_fits_budget(self):
        block = generation._build_context_block(
            [{
                "chunk_id": "Doc_CHUNK_1",
                "document_name": "agreement.pdf",
                "page": 3,
                "text": "Evidence [Doc_CHUNK_1].",
            }],
            char_budget=500,
        )

        self.assertIn("[Doc_CHUNK_1].", block)
        self.assertIn("(page: 3)", block)

    def test_429_retries_then_succeeds_with_retry_after(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise rate_limit_error("0")
            return completion("answer")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(generation.time, "sleep") as sleep:
            result = generation.create_chat_completion(client, model="test-model")

        self.assertEqual(result.choices[0].message.content, "answer")
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(0.0)

    def test_429_respects_http_date_retry_after(self):
        error = rate_limit_error(formatdate(1030, usegmt=True))

        with patch.object(generation.time, "time", return_value=1000):
            delay = generation._retry_delay(error, 0)

        self.assertEqual(delay, 30)

    def test_exhausted_rate_limit_returns_user_friendly_error(self):
        error = rate_limit_error("12")

        message = generation.user_friendly_model_error(error)

        self.assertIn("after retries", message)
        self.assertIn("No unsupported summary", message)

    def test_permanent_error_is_not_retried(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            raise APIStatusError(
                "bad request",
                response=response(400),
                body={"error": {"code": "invalid_request"}},
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with patch.object(generation.time, "sleep") as sleep:
            with self.assertRaises(APIStatusError):
                generation.create_chat_completion(client, model="test-model")

        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_exhausted_daily_token_budget_is_not_retried(self):
        error = APIStatusError(
            "daily token budget exhausted",
            response=response(429),
            body={"error": {"message": "daily token budget exhausted"}},
        )
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            raise error

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with patch.object(generation.time, "sleep") as sleep:
            with self.assertRaises(APIStatusError):
                generation.create_chat_completion(client, model="test-model")

        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_long_summary_uses_bounded_map_reduce_and_citations(self):
        chunks = [
            {"chunk_id": f"Doc_CHUNK_{index}", "document_name": "Doc.pdf", "text": word * 1800}
            for index, word in enumerate("ABC", start=1)
        ]
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) <= 3:
                index = len(calls)
                return completion(f"- Evidence point [Doc_CHUNK_{index}]")
            return completion(
                "### Overview\nEvidence point [Doc_CHUNK_1] [Doc_CHUNK_2] [Doc_CHUNK_3]."
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(generation, "_get_client", return_value=client):
            result = generation.generate_answer("Summarize this PDF", chunks, model="test-model")

        self.assertEqual(len(calls), 4)
        self.assertEqual(result["used_chunk_ids"], [
            "Doc_CHUNK_1", "Doc_CHUNK_2", "Doc_CHUNK_3"
        ])
        self.assertIn("[Doc_CHUNK_1]", result["answer"])
        for call in calls:
            prompt = call["messages"][-1]["content"]
            self.assertLessEqual(len(prompt), 7000)

    def test_summary_batches_small_chunks_to_reduce_map_calls(self):
        chunks = [
            {
                "chunk_id": f"Doc_CHUNK_{index}",
                "document_name": "Doc.pdf",
                "text": f"Evidence text {index}. " + ("evidence " * 110),
            }
            for index in range(1, 7)
        ]
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            prompt = kwargs["messages"][-1]["content"]
            citations = re.findall(r"\[(Doc_CHUNK_\d+)\]", prompt)
            unique_citations = list(dict.fromkeys(citations))
            if len(calls) <= 3:
                return completion(
                    "- Summary " + " ".join(f"[{chunk_id}]" for chunk_id in unique_citations)
                )
            return completion(
                "Summary " + " ".join(f"[{chunk_id}]" for chunk_id in unique_citations)
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(generation, "_get_client", return_value=client):
            result = generation.generate_answer(
                "Summarize this PDF",
                chunks,
                model="test-model",
            )

        self.assertEqual(len(calls), 4)
        self.assertEqual(
            result["used_chunk_ids"],
            [f"Doc_CHUNK_{index}" for index in range(1, 7)],
        )
        for call in calls:
            self.assertLessEqual(
                len(call["messages"][-1]["content"]),
                generation.MAP_STAGE_CHAR_BUDGET + 300,
            )

    def test_partial_map_failure_does_not_return_incomplete_summary(self):
        chunks = [
            {
                "chunk_id": f"Doc_CHUNK_{index}",
                "document_name": "Doc.pdf",
                "text": f"Evidence section {index}. " + ("evidence " * 400),
            }
            for index in range(1, 4)
        ]
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                return completion("")
            return completion("- Summary [Doc_CHUNK_1]")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(generation, "_get_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "summary is incomplete"):
                generation.generate_answer(
                    "Summarize this PDF",
                    chunks,
                    model="test-model",
                )

        self.assertEqual(len(calls), 3)

    def test_empty_retrieval_returns_evidence_based_message_without_api_call(self):
        with patch.object(generation, "_get_client") as get_client:
            result = generation.generate_answer("Summarize this PDF", [])

        get_client.assert_not_called()
        self.assertEqual(result["used_chunk_ids"], [])
        self.assertIn("No relevant context", result["answer"])


if __name__ == "__main__":
    unittest.main()
