import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils import query_rewriter


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class QueryRewriterTests(unittest.TestCase):
    def test_standalone_question_does_not_make_model_call(self):
        with patch.object(query_rewriter, "_get_client") as get_client:
            rewritten = query_rewriter.rewrite_query(
                "What are the termination terms?",
                [{"role": "assistant", "content": "The agreement has a term."}],
            )

        self.assertEqual(rewritten, "What are the termination terms?")
        get_client.assert_not_called()

    def test_follow_up_reuses_configured_client_and_returns_rewritten_query(self):
        client = object()
        response = completion("termination obligations under the agreement")

        with patch.object(query_rewriter, "_get_client", return_value=client), patch.object(
            query_rewriter,
            "create_chat_completion",
            return_value=response,
        ) as create:
            rewritten = query_rewriter.rewrite_query(
                "What about termination?",
                [{"role": "assistant", "content": "The agreement describes payment terms."}],
            )

        self.assertEqual(rewritten, "termination obligations under the agreement")
        create.assert_called_once()
        self.assertIs(create.call_args.args[0], client)
        self.assertEqual(create.call_args.kwargs["stage"], "query_rewrite")
        self.assertEqual(create.call_args.kwargs["timeout"], 30)

    def test_follow_up_failure_falls_back_to_original_question(self):
        with patch.object(query_rewriter, "_get_client", side_effect=RuntimeError("offline")):
            rewritten = query_rewriter.rewrite_query(
                "What about termination?",
                [{"role": "assistant", "content": "The agreement describes payment terms."}],
            )

        self.assertEqual(rewritten, "What about termination?")


if __name__ == "__main__":
    unittest.main()
