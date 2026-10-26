import unittest
from unittest.mock import patch

import app


def _hit(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "document_name": "contract.pdf",
        "section": "Terms",
    }


class QueryPipelineTests(unittest.TestCase):
    def _classification(self, difficulty):
        return {
            "intent": "comparison",
            "difficulty": difficulty,
            "rule_analysis": {},
            "source": "llm",
            "max_rounds": 99,
        }

    def test_easy_query_retrieves_only_once(self):
        with (
            patch.object(app, "classify_query", return_value=self._classification("EASY")),
            patch.object(app, "retrieve", return_value=[_hit("c1", "Evidence")]) as retrieve,
            patch.object(app, "generate_answer", return_value={"answer": "Answer"}),
        ):
            result = app.query_legal_documents("What is the term?")

        self.assertTrue(result["success"])
        self.assertEqual(retrieve.call_count, 1)
        self.assertEqual(len(result["retrieval_rounds"]), 1)

    def test_hard_query_stops_when_evidence_is_sufficient(self):
        with (
            patch.object(app, "classify_query", return_value=self._classification("HARD")),
            patch.object(app, "retrieve", return_value=[_hit("c1", "Evidence")]) as retrieve,
            patch.object(
                app,
                "assess_evidence",
                return_value={"sufficient": True, "next_query": "", "source": "llm"},
            ) as assess,
            patch.object(app, "generate_answer", return_value={"answer": "Answer"}),
        ):
            result = app.query_legal_documents("Compare both obligations.")

        self.assertTrue(result["success"])
        self.assertEqual(retrieve.call_count, 1)
        assess.assert_called_once()
        self.assertTrue(result["retrieval_rounds"][0]["evidence_sufficient"])

    def test_hard_query_never_exceeds_three_rounds(self):
        retrieval_queries = []
        assessments = []

        def retrieve(query, **_kwargs):
            self.assertIn("collection_name", _kwargs)
            retrieval_queries.append(query)
            return [_hit(f"c{len(retrieval_queries)}", "Evidence")]

        def assess(query, search_query, hits):
            assessments.append((query, search_query, len(hits)))
            return {
                "sufficient": False,
                "next_query": f"follow-up {len(retrieval_queries)}",
                "source": "llm",
            }

        with (
            patch.object(app, "classify_query", return_value=self._classification("HARD")),
            patch.object(app, "retrieve", side_effect=retrieve),
            patch.object(app, "assess_evidence", side_effect=assess),
            patch.object(app, "generate_answer", return_value={"answer": "Answer"}),
        ):
            result = app.query_legal_documents("Compare both obligations.")

        self.assertTrue(result["success"])
        self.assertEqual(retrieval_queries, [
            "Compare both obligations.",
            "follow-up 1",
            "follow-up 2",
        ])
        self.assertEqual(
            assessments,
            [
                ("Compare both obligations.", "Compare both obligations.", 1),
                ("Compare both obligations.", "follow-up 1", 2),
            ],
        )
        self.assertEqual(len(result["retrieval_rounds"]), 3)
        self.assertEqual(len(result["hits"]), 3)

    def test_multi_round_hits_remain_within_top_n_and_include_new_evidence(self):
        def retrieve(query, **kwargs):
            self.assertIn("collection_name", kwargs)
            return [
                _hit(f"{query}-1", "First-ranked evidence"),
                _hit(f"{query}-2", "Second-ranked evidence"),
            ]

        with (
            patch.object(app, "classify_query", return_value=self._classification("HARD")),
            patch.object(app, "retrieve", side_effect=retrieve),
            patch.object(
                app,
                "assess_evidence",
                side_effect=[
                    {
                        "sufficient": False,
                        "next_query": "focused search",
                        "source": "llm",
                    },
                    {
                        "sufficient": True,
                        "next_query": "",
                        "source": "llm",
                    },
                ],
            ),
            patch.object(app, "generate_answer", return_value={"answer": "Answer"}),
        ):
            result = app.query_legal_documents(
                "Compare both obligations.", top_n_final=2
            )

        self.assertEqual(len(result["hits"]), 2)
        self.assertIn("Compare both obligations.-1", [
            hit["chunk_id"] for hit in result["hits"]
        ])
        self.assertIn("focused search-1", [
            hit["chunk_id"] for hit in result["hits"]
        ])

    def test_follow_up_rewrite_is_sent_to_retrieval(self):
        with (
            patch.object(app, "classify_query", return_value=self._classification("EASY")),
            patch.object(app, "rewrite_query", return_value="termination notice period"),
            patch.object(app, "retrieve", return_value=[_hit("c1", "Evidence")]) as retrieve,
            patch.object(app, "generate_answer", return_value={"answer": "Answer"}),
        ):
            result = app.query_legal_documents(
                "What about that?", conversation_history=[{"role": "user", "content": "termination"}]
            )

        self.assertTrue(result["success"])
        self.assertEqual(retrieve.call_args.kwargs["query"], "termination notice period")

    def test_empty_retrieval_is_explicit_and_skips_generation(self):
        with (
            patch.object(app, "classify_query", return_value=self._classification("EASY")),
            patch.object(app, "rewrite_query", return_value="unanswerable question"),
            patch.object(app, "retrieve", return_value=[]),
            patch.object(app, "generate_answer") as generate,
        ):
            result = app.query_legal_documents("unanswerable question")

        self.assertTrue(result["success"])
        self.assertEqual(result["retrieval_status"], "no_results")
        self.assertEqual(result["hits"], [])
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
