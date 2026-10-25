import unittest
from unittest.mock import patch

import app


class QueryPipelineTests(unittest.TestCase):
    def test_follow_up_rewrites_retrieval_query_but_generates_for_original_question(self):
        hits = [{"chunk_id": "contract_CHUNK_1", "text": "Termination clause."}]
        answer = {"answer": "The agreement permits termination.", "used_chunk_ids": []}
        history = [{"role": "assistant", "content": "The agreement has a fixed term."}]

        with patch.object(app, "looks_like_follow_up", return_value=True), patch.object(
            app,
            "rewrite_query",
            return_value="termination clause agreement",
        ), patch.object(app, "retrieve", return_value=hits) as retrieve, patch.object(
            app, "generate_answer", return_value=answer
        ) as generate:
            result = app.query_legal_documents(
                "What about termination?",
                conversation_history=history,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["query"], "What about termination?")
        self.assertEqual(result["retrieval_query"], "termination clause agreement")
        self.assertEqual(retrieve.call_args.kwargs["query"], "termination clause agreement")
        self.assertEqual(generate.call_args.args[0], "What about termination?")
        self.assertEqual(generate.call_args.kwargs["conversation_history"], history)

    def test_no_results_do_not_call_answer_generation(self):
        with patch.object(app, "retrieve", return_value=[]), patch.object(
            app, "generate_answer"
        ) as generate:
            result = app.query_legal_documents("What does the contract say?")

        self.assertTrue(result["success"])
        self.assertEqual(result["hits"], [])
        generate.assert_not_called()

    def test_generation_failure_logs_traceback_and_preserves_retrieved_hits(self):
        hits = [
            {
                "chunk_id": f"contract_CHUNK_{index}",
                "text": f"Evidence passage {index}.",
                "rerank_score": 1.0 / index,
            }
            for index in range(1, 6)
        ]
        with patch.object(app, "retrieve", return_value=hits) as retrieve, patch.object(
            app, "generate_answer", side_effect=ValueError("bad model response")
        ) as generate:
            with self.assertLogs(app.logger, level="ERROR") as captured:
                result = app.query_legal_documents("Summarize this agreement")

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["stage"], "generation")
        self.assertEqual(result["hits"], hits)
        retrieve.assert_called_once()
        generate.assert_called_once()
        self.assertTrue(any("Traceback" in line for line in captured.output))
        self.assertNotIn("bad model response", result["error"]["message"])

    def test_explainability_setup_failure_keeps_available_report(self):
        attribution = [{"claim": "A claim.", "status": "NEEDS_REVIEW"}]
        report = {"coverage": {"total_claims": 1}}
        with patch.object(
            app, "build_llm_verifiers", side_effect=RuntimeError("Missing GROQ_API_KEY")
        ), patch.object(
            app, "compute_claim_attribution", return_value=attribution
        ), patch.object(
            app, "generate_evidence_report", return_value=report
        ):
            with self.assertLogs(app.logger, level="ERROR"):
                result = app.build_explainability_data(
                    "A claim.", [{"chunk_id": "doc_CHUNK_1", "text": "Evidence."}]
                )

        self.assertEqual(result["claim_attributions"], attribution)
        self.assertEqual(result["evidence_report"], report)
        self.assertIn("GROQ_API_KEY", result["error"])

    def test_explainability_processing_failure_is_reported_not_raised(self):
        with patch.object(
            app, "compute_claim_attribution", side_effect=ValueError("bad attribution")
        ):
            with self.assertLogs(app.logger, level="ERROR"):
                result = app.build_explainability_data("A claim.", [])

        self.assertIsNone(result["evidence_report"])
        self.assertEqual(result["claim_attributions"], [])
        self.assertIn("ValueError", result["error"])


if __name__ == "__main__":
    unittest.main()
