import unittest
from contextlib import contextmanager

from explainability import render_explainability_ui
from utils.explainability_data import build_explainability_data
from utils.generation import verify_citations


def _chunk(chunk_id, text, **scores):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "document_name": "contract.pdf",
        **scores,
    }


class _StreamlitStub:
    def __init__(self):
        self.messages = []

    def markdown(self, value, **_kwargs):
        self.messages.append(str(value))

    def caption(self, value):
        self.messages.append(str(value))

    def success(self, value):
        self.messages.append(str(value))

    def warning(self, value):
        self.messages.append(str(value))

    def info(self, value):
        self.messages.append(str(value))

    @contextmanager
    def expander(self, *_args, **_kwargs):
        yield


class ExplainabilityTests(unittest.TestCase):
    def test_claim_report_resolves_multiple_citations(self):
        chunks = [
            _chunk(
                "contract_CHUNK_0001",
                "The agreement requires written notice within thirty days.",
                dense_score=0.91,
                bm25_score=4.2,
                rrf_score=0.03,
                rerank_score=2.1,
            ),
            _chunk(
                "contract_CHUNK_0002",
                "Notice must be delivered to the other party by registered mail.",
                dense_score=0.84,
                bm25_score=2.8,
                rrf_score=0.02,
                rerank_score=1.8,
            ),
        ]
        answer = (
            "Written notice is required within thirty days [contract_CHUNK_0001] "
            "and must be sent by registered mail [contract_CHUNK_0002]."
        )
        citation_report = verify_citations(answer, chunks)

        result = build_explainability_data(
            answer,
            chunks,
            citation_report=citation_report,
            retrieval_rounds=[{"evidence_sufficient": True}],
        )

        self.assertEqual(len(result["claims"]), 1)
        claim = result["claim_attributions"][0]
        self.assertEqual(
            claim["validated_citation_ids"],
            ["contract_CHUNK_0001", "contract_CHUNK_0002"],
        )
        self.assertEqual(claim["verification_status"], "citation_reference_verified")
        self.assertEqual(result["evidence_report"]["citation_coverage"], 1.0)
        self.assertTrue(result["evidence_report"]["retrieval_sufficiency"])
        self.assertEqual(
            result["evidence_report"]["contradiction_analysis"]["status"],
            "not_available",
        )

        streamlit = _StreamlitStub()
        render_explainability_ui(
            st_module=streamlit,
            query="What are the notice requirements?",
            retrieved_chunks=chunks,
            answer_claims=result["claims"],
            retrieval_metadata=chunks,
            citation_report=citation_report,
            claim_attributions=result["claim_attributions"],
            evidence_report=result["evidence_report"],
            show_answer=False,
        )
        rendered = "\n".join(streamlit.messages)
        for score in ("Dense: 0.910", "BM25: 4.200", "RRF: 0.030", "Cross-Encoder: 2.100"):
            self.assertIn(score, rendered)
        self.assertIn("Contradiction analysis was not performed", rendered)

    def test_uncited_answer_is_reported_as_unverified(self):
        chunks = [_chunk("contract_CHUNK_1", "The contract term is one year.")]
        answer = "The contract term is one year."
        citation_report = verify_citations(answer, chunks)

        result = build_explainability_data(
            answer, chunks, citation_report=citation_report
        )

        self.assertEqual(
            result["claim_attributions"][0]["verification_status"],
            "missing_citation",
        )
        self.assertEqual(result["evidence_report"]["citation_coverage"], 0.0)
        self.assertEqual(result["evidence_report"]["unverified_claims"], [answer])

    def test_partial_multi_citation_does_not_count_as_verified_coverage(self):
        chunks = [_chunk("contract_CHUNK_1", "A notice must be in writing.")]
        answer = (
            "Notice must be written [contract_CHUNK_1] and delivered promptly "
            "[missing_CHUNK_2]."
        )
        citation_report = verify_citations(answer, chunks)

        result = build_explainability_data(
            answer, chunks, citation_report=citation_report
        )

        claim = result["claim_attributions"][0]
        self.assertEqual(claim["verification_status"], "partially_verified")
        self.assertEqual(claim["validated_citation_ids"], ["contract_CHUNK_1"])
        self.assertEqual(claim["unresolved_citations"], ["missing_CHUNK_2"])
        self.assertEqual(result["evidence_report"]["citation_coverage"], 0.0)

    def test_missing_evidence_is_not_reported_as_success(self):
        result = build_explainability_data(
            "The answer is unknown.", [], citation_report=None
        )

        self.assertEqual(
            result["evidence_report"]["status"], "no_retrieved_evidence"
        )
        self.assertIsNone(result["evidence_report"]["citation_coverage"])
        self.assertFalse(
            result["evidence_report"]["citation_validation_available"]
        )


if __name__ == "__main__":
    unittest.main()
