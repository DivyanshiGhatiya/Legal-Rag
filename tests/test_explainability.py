import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from explainability import (
    build_retrieval_trace,
    build_llm_verifiers,
    compute_claim_attribution,
    extract_and_map_page_number,
    extract_page_reference,
    format_legal_answer,
    get_document_page_stats,
    generate_evidence_report,
    render_explainable_chunk_heatmap,
    render_claim_status_answer,
    render_explainability_ui,
    split_answer_claims,
)


class MockStreamlit:
    def __init__(self):
        self.events = []

    def markdown(self, value, **kwargs):
        self.events.append(("markdown", value))

    def dataframe(self, value, **kwargs):
        self.events.append(("dataframe", value))

    def write(self, *values):
        self.events.append(("write", values))

    def text(self, value):
        self.events.append(("text", value))

    def caption(self, value):
        self.events.append(("caption", value))

    def info(self, value):
        self.events.append(("info", value))

    def warning(self, value):
        self.events.append(("warning", value))

    def metric(self, label, value):
        self.events.append(("metric", label, value))

    def columns(self, count):
        return [self for _ in range(count)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @contextmanager
    def expander(self, label, expanded=False):
        self.events.append(("expander", label, expanded))
        yield self


class ExplainabilityTests(unittest.TestCase):
    def setUp(self):
        self.chunk_one = {
            "chunk_id": "Agreement_CHUNK_0001",
            "document_name": "agreement.pdf",
            "text": "[PAGE 4] The supplier shall protect confidential information.",
            "dense_score": 0.81,
            "bm25_score": 3.2,
            "rrf_score": 0.03,
            "rerank_score": 5.4,
        }
        self.chunk_two = {
            "chunk_id": "Agreement_CHUNK_0002",
            "document_name": "agreement.pdf",
            "text": "[PAGE 5] Confidential information must not be disclosed to outsiders.",
        }

    def test_one_chunk_support_requires_verifier_and_valid_id(self):
        verifier = lambda claim, chunks: {
            "status": "SUPPORTED",
            "supporting_chunk_ids": ["Agreement_CHUNK_0001"],
            "evidence_scores": {"Agreement_CHUNK_0001": 0.91},
        }

        result = compute_claim_attribution(
            ["The supplier protects confidential information."],
            [self.chunk_one],
            verifier,
        )[0]

        self.assertEqual(result["status"], "SUPPORTED")
        self.assertEqual(len(result["supporting_evidence"]), 1)
        self.assertEqual(result["supporting_evidence"][0]["page_number"], 4)
        self.assertEqual(result["supporting_evidence"][0]["evidence_support_score"], 0.91)

    def test_multiple_chunks_can_support_one_claim(self):
        verifier = lambda claim, chunks: {
            "status": "SUPPORTED",
            "supporting_chunk_ids": [
                "Agreement_CHUNK_0001",
                "Agreement_CHUNK_0002",
            ],
        }

        result = compute_claim_attribution(
            ["The agreement restricts disclosure."],
            [self.chunk_one, self.chunk_two],
            verifier,
        )[0]

        self.assertEqual(
            [item["chunk_id"] for item in result["supporting_evidence"]],
            ["Agreement_CHUNK_0001", "Agreement_CHUNK_0002"],
        )

    def test_unsupported_claim_is_not_attributed_to_best_overlap(self):
        result = compute_claim_attribution(
            ["The supplier shall pay a $5,000 penalty."],
            [self.chunk_one],
            lambda claim, chunks: {"status": "UNSUPPORTED", "supporting_chunk_ids": []},
        )[0]

        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertEqual(result["supporting_evidence"], [])

    def test_unrelated_retrieved_chunks_do_not_become_claim_sources(self):
        result = compute_claim_attribution(
            ["The lease grants an option to renew."],
            [{"chunk_id": "c1", "text": "The supplier shall protect confidential information."}],
        )[0]

        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertEqual(result["mapped_source"], "No source found")

    def test_similar_wording_without_verifier_needs_review(self):
        result = compute_claim_attribution(
            ["The supplier shall protect confidential information."],
            [self.chunk_one],
        )[0]

        self.assertEqual(result["status"], "NEEDS_REVIEW")
        self.assertEqual(result["supporting_evidence"], [])
        self.assertTrue(result["candidate_evidence"])

    def test_valid_citation_alone_does_not_prove_support(self):
        result = compute_claim_attribution(
            ["The supplier pays a penalty [Agreement_CHUNK_0001]."],
            [self.chunk_one],
        )[0]

        self.assertEqual(result["status"], "NEEDS_REVIEW")
        self.assertEqual(result["valid_citations"], ["Agreement_CHUNK_0001"])
        self.assertEqual(result["supporting_evidence"], [])

    def test_claim_verifier_conflict_populates_report(self):
        claim = "The supplier may disclose the information."
        attributions = compute_claim_attribution(
            [claim],
            [self.chunk_one],
            lambda text, chunks: {
                "status": "CONFLICTING",
                "conflicting_chunk_ids": ["Agreement_CHUNK_0001"],
            },
        )
        report = generate_evidence_report(
            [claim], [self.chunk_one], claim_attributions=attributions
        )

        self.assertEqual(report["contradiction_status"], "CONFLICTS_FOUND")
        self.assertEqual(report["contradictory_evidence"][0]["chunk_id"], "Agreement_CHUNK_0001")

    def test_overall_status_requires_contradiction_check_to_clear(self):
        claims = ["The supplier protects confidential information."]
        claim_verifier = lambda claim, chunks: {
            "status": "SUPPORTED",
            "supporting_chunk_ids": ["Agreement_CHUNK_0001"],
        }
        attributions = compute_claim_attribution(claims, [self.chunk_one], claim_verifier)
        unchecked = generate_evidence_report(
            claims, [self.chunk_one], claim_attributions=attributions
        )
        checked = generate_evidence_report(
            claims,
            [self.chunk_one],
            claim_attributions=attributions,
            contradiction_verifier=lambda claim, chunks: {"status": "CLEAR"},
        )

        self.assertEqual(unchecked["overall_status"], "NEEDS_REVIEW")
        self.assertEqual(unchecked["contradiction_status"], "NOT_EVALUATED")
        self.assertEqual(checked["overall_status"], "SUPPORTED")
        self.assertEqual(checked["contradiction_status"], "NO_CONTRADICTION_FOUND")

    def test_configured_llm_verifiers_use_mocked_project_client(self):
        responses = [
            '{"status":"SUPPORTED","supporting_chunk_ids":["Agreement_CHUNK_0001"],'
            '"conflicting_chunk_ids":[],"contradiction_status":"CLEAR",'
            '"rationale":"Text supports it; no direct contradiction found."}',
        ]
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            content = responses[len(calls) - 1]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        claim_verifier, contradiction_verifier = build_llm_verifiers(
            model="test-model",
            client=fake_client,
        )
        claim_result = claim_verifier("The supplier protects data.", [self.chunk_one])
        contradiction_result = contradiction_verifier(
            "The supplier protects data.", [self.chunk_one]
        )

        self.assertEqual(claim_result["status"], "SUPPORTED")
        self.assertEqual(contradiction_result["status"], "CLEAR")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "test-model")

    def test_contradiction_report_records_only_resolved_chunk_ids(self):
        claims = ["The supplier may disclose confidential information."]
        attributions = compute_claim_attribution(claims, [self.chunk_one])
        report = generate_evidence_report(
            claims,
            [self.chunk_one, self.chunk_two],
            claim_attributions=attributions,
            contradiction_verifier=lambda claim, chunks: {
                "status": "CONFLICTING",
                    "conflicting_chunk_ids": [
                        "Agreement_CHUNK_0001",
                        "Agreement_CHUNK_0002",
                        "missing_CHUNK_9",
                    ],
                "rationale": "One passage prohibits disclosure.",
            },
        )

        self.assertEqual(report["contradiction_status"], "CONFLICTS_FOUND")
        self.assertEqual(len(report["contradictory_evidence"]), 2)
        self.assertEqual(
            {item["chunk_id"] for item in report["contradictory_evidence"]},
            {"Agreement_CHUNK_0001", "Agreement_CHUNK_0002"},
        )
        self.assertEqual(report["claim_attributions"][0]["status"], "CONFLICTING")

    def test_missing_id_and_page_metadata_are_not_fabricated(self):
        chunk = {"text": "A passage with no identifiers or page metadata."}

        reference = extract_page_reference(chunk)
        trace = build_retrieval_trace([chunk])[0]

        self.assertIsNone(reference["page_number"])
        self.assertFalse(reference["verified"])
        self.assertIsNone(trace["chunk_id"])
        self.assertEqual(trace["page_provenance"], "UNAVAILABLE")

    def test_exact_and_estimated_page_provenance_are_distinct(self):
        exact = extract_page_reference({"metadata": {"page_number": 7}})
        marker = extract_page_reference({"text": "[PAGE 8] Text from page."})
        estimated = extract_page_reference({"char_offset": 4000})

        self.assertEqual(exact["provenance"], "EXACT_METADATA")
        self.assertTrue(exact["verified"])
        self.assertEqual(marker["provenance"], "EXACT_TEXT_MARKER")
        self.assertTrue(marker["verified"])
        self.assertEqual(estimated["page_number"], 3)
        self.assertFalse(estimated["verified"])
        self.assertIn("Estimated", estimated["page_label"])
        self.assertEqual(extract_and_map_page_number({"char_offset": 4000}), (3, "Estimated page 3"))
        self.assertEqual(
            get_document_page_stats([{"document_name": "x.pdf", "char_offset": 4000}]),
            {"x.pdf": []},
        )

    def test_missing_retrieval_scores_name_the_producing_components(self):
        trace = build_retrieval_trace([{"chunk_id": "c1", "text": "Evidence."}])[0]

        self.assertIsNone(trace["dense_score"])
        self.assertEqual(trace["missing_score_sources"]["dense_score"], "dense_search()")
        self.assertEqual(trace["missing_score_sources"]["bm25_score"], "bm25_search()")
        self.assertEqual(trace["missing_score_sources"]["rrf_score"], "rrf_fuse()")
        self.assertEqual(
            trace["missing_score_sources"]["rerank_score"],
            "rerank() / Cross-Encoder",
        )

    def test_retrieval_trace_preserves_producer_scores_and_final_order(self):
        chunk = {
            "chunk_id": "ranked-1",
            "document_name": "contract.pdf",
            "text": "[PAGE 2] Evidence.",
            "final_rank": 3,
            "dense_rank": 2,
            "dense_score": 0.7,
            "bm25_rank": 4,
            "bm25_score": 8.5,
            "rrf_rank": 1,
            "rrf_score": 0.04,
            "rerank_score": -1.2,
            "retrieval_stages": ["DENSE", "BM25", "RRF", "CROSS_ENCODER"],
        }

        trace = build_retrieval_trace([chunk])[0]

        self.assertEqual(trace["final_rank"], 3)
        self.assertEqual(trace["dense_score"], 0.7)
        self.assertEqual(trace["bm25_score"], 8.5)
        self.assertEqual(trace["rrf_score"], 0.04)
        self.assertEqual(trace["cross_encoder_score"], -1.2)
        self.assertEqual(trace["retrieval_stages"], chunk["retrieval_stages"])

    def test_partial_support_requires_a_resolved_supporting_chunk(self):
        result = compute_claim_attribution(
            ["The supplier shall protect confidential information."],
            [self.chunk_one],
            lambda claim, chunks: {
                "status": "PARTIALLY_SUPPORTED",
                "supporting_chunk_ids": ["Agreement_CHUNK_0001"],
            },
        )[0]

        self.assertEqual(result["status"], "PARTIALLY_SUPPORTED")
        self.assertEqual(len(result["supporting_evidence"]), 1)

    def test_claim_status_html_is_escaped(self):
        rendered = render_claim_status_answer([
            {"claim": "<script>alert(1)</script>", "status": "NEEDS_REVIEW"}
        ])

        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_answer_text_splits_into_claims(self):
        self.assertEqual(
            split_answer_claims("First claim. Second claim.\n- Third claim."),
            ["First claim.", "Second claim.", "Third claim."],
        )

    def test_format_legal_answer_compacts_markdown_spacing(self):
        answer = """Overview:\n\nThe term runs for 12 months [Agreement_CHUNK_0001].\n\n\nRequirements:\n- Written notice\n- Full payment [Agreement_CHUNK_0002]\n\n\nConclusion:\nBoth conditions apply."""

        formatted = format_legal_answer(answer)

        self.assertEqual(
            formatted,
            "### Overview\n\nThe term runs for 12 months [Agreement_CHUNK_0001].\n\n"
            "### Requirements\n\n- Written notice\n- Full payment [Agreement_CHUNK_0002]\n\n"
            "### Conclusion\n\nBoth conditions apply.",
        )

    def test_format_legal_answer_preserves_unsupported_warning_text_and_citations(self):
        answer = (
            "UNSUPPORTED  Claim: The clause applies. [Agreement_CHUNK_0001]\n"
            "\nNEEDS_REVIEW: The source does not resolve the exception."
        )

        formatted = format_legal_answer(answer)

        self.assertIn("UNSUPPORTED Claim", formatted)
        self.assertIn("[Agreement_CHUNK_0001]", formatted)
        self.assertIn("NEEDS_REVIEW", formatted)

    def test_format_legal_answer_hides_internal_status_tags_only(self):
        formatted = format_legal_answer(
            "[NEEDS_REVIEW] The clause applies. [Agreement_CHUNK_0001]"
        )

        self.assertEqual(
            formatted,
            "The clause applies. [Agreement_CHUNK_0001]",
        )

    def test_answer_claim_splitting_ignores_internal_status_tags(self):
        self.assertEqual(
            split_answer_claims(
                "[UNSUPPORTED] The contract imposes a penalty [Agreement_CHUNK_0001]."
            ),
            ["The contract imposes a penalty [Agreement_CHUNK_0001]."],
        )

    def test_wrapped_bullet_citation_covers_its_claim(self):
        claims = split_answer_claims(
            "- The supplier shall protect information\n  and not disclose it [Agreement_CHUNK_0001]\n"
            "- The buyer shall return copies [Agreement_CHUNK_0002]"
        )

        self.assertEqual(len(claims), 2)
        self.assertIn("[Agreement_CHUNK_0001]", claims[0])
        self.assertIn("[Agreement_CHUNK_0002]", claims[1])

    def test_empty_results_and_empty_claims(self):
        empty_evidence = compute_claim_attribution(["A claim."], [])[0]
        empty_report = generate_evidence_report([], [])

        self.assertEqual(empty_evidence["status"], "UNSUPPORTED")
        self.assertEqual(empty_report["overall_status"], "NOT_EVALUATED")
        self.assertEqual(empty_report["evidence_coverage"]["total_claims"], 0)
        self.assertEqual(empty_report["contradiction_status"], "NOT_EVALUATED")

    def test_verifier_failure_is_not_reported_as_support(self):
        def failing_verifier(claim, chunks):
            raise TimeoutError("mock timeout")

        result = compute_claim_attribution(
            ["The supplier protects information."],
            [self.chunk_one],
            failing_verifier,
        )[0]
        report = generate_evidence_report(
            ["The supplier protects information."],
            [self.chunk_one],
            contradiction_verifier=failing_verifier,
        )

        self.assertEqual(result["status"], "NEEDS_REVIEW")
        self.assertEqual(report["contradiction_status"], "NOT_EVALUATED")

    def test_coverage_and_citation_coverage_are_distinct(self):
        claims = [
            "The supplier protects information [Agreement_CHUNK_0001].",
            "The supplier pays a penalty.",
        ]
        verifier = lambda claim, chunks: {
            "status": "SUPPORTED" if "protects" in claim else "UNSUPPORTED",
            "supporting_chunk_ids": ["Agreement_CHUNK_0001"] if "protects" in claim else [],
        }
        attributions = compute_claim_attribution(claims, [self.chunk_one], verifier)
        report = generate_evidence_report(claims, [self.chunk_one], attributions)

        coverage = report["evidence_coverage"]
        self.assertEqual(coverage["supported_claims"], 1)
        self.assertEqual(coverage["unsupported_claims"], 1)
        self.assertEqual(coverage["sufficient_evidence_percentage"], 50.0)
        self.assertEqual(coverage["citation_coverage_percentage"], 50.0)

    def test_html_special_characters_are_escaped(self):
        rendered = render_explainable_chunk_heatmap([("<script>alert(1)</script>", 1.0)])

        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_backward_compatible_renderer_call(self):
        mock_st = MockStreamlit()

        render_explainability_ui(mock_st, "query", [], ["An unsupported claim."])

        self.assertTrue(any(event[0] == "dataframe" for event in mock_st.events))
        self.assertTrue(any("Contradiction detection was not evaluated" in str(event) for event in mock_st.events))
        self.assertTrue(
            any(
                event[0] == "markdown"
                and "4. Evidence Coverage and Verification Report" in event[1]
                for event in mock_st.events
            )
        )

    def test_renderer_can_omit_duplicate_answer(self):
        mock_st = MockStreamlit()

        render_explainability_ui(
            mock_st,
            "query",
            [],
            "The actual answer.",
            show_answer=False,
        )

        self.assertFalse(
            any(
                event[0] == "markdown"
                and ("**Answer**" in event[1] or "The actual answer." in event[1])
                for event in mock_st.events
            )
        )

    def test_renderer_uses_cached_results_without_recalling_verifiers(self):
        claims = ["The supplier protects confidential information."]
        attributions = compute_claim_attribution(
            claims,
            [self.chunk_one],
            lambda claim, chunks: {
                "status": "SUPPORTED",
                "supporting_chunk_ids": ["Agreement_CHUNK_0001"],
            },
        )
        report = generate_evidence_report(
            claims,
            [self.chunk_one],
            claim_attributions=attributions,
            contradiction_verifier=lambda claim, chunks: {"status": "CLEAR"},
        )
        mock_st = MockStreamlit()

        def should_not_run(claim, chunks):
            raise AssertionError("Cached rendering must not re-run the verifier")

        render_explainability_ui(
            mock_st,
            "query",
            [self.chunk_one],
            claims,
            claim_verifier=should_not_run,
            claim_attributions=attributions,
            evidence_report=report,
        )

        self.assertTrue(any(event[0] == "dataframe" for event in mock_st.events))


if __name__ == "__main__":
    unittest.main()
