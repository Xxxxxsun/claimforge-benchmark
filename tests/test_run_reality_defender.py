import tempfile
import unittest
from pathlib import Path

from eval.commercial.run_illuminarty import ImageItem, append_jsonl
from eval.commercial.run_reality_defender import parse_result, write_summary


class ParseRealityDefenderResultTest(unittest.TestCase):
    def test_parses_terminal_authentic_result_and_strips_account_fields(self):
        parsed = parse_result(
            {
                "requestId": "request-1",
                "userId": "must-not-be-persisted",
                "institutionId": "must-not-be-persisted",
                "releaseVersion": "2.3.1",
                "mediaType": "IMAGE",
                "resultsSummary": {
                    "status": "AUTHENTIC",
                    "metadata": {"finalScore": 12.5},
                },
                "models": [
                    {
                        "name": "image-model",
                        "status": "AUTHENTIC",
                        "predictionNumber": 0.125,
                        "privateField": "drop-me",
                    }
                ],
            }
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["provider_status"], "AUTHENTIC")
        self.assertEqual(parsed["provider_score"], 0.125)
        self.assertTrue(parsed["applicable"])
        self.assertNotIn("userId", parsed["provider_response"])
        self.assertNotIn("institutionId", parsed["provider_response"])
        self.assertEqual(
            parsed["models"],
            [
                {
                    "name": "image-model",
                    "status": "AUTHENTIC",
                    "prediction_number": 0.125,
                }
            ],
        )

    def test_preserves_not_applicable_reason_without_treating_it_as_failure(self):
        parsed = parse_result(
            {
                "requestId": "request-2",
                "resultsSummary": {
                    "status": "NOT_APPLICABLE",
                    "metadata": {
                        "reasons": [
                            {
                                "code": "relevance",
                                "message": "no faces detected/faces too small",
                            }
                        ]
                    },
                },
                "models": None,
            }
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertFalse(parsed["applicable"])
        self.assertEqual(parsed["provider_status"], "NOT_APPLICABLE")
        self.assertEqual(parsed["provider_score"], None)
        self.assertEqual(parsed["not_applicable_reasons"][0]["code"], "relevance")

    def test_nonterminal_result_is_not_parsed_as_complete(self):
        self.assertIsNone(
            parse_result(
                {
                    "requestId": "request-3",
                    "resultsSummary": {
                        "status": "ANALYZING",
                        "metadata": {},
                    },
                }
            )
        )


class RealityDefenderSummaryTest(unittest.TestCase):
    def test_reports_coverage_separately_from_terminal_responses(self):
        items = [
            ImageItem(
                id=f"task-1__{kind}",
                task_id="task-1",
                domain="lodging",
                kind=kind,
                label=label,
                path=Path(f"/{kind}.jpg"),
                relative_path=f"{kind}.jpg",
                image_size=None,
                sha256=kind,
                file_bytes=1,
            )
            for kind, label in (("real", "not_edited"), ("forged", "edited"))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results.jsonl"
            append_jsonl(
                output,
                {
                    "id": "task-1__real",
                    "task_id": "task-1",
                    "kind": "real",
                    "status": "ok",
                    "provider_status": "AUTHENTIC",
                    "provider_score": 0.1,
                    "applicable": True,
                },
            )
            append_jsonl(
                output,
                {
                    "id": "task-1__forged",
                    "task_id": "task-1",
                    "kind": "forged",
                    "status": "ok",
                    "provider_status": "NOT_APPLICABLE",
                    "provider_score": None,
                    "applicable": False,
                    "not_applicable_reasons": [
                        {
                            "code": "relevance",
                            "message": "no faces detected/faces too small",
                        }
                    ],
                },
            )

            summary = write_summary(output, items, "manifest-hash", "both")

        self.assertEqual(summary["valid_terminal_images"], 2)
        self.assertEqual(summary["applicable_images"], 1)
        self.assertEqual(summary["applicable_coverage"], 0.5)
        self.assertEqual(summary["coverage_by_kind"]["forged"]["applicable"], 0)
        self.assertEqual(summary["paired_score_delta"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
