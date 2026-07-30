from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
SCRIPT_DIR = MODEL_DIR / "scripts" / "eval"
for path in (MODEL_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

comparison = importlib.import_module("scripts.eval.eval_retrieval_comparison")


class CandidateTimeDeltaModeTests(unittest.TestCase):
    def test_zero_mode_replaces_available_or_missing_values(self) -> None:
        available = comparison.apply_candidate_time_delta_mode(
            np.asarray([0.4, 12.0], dtype=np.float32),
            mode="zero",
            n_candidates=2,
        )
        missing = comparison.apply_candidate_time_delta_mode(
            None,
            mode="zero",
            n_candidates=3,
        )

        np.testing.assert_array_equal(available, np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(missing, np.zeros(3, dtype=np.float32))

    def test_native_mode_is_an_identity_operation(self) -> None:
        original = np.asarray([1.25, 9.5], dtype=np.float32)
        result = comparison.apply_candidate_time_delta_mode(
            original,
            mode="native",
            n_candidates=2,
        )

        self.assertIs(result, original)

    def test_positive_shared_mode_repeats_positive_delay(self) -> None:
        result = comparison.apply_candidate_time_delta_mode(
            np.asarray([1.0, 2.0], dtype=np.float32),
            mode="positive_shared",
            n_candidates=2,
            shared_value=0.75,
        )

        np.testing.assert_array_equal(
            result, np.asarray([0.75, 0.75], dtype=np.float32)
        )

    def test_positive_shared_requires_finite_shared_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "shared_value"):
            comparison.apply_candidate_time_delta_mode(
                None,
                mode="positive_shared",
                n_candidates=2,
            )

    def test_coordinate_mode_fails_closed(self) -> None:
        self.assertEqual(
            comparison.normalize_candidate_coordinate_mode("positive_shared"),
            "positive_shared",
        )
        with self.assertRaisesRegex(ValueError, "candidate_coordinate_mode"):
            comparison.normalize_candidate_coordinate_mode("random")

    def test_synthetic_coordinate_rejects_stale_optical_embedding(self) -> None:
        with self.assertRaisesRegex(ValueError, "original coordinate"):
            comparison._candidate_z_l_for_chunk(
                object(),
                {"z_l_cls": torch.zeros((1, 2))},
                torch.tensor([0]),
                torch.zeros((1, 2)),
                torch.device("cpu"),
                use_synthetic_coords=True,
            )

    def test_invalid_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_time_delta_mode"):
            comparison.normalize_candidate_time_delta_mode("shuffle")

    def test_dispatch_forwards_mode_only_to_logit_scorer(self) -> None:
        self.assertEqual(
            comparison.candidate_time_delta_scoring_kwargs("logits", "zero"),
            {"candidate_time_delta_mode": "zero"},
        )
        self.assertEqual(
            comparison.candidate_time_delta_scoring_kwargs("contrastive", "zero"),
            {},
        )

    def test_zero_mode_reaches_positive_and_negative_fusion_scoring(self) -> None:
        galleries = {
            (3, 0, 7): {
                "positive_index": 0,
                "negative_indices": np.asarray([2, 4], dtype=np.int64),
            }
        }
        captured = []

        def capture_scores(*args, **kwargs):
            captured.append(np.asarray(kwargs["candidate_abs_dt_days"]).copy())
            candidate_indices = np.asarray(args[2])
            return np.zeros(candidate_indices.shape[0], dtype=np.float32)

        with (
            mock.patch.object(
                comparison,
                "_build_gallery_query_cache",
                return_value={7: {}},
            ),
            mock.patch.object(
                comparison,
                "_model_requires_time_delta",
                return_value=True,
            ),
            mock.patch.object(
                comparison,
                "extract_gallery_negative_abs_dt_days",
                return_value=np.asarray([7.0, 9.0], dtype=np.float32),
            ),
            mock.patch.object(
                comparison,
                "_score_candidate_bank_with_logits",
                side_effect=capture_scores,
            ),
        ):
            comparison.score_all_galleries_multimodal(
                model=object(),
                positive_bank={
                    "dual_fusion": False,
                    "first_detection_mjd": torch.tensor([102.0]),
                },
                negative_bank={},
                galleries=galleries,
                unique_gw=[7],
                test_data_path="unused.h5",
                device=torch.device("cpu"),
                model_args={},
                gw_event_time_mjd_table=np.asarray(
                    [np.nan] * 7 + [100.0], dtype=np.float64
                ),
                candidate_time_delta_mode="zero",
            )

        self.assertEqual(len(captured), 2)
        np.testing.assert_array_equal(captured[0], np.zeros(1, dtype=np.float32))
        np.testing.assert_array_equal(captured[1], np.zeros(2, dtype=np.float32))

    def test_zero_mode_rejects_runtime_model_without_dt_input(self) -> None:
        with mock.patch.object(
            comparison,
            "_model_requires_time_delta",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "does not consume"):
                comparison.score_all_galleries_multimodal(
                    model=object(),
                    positive_bank={},
                    negative_bank={},
                    galleries={},
                    unique_gw=[],
                    test_data_path="unused.h5",
                    device=torch.device("cpu"),
                    model_args={},
                    candidate_time_delta_mode="zero",
                )

    def test_positive_shared_time_and_coordinate_reach_every_candidate(self) -> None:
        galleries = {
            (3, 0, 7): {
                "positive_index": 0,
                "negative_indices": np.asarray([2, 4], dtype=np.int64),
            }
        }
        captured = []

        def capture_scores(*args, **kwargs):
            captured.append(
                {
                    "dt": np.asarray(kwargs["candidate_abs_dt_days"]).copy(),
                    "coords": np.asarray(kwargs["candidate_coords"]).copy(),
                }
            )
            return np.zeros(np.asarray(args[2]).shape[0], dtype=np.float32)

        with (
            mock.patch.object(
                comparison, "_build_gallery_query_cache", return_value={7: {}}
            ),
            mock.patch.object(
                comparison, "_model_requires_time_delta", return_value=True
            ),
            mock.patch.object(
                comparison,
                "extract_gallery_negative_abs_dt_days",
                return_value=np.asarray([7.0, 9.0], dtype=np.float32),
            ),
            mock.patch.object(
                comparison,
                "_score_candidate_bank_with_logits",
                side_effect=capture_scores,
            ),
        ):
            comparison.score_all_galleries_multimodal(
                model=object(),
                positive_bank={
                    "dual_fusion": False,
                    "first_detection_mjd": torch.tensor([102.0]),
                    "opt_coords": torch.tensor([[1.25, -0.5]]),
                },
                negative_bank={},
                galleries=galleries,
                unique_gw=[7],
                test_data_path="unused.h5",
                device=torch.device("cpu"),
                model_args={},
                gw_event_time_mjd_table=np.asarray(
                    [np.nan] * 7 + [100.0], dtype=np.float64
                ),
                candidate_time_delta_mode="positive_shared",
                candidate_coordinate_mode="positive_shared",
            )

        np.testing.assert_array_equal(captured[0]["dt"], [2.0])
        np.testing.assert_array_equal(captured[1]["dt"], [2.0, 2.0])
        np.testing.assert_array_equal(captured[0]["coords"], [[1.25, -0.5]])
        np.testing.assert_array_equal(
            captured[1]["coords"], [[1.25, -0.5], [1.25, -0.5]]
        )


class TimeDeltaComparisonArtifactTests(unittest.TestCase):
    def test_metric_deltas_include_overall_and_source(self) -> None:
        baseline = {
            "retrieval": {"gallery_100_recall_at_1": 0.8},
            "retrieval_by_source": {
                "bns": {"gallery_100_recall_at_1": 0.9}
            },
        }
        ablated = {
            "retrieval": {"gallery_100_recall_at_1": 0.5},
            "retrieval_by_source": {
                "bns": {"gallery_100_recall_at_1": 0.6}
            },
        }

        rows = comparison.build_time_delta_metric_delta_rows(baseline, ablated)

        self.assertEqual({row["scope"] for row in rows}, {"overall", "bns"})
        overall = next(row for row in rows if row["scope"] == "overall")
        self.assertAlmostEqual(overall["absolute_change"], -0.3)
        self.assertAlmostEqual(overall["relative_change"], -0.375)

    def test_gallery_identity_excludes_time_but_detects_candidate_changes(self) -> None:
        galleries = {
            (10, 0, 1): {
                "positive_index": 3,
                "negative_indices": np.asarray([4, 5]),
                "negative_abs_dt_days": np.asarray([1.0, 20.0]),
                "requested_gallery_size": 10,
                "actual_gallery_size": 3,
            }
        }
        changed_time = {
            key: {**value, "negative_abs_dt_days": np.zeros(2)}
            for key, value in galleries.items()
        }
        changed_candidate = {
            key: {**value, "negative_indices": np.asarray([4, 6])}
            for key, value in galleries.items()
        }

        original_hash = comparison.summarize_gallery_identity(galleries)["sha256"]
        self.assertEqual(
            original_hash,
            comparison.summarize_gallery_identity(changed_time)["sha256"],
        )
        self.assertNotEqual(
            original_hash,
            comparison.summarize_gallery_identity(changed_candidate)["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
