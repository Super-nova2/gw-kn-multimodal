from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[2] / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.run_fixed_checkpoint_attribution import (  # noqa: E402
    GWInputTransform,
    build_kn_nuisance_matched_galleries,
    build_kn_random_same_source_galleries,
    effective_physical_parameters,
    flux_sigma_to_luptitude,
    gallery_identity_digest,
    luptitude_to_flux_sigma,
    minimum_cost_derangement,
    normalize_condition,
    physical_mismatch,
    source_conditional_derangement,
    transform_optical_bank,
    validate_kn_matched_galleries,
)


class LuptitudeAblationTests(unittest.TestCase):
    def test_luptitude_flux_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        flux = rng.normal(50.0, 150.0, size=(3, 5, 6))
        sigma = rng.uniform(1.0, 20.0, size=flux.shape)

        values, errors = flux_sigma_to_luptitude(flux, sigma)
        restored_flux, restored_sigma = luptitude_to_flux_sigma(values, errors)

        np.testing.assert_allclose(restored_flux, flux, rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(restored_sigma, sigma, rtol=1e-11, atol=1e-10)

    @staticmethod
    def _bank() -> dict[str, torch.Tensor]:
        flux = np.asarray(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                    [4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
                    [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
                ],
                [
                    [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
                    [8.0, 4.0, 16.0, 12.0, 24.0, 20.0],
                    [4.0, 2.0, 8.0, 6.0, 12.0, 10.0],
                    [1.0, 0.5, 2.0, 1.5, 3.0, 2.5],
                ],
            ]
        )
        sigma = np.maximum(np.abs(flux) / 5.0, 0.1)
        values, errors = flux_sigma_to_luptitude(flux, sigma)
        return {
            "times": torch.tensor(
                [[-0.1, 0.0, 0.1, 0.2], [-0.1, 0.0, 0.1, 0.2]],
                dtype=torch.float32,
            ),
            "values": torch.tensor(values, dtype=torch.float32),
            "masks": torch.ones((2, 4, 6), dtype=torch.float32),
            "errors": torch.tensor(errors, dtype=torch.float32),
        }

    def test_brightness_normalization_removes_amplitude_and_preserves_snr(self) -> None:
        bank = self._bank()
        before_flux, before_sigma = luptitude_to_flux_sigma(
            bank["values"].numpy(), bank["errors"].numpy()
        )

        transformed, audit = transform_optical_bank(
            bank, "brightness_norm", seed=42, amplitude_quantile=1.0
        )
        after_flux, after_sigma = luptitude_to_flux_sigma(
            transformed["values"].numpy(), transformed["errors"].numpy()
        )

        np.testing.assert_allclose(
            np.max(np.abs(after_flux), axis=(1, 2)), np.full(2, 100.0), rtol=2e-5
        )
        np.testing.assert_allclose(
            after_flux / after_sigma, before_flux / before_sigma, rtol=2e-4
        )
        self.assertEqual(len(audit), 2)

    def test_per_band_normalization_removes_color_amplitudes(self) -> None:
        transformed, _ = transform_optical_bank(
            self._bank(), "per_band_norm", seed=42, amplitude_quantile=1.0
        )
        flux, _ = luptitude_to_flux_sigma(
            transformed["values"].numpy(), transformed["errors"].numpy()
        )
        np.testing.assert_allclose(
            np.max(np.abs(flux), axis=1), np.full((2, 6), 100.0), rtol=2e-5
        )

    def test_time_shuffle_is_deterministic_and_peak_align_places_peak_at_zero(self) -> None:
        bank = self._bank()
        shuffled_a, _ = transform_optical_bank(bank, "time_shuffle", seed=123)
        shuffled_b, _ = transform_optical_bank(bank, "time_shuffle", seed=123)
        np.testing.assert_array_equal(shuffled_a["times"], shuffled_b["times"])
        self.assertFalse(torch.equal(shuffled_a["times"], bank["times"]))

        aligned, audit = transform_optical_bank(bank, "peak_align", seed=123)
        self.assertAlmostEqual(float(aligned["times"][0, 2]), 0.0, places=6)
        self.assertAlmostEqual(float(aligned["times"][1, 1]), 0.0, places=6)
        self.assertAlmostEqual(audit[0]["peak_time_shift_days"], 0.1, places=6)
        self.assertAlmostEqual(audit[1]["peak_time_shift_days"], 0.0, places=6)

    def test_time_shuffle_is_invariant_to_streaming_chunks(self) -> None:
        bank = self._bank()
        full, _ = transform_optical_bank(
            bank, "time_shuffle", seed=17, item_indices=[100, 101]
        )
        chunk_times = []
        for idx, item_index in enumerate((100, 101)):
            chunk = {
                key: value[idx : idx + 1]
                for key, value in bank.items()
            }
            transformed, audit = transform_optical_bank(
                chunk,
                "time_shuffle",
                seed=17,
                item_indices=[item_index],
            )
            chunk_times.append(transformed["times"])
            self.assertEqual(audit[0]["item_index"], item_index)
        np.testing.assert_array_equal(full["times"], torch.cat(chunk_times))


class GWTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gw_ids = np.asarray([10, 11, 20, 21])
        self.sources = ["bns", "bns", "nsbh", "nsbh"]
        self.scalars = np.asarray(
            [
                [1.5, 1.2, 0.01, 0.02, -0.2, 0.1, 0.01],
                [1.7, 1.3, 0.03, 0.04, 0.6, 0.2, 0.02],
                [6.0, 1.4, -0.5, 0.01, -0.8, 0.3, 0.03],
                [7.0, 1.5, 0.7, 0.02, 0.4, 0.4, 0.04],
            ],
            dtype=np.float32,
        )
        self.skymaps = np.zeros((4, 7, 3), dtype=np.float32)
        self.skymaps[:, 4] = np.asarray([1.0, 2.0, 3.0])
        for idx in range(4):
            self.skymaps[idx, 5] = idx + np.asarray([1.0, 2.0, 3.0])
            self.skymaps[idx, 6] = 10 + idx + np.asarray([1.0, 2.0, 3.0])

    def test_minimum_cost_and_source_conditional_derangements_have_no_fixed_points(self) -> None:
        features = np.arange(8, dtype=np.float64).reshape(4, 2)
        order = minimum_cost_derangement(features)
        self.assertEqual(np.unique(order).size, 4)
        self.assertFalse(np.any(order == np.arange(4)))

        donors = source_conditional_derangement(
            self.gw_ids, self.sources, features
        )
        self.assertEqual(donors, {10: 11, 11: 10, 20: 21, 21: 20})

    def test_sign_flip_is_exact_and_distance_permutation_is_joint(self) -> None:
        sign = GWInputTransform(
            "inclination_sign_flip",
            self.gw_ids,
            self.scalars,
            None,
            self.sources,
        )
        scalar_after, _ = sign(10, self.scalars[0], self.skymaps[0])
        self.assertAlmostEqual(float(scalar_after[4]), 0.2)
        self.assertGreater(sign.audit_rows[0]["scalar_delta_l1"], 0.0)

        distance = GWInputTransform(
            "distance_perm",
            self.gw_ids,
            self.scalars,
            self.skymaps,
            self.sources,
        )
        scalar_after, sky_after = distance(10, self.scalars[0], self.skymaps[0])
        np.testing.assert_array_equal(scalar_after[5:7], self.scalars[1, 5:7])
        np.testing.assert_array_equal(sky_after[5:7], self.skymaps[1, 5:7])
        np.testing.assert_array_equal(sky_after[:5], self.skymaps[0, :5])
        self.assertGreater(
            distance.audit_rows[0]["skymap_distance_delta_mean_abs"], 0.0
        )

        distance_channels = GWInputTransform(
            "distance_perm",
            self.gw_ids,
            self.scalars,
            self.skymaps[:, 5:7],
            self.sources,
        )
        _, compact_sky_after = distance_channels(
            10, self.scalars[0], self.skymaps[0]
        )
        np.testing.assert_array_equal(
            compact_sky_after[5:7], self.skymaps[1, 5:7]
        )

        with self.assertRaisesRegex(ValueError, "distance_perm requires"):
            GWInputTransform(
                "distance_perm", self.gw_ids, self.scalars, None, self.sources
            )

    def test_skymap_distance_flat_keeps_probability_and_global_distance(self) -> None:
        transform = GWInputTransform(
            "skymap_distance_flat",
            self.gw_ids,
            self.scalars,
            self.skymaps,
            self.sources,
        )
        _, sky_after = transform(10, self.scalars[0], self.skymaps[0])
        np.testing.assert_array_equal(sky_after[:5], self.skymaps[0, :5])
        self.assertEqual(np.unique(sky_after[5]).size, 1)
        self.assertEqual(np.unique(sky_after[6]).size, 1)


class MatchedGalleryTests(unittest.TestCase):
    def test_matched_gallery_is_deterministic_leak_free_and_prefix_nested(self) -> None:
        positive = {idx: np.asarray([idx], dtype=np.int64) for idx in range(4)}
        parent = np.arange(4, dtype=np.int64)
        source = ["bns"] * 4
        nuisance = np.asarray(
            [
                [0.0, 0.0, 1.0, 1.0, 0.1],
                [0.1, 0.1, 1.1, 1.0, 0.2],
                [0.2, 0.2, 1.2, 1.0, 0.3],
                [0.3, 0.3, 1.3, 1.0, 0.4],
            ]
        )
        galleries_a, used_a = build_kn_nuisance_matched_galleries(
            gw_positive_indices=positive,
            optical_parent_gw_idx=parent,
            gw_source_types=source,
            nuisance_features=nuisance,
            gallery_sizes=[2, 3],
            n_trials=2,
            seed=42,
        )
        galleries_b, used_b = build_kn_nuisance_matched_galleries(
            gw_positive_indices=positive,
            optical_parent_gw_idx=parent,
            gw_source_types=source,
            nuisance_features=nuisance,
            gallery_sizes=[2, 3],
            n_trials=2,
            seed=42,
        )

        validate_kn_matched_galleries(galleries_a, parent)
        self.assertEqual(used_a, used_b)
        self.assertEqual(
            gallery_identity_digest(galleries_a), gallery_identity_digest(galleries_b)
        )
        small = galleries_a[(2, 0, 0)]["negative_indices"]
        large = galleries_a[(3, 0, 0)]["negative_indices"]
        np.testing.assert_array_equal(small, large[:1])

    def test_random_same_source_gallery_reuses_positives_and_is_leak_free(self) -> None:
        positive = {idx: np.asarray([idx], dtype=np.int64) for idx in range(6)}
        parent = np.arange(6, dtype=np.int64)
        source = ["bns"] * 3 + ["nsbh"] * 3
        nuisance = np.asarray(
            [[idx, idx / 2, 1.0, 1.0, idx / 10] for idx in range(6)],
            dtype=np.float64,
        )
        nearest, _ = build_kn_nuisance_matched_galleries(
            gw_positive_indices=positive,
            optical_parent_gw_idx=parent,
            gw_source_types=source,
            nuisance_features=nuisance,
            gallery_sizes=[2, 3],
            n_trials=2,
            seed=42,
        )
        random_a, used_a = build_kn_random_same_source_galleries(
            gw_positive_indices=positive,
            optical_parent_gw_idx=parent,
            gw_source_types=source,
            nuisance_features=nuisance,
            gallery_sizes=[2, 3],
            n_trials=2,
            seed=42,
        )
        random_b, used_b = build_kn_random_same_source_galleries(
            gw_positive_indices=positive,
            optical_parent_gw_idx=parent,
            gw_source_types=source,
            nuisance_features=nuisance,
            gallery_sizes=[2, 3],
            n_trials=2,
            seed=42,
        )

        self.assertEqual(used_a, used_b)
        self.assertEqual(
            gallery_identity_digest(random_a), gallery_identity_digest(random_b)
        )
        validate_kn_matched_galleries(random_a, parent)
        for key, spec in random_a.items():
            self.assertEqual(spec["positive_index"], nearest[key]["positive_index"])
            self.assertTrue(
                np.all(np.asarray(source)[parent[spec["negative_indices"]]] == source[key[2]])
            )

    def test_template_effective_parameters_clip_and_use_absolute_inclination(self) -> None:
        query = effective_physical_parameters(
            "bns", {"mej_dyn": 0.1, "mej_wind": 0.001, "costheta": -0.3}
        )
        candidate = effective_physical_parameters(
            "bns", {"mej_dyn": 0.02, "mej_wind": 0.13, "costheta": 0.3}
        )
        self.assertEqual(query["effective_mej_dyn"], 0.02)
        self.assertEqual(query["effective_mej_wind"], 0.01)
        mismatch = physical_mismatch(query, candidate)
        self.assertAlmostEqual(mismatch["delta_abs_costheta"], 0.0)


class ConditionValidationTests(unittest.TestCase):
    def test_unknown_condition_field_and_transform_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported condition fields"):
            normalize_condition({"name": "x", "typo": True})
        with self.assertRaisesRegex(ValueError, "gw_transform"):
            normalize_condition({"name": "x", "gw_transform": "random_noise"})


if __name__ == "__main__":
    unittest.main()
