import numpy as np

import sky_sampling


def test_fixed_distance_samples_keep_parent_distance(monkeypatch):
    monkeypatch.setattr(
        sky_sampling,
        "sample_posterior_3d",
        lambda *args, **kwargs: (
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([-1.0, -2.0, -3.0]),
            np.asarray([10.0, 20.0, 30.0]),
            np.asarray([0.5, 0.3, 0.2]),
        ),
    )

    ra, dec, distance, probability, is_true = (
        sky_sampling.build_fixed_distance_coordinate_samples(
            object(),
            distance_mpc=700.0,
            samples_per_event=3,
            seed=42,
        )
    )

    np.testing.assert_array_equal(ra, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dec, [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(distance, [700.0, 700.0, 700.0])
    np.testing.assert_array_equal(probability, [0.5, 0.3, 0.2])
    np.testing.assert_array_equal(is_true, [False, False, False])


def test_truth_fixed_distance_samples_prepend_truth_and_request_unique_pixels(
    monkeypatch,
):
    captured = {}

    def fake_sample(_mocmap, n_samples, **kwargs):
        captured.update(n_samples=n_samples, **kwargs)
        return (
            np.asarray([1.0, 2.0]),
            np.asarray([-1.0, -2.0]),
            np.asarray([30.0, 40.0]),
            np.asarray([0.6, 0.4]),
        )

    monkeypatch.setattr(sky_sampling, "sample_posterior_3d", fake_sample)
    ra, dec, distance, probability, is_true = (
        sky_sampling.build_fixed_distance_truth_coordinate_samples(
            object(),
            true_ra=197.45,
            true_dec=-23.38,
            distance_mpc=40.7,
            samples_per_event=3,
            seed=170817,
        )
    )

    assert captured["n_samples"] == 2
    assert captured["replace"] is False
    np.testing.assert_allclose(ra, [197.45, 1.0, 2.0])
    np.testing.assert_allclose(dec, [-23.38, -1.0, -2.0])
    np.testing.assert_allclose(distance, [40.7, 40.7, 40.7])
    assert np.isnan(probability[0])
    np.testing.assert_array_equal(is_true, [True, False, False])
