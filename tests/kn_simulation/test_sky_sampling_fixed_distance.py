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
