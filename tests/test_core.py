import numpy as np
import pytest

from dmf import DMF


def sine_with_outliers():
    """The example of the original Spark notebook: a sine wave with 3 spikes."""
    values = np.sin(np.arange(50))
    values[[10, 20, 30]] = [5, 19, 20]
    return values.reshape(-1, 1)


def test_detects_spikes_in_sine_wave():
    model = DMF(dissimilarity="euclidean", filtering="zscore", alpha=0).fit(sine_with_outliers())
    np.testing.assert_array_equal(model.outlier_indices(), [10, 20, 30])


def test_lower_alpha_flags_more():
    data = sine_with_outliers()
    strict = DMF(alpha=1).fit_predict(data).sum()
    loose = DMF(alpha=-0.5).fit_predict(data).sum()
    assert loose > strict


def test_threshold_filtering_uses_raw_scores():
    model = DMF(filtering="threshold", alpha=100).fit([[0], [1], [2], [50]])
    np.testing.assert_array_equal(model.outlier_indices(), [3])
    assert not hasattr(model, "zscores_")


def test_dtw_on_series_of_different_lengths():
    t = np.linspace(0, 2 * np.pi, 40)
    series = [np.sin(t), np.sin(t[:35]), np.sin(t[2:]), np.cos(t) * 3]
    assert list(DMF(dissimilarity="dtw", alpha=1).fit(series).outlier_indices()) == [3]


def test_hamming_on_categorical_rows():
    rows = [["a", "b", "c"]] * 5 + [["x", "y", "z"]]
    assert list(DMF(dissimilarity="hamming", alpha=1).fit(rows).outlier_indices()) == [5]


def test_all_equal_examples_have_no_outliers():
    model = DMF().fit([[1, 1]] * 4)
    assert not model.outliers_.any()
    np.testing.assert_array_equal(model.zscores_, 0)


def test_custom_mapping():
    model = DMF(mapping=lambda m: m.max(axis=1), filtering="threshold", alpha=5).fit([[0], [1], [10]])
    np.testing.assert_array_equal(model.scores_, [10, 9, 10])


@pytest.mark.parametrize(
    "kwargs, message",
    [({"filtering": "median"}, "unknown filtering"), ({"mapping": "mean"}, "unknown mapping")],
)
def test_bad_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DMF(**kwargs).fit([[0], [1]])


def test_needs_two_examples():
    with pytest.raises(ValueError):
        DMF().fit([[1]])
