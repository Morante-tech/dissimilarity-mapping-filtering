import numpy as np
import pytest

from dmf import dtw, euclidean, hamming, pairwise


def test_euclidean():
    assert euclidean([0, 0], [3, 4]) == 5.0


def test_hamming_compares_element_by_element():
    assert hamming(["a", "b", "c"], ["a", "x", "y"]) == 2
    assert hamming([1, 2, 3], [1, 2, 3]) == 0


def test_hamming_rejects_different_lengths():
    with pytest.raises(ValueError):
        hamming([1, 2], [1, 2, 3])


def test_dtw_identical_and_shifted_sequences():
    a = [0, 1, 2, 1, 0]
    assert dtw(a, a) == 0.0
    # a delayed copy of the same shape aligns perfectly
    assert dtw(a, [0, 0, 1, 2, 1, 0]) == 0.0


def test_dtw_known_value():
    # optimal alignment: 1-1, 2-2, 3-2 (cost 1), 3-4 (cost 1)
    assert dtw([1, 2, 3], [1, 2, 4]) == 1.0


def test_dtw_multivariate_and_path():
    a = np.array([[0, 0], [1, 1], [2, 2]])
    b = np.array([[0, 0], [2, 2]])
    distance, cost, (path_a, path_b) = dtw(a, b, return_path=True)
    assert cost.shape == (3, 2)
    assert path_a[0] == 0 and path_b[0] == 0
    assert path_a[-1] == 2 and path_b[-1] == 1
    assert distance == pytest.approx(cost[path_a, path_b].sum())


def test_pairwise_is_symmetric_with_zero_diagonal():
    matrix = pairwise([[0], [1], [3]])
    np.testing.assert_array_equal(matrix, [[0, 1, 3], [1, 0, 2], [3, 2, 0]])


def test_pairwise_accepts_custom_function():
    matrix = pairwise([1, 2, 4], lambda a, b: abs(a - b) ** 2)
    assert matrix[0, 2] == 9


def test_unknown_measure():
    with pytest.raises(ValueError, match="unknown dissimilarity"):
        pairwise([[0], [1]], "manhattan")
