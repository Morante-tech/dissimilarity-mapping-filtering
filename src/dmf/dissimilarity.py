"""Dissimilarity measures between examples.

Every measure takes two examples and returns a non-negative number: 0 when the
examples are identical, larger when they are more different.
"""

import numpy as np


def euclidean(a, b):
    """Euclidean distance between two vectors of the same length."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def hamming(a, b):
    """Number of positions where two sequences of the same length differ.

    Suitable for categorical data.
    """
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    if a.shape != b.shape:
        raise ValueError(f"hamming needs sequences of equal shape, got {a.shape} and {b.shape}")
    return int(np.count_nonzero(a != b))


def dtw(a, b, return_path=False):
    """Dynamic Time Warping distance between two sequences.

    The sequences may have different lengths. Each element can be a scalar or a
    vector (all vectors with the same number of components). The local cost
    between two elements is their L1 distance, and the returned distance is the
    accumulated cost of the optimal alignment.

    :param a: first sequence, shape (n,) or (n, d)
    :param b: second sequence, shape (m,) or (m, d)
    :param return_path: if True, also return the optimal alignment as two
        arrays of indices (into ``a`` and into ``b``)
    :return: ``distance`` or ``(distance, cost_matrix, (path_a, path_b))``
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dtw needs elements with the same dimension, got {a.shape[1]} and {b.shape[1]}")

    n, m = len(a), len(b)
    # local cost between every pair of elements
    cost = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)

    # acc[i + 1, j + 1] is the cheapest way to align a[:i + 1] with b[:j + 1]
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row_prev = acc[i - 1]
        row = acc[i]
        cost_row = cost[i - 1]
        for j in range(1, m + 1):
            row[j] = cost_row[j - 1] + min(row_prev[j - 1], row_prev[j], row[j - 1])

    distance = float(acc[n, m])
    if not return_path:
        return distance
    return distance, cost, _backtrack(acc)


def _backtrack(acc):
    """Recover the optimal alignment from an accumulated cost matrix."""
    i, j = acc.shape[0] - 1, acc.shape[1] - 1
    path_a, path_b = [i - 1], [j - 1]
    while i > 1 or j > 1:
        steps = [(acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1)]
        _, i, j = min(steps, key=lambda step: step[0])
        path_a.append(i - 1)
        path_b.append(j - 1)
    return np.array(path_a[::-1]), np.array(path_b[::-1])


MEASURES = {"euclidean": euclidean, "hamming": hamming, "dtw": dtw}


def get_measure(name_or_function):
    """Return a dissimilarity function from its name, or the function itself."""
    if callable(name_or_function):
        return name_or_function
    try:
        return MEASURES[name_or_function]
    except KeyError:
        raise ValueError(
            f"unknown dissimilarity {name_or_function!r}; use one of {sorted(MEASURES)} or a function"
        ) from None


def pairwise(examples, measure="euclidean"):
    """Matrix of dissimilarities between every pair of examples.

    :param examples: sequence of examples (rows of a table, time series...)
    :param measure: name of a measure in :data:`MEASURES` or a function ``f(a, b)``
    :return: symmetric array of shape (n, n) with zeros on the diagonal
    """
    function = get_measure(measure)
    n = len(examples)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = function(examples[i], examples[j])
    return matrix
