"""Dissimilarity Mapping Filtering (DMF) on in-memory data."""

import numpy as np

from .dissimilarity import get_measure, pairwise

MAPPINGS = {"sum": lambda matrix: matrix.sum(axis=1)}
FILTERINGS = ("zscore", "threshold")


class DMF:
    """Unsupervised detection of anomalous examples.

    DMF works in three steps:

    1. **Dissimilarity**: compare every pair of examples.
    2. **Mapping**: reduce the comparisons of each example to a single score.
    3. **Filtering**: flag the examples whose score is above ``alpha``, either
       directly (``"threshold"``) or after standardizing the scores (``"zscore"``).

    Reference: S. Morante, J. G. Victores and C. Balaguer, "Automatic
    demonstration and feature selection for robot learning", IEEE-RAS
    Humanoids 2015, pp. 428-433. doi:10.1109/HUMANOIDS.2015.7363569

    :param dissimilarity: ``"euclidean"``, ``"dtw"``, ``"hamming"`` or a function ``f(a, b)``
    :param mapping: ``"sum"`` or a function that turns the (n, n) dissimilarity matrix into n scores
    :param filtering: ``"zscore"`` or ``"threshold"``
    :param alpha: filtering threshold

    After :meth:`fit`, the model exposes:

    - ``dissimilarities_``: (n, n) dissimilarity matrix
    - ``scores_``: score of each example after mapping
    - ``zscores_``: standardized scores (only with ``filtering="zscore"``)
    - ``outliers_``: boolean mask, True for anomalous examples

    Example::

        >>> import numpy as np
        >>> from dmf import DMF
        >>> data = np.r_[np.sin(np.arange(20)), 8.0].reshape(-1, 1)
        >>> DMF(alpha=2).fit(data).outlier_indices()
        array([20])
    """

    def __init__(self, dissimilarity="euclidean", mapping="sum", filtering="zscore", alpha=0.0):
        self.dissimilarity = dissimilarity
        self.mapping = mapping
        self.filtering = filtering
        self.alpha = alpha

    def fit(self, examples):
        """Compute dissimilarities, scores and outliers for ``examples``.

        :param examples: a 2-D array (one example per row) or a list of
            examples, which may have different lengths when using DTW
        :return: the fitted model
        """
        get_measure(self.dissimilarity)  # fail early on a bad name
        if self.filtering not in FILTERINGS:
            raise ValueError(f"unknown filtering {self.filtering!r}; use one of {FILTERINGS}")
        if len(examples) < 2:
            raise ValueError("DMF needs at least two examples")

        self.dissimilarities_ = pairwise(examples, self.dissimilarity)
        self.scores_ = self._map(self.dissimilarities_)
        if self.filtering == "zscore":
            self.zscores_ = _zscore(self.scores_)
            self.outliers_ = self.zscores_ > self.alpha
        else:
            self.outliers_ = self.scores_ > self.alpha
        return self

    def fit_predict(self, examples):
        """Fit the model and return the boolean outlier mask."""
        return self.fit(examples).outliers_

    def outlier_indices(self):
        """Indices of the examples flagged as anomalous."""
        return np.flatnonzero(self.outliers_)

    def _map(self, matrix):
        if callable(self.mapping):
            return np.asarray(self.mapping(matrix), dtype=float)
        try:
            return MAPPINGS[self.mapping](matrix)
        except KeyError:
            raise ValueError(f"unknown mapping {self.mapping!r}; use one of {sorted(MAPPINGS)} or a function") from None


def _zscore(values):
    """Standardize values; all zeros when every value is equal."""
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std
