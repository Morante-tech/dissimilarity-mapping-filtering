"""Dissimilarity Mapping Filtering: unsupervised detection of anomalous examples.

The Spark implementation lives in :mod:`dmf.spark` and needs the ``spark`` extra.
"""

from .core import DMF
from .dissimilarity import dtw, euclidean, hamming, pairwise

__all__ = ["DMF", "dtw", "euclidean", "hamming", "pairwise"]
__version__ = "1.0.0"
