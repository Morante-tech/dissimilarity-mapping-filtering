"""Distributed Dissimilarity Mapping Filtering on Apache Spark.

Requires the ``spark`` extra: ``pip install "dissimilarity-mapping-filtering[spark]"``.

Example::

    from pyspark.sql import SparkSession
    from dmf.spark import SparkDMF, to_signals

    spark = SparkSession.builder.getOrCreate()
    signals = to_signals(spark.read.table("my_table"))
    anomalies = SparkDMF(dissimilarity="euclidean", alpha=2).detect(signals)
    print(anomalies.collect())  # [(row_index, score), ...]
"""

import math

from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.sql import DataFrame as SparkDataFrame

from .core import FILTERINGS
from .dissimilarity import get_measure

KINDS = ("numerical", "categorical")


def to_signals(dataset, kind="numerical", missing_numerical=0.0, missing_categorical="NA"):
    """Turn a table into an RDD of indexed rows, the input of :class:`SparkDMF`.

    1. The dataset is converted into an RDD of rows (lists of values).
    2. Every value is cast to ``float`` (numerical) or ``str`` (categorical),
       and missing values (``None`` or NaN) are replaced by a default.
    3. Every row becomes a tuple ``(index, values)``.

    :param dataset: an RDD, a Spark DataFrame, a pandas DataFrame or any
        sequence of rows (e.g. a list of lists or a 2-D NumPy array)
    :param kind: ``"numerical"`` or ``"categorical"``
    :param missing_numerical: value used for missing numbers
    :param missing_categorical: value used for missing categories
    :return: RDD of ``(index, list_of_values)``
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; use one of {KINDS}")

    rows = _to_rdd(dataset)
    if kind == "numerical":
        rows = rows.map(lambda row: [missing_numerical if _is_missing(x) else float(x) for x in row])
    else:
        rows = rows.map(lambda row: [missing_categorical if _is_missing(x) else str(x) for x in row])
    return rows.zipWithIndex().map(lambda pair: (pair[1], pair[0]))


def _to_rdd(dataset):
    if isinstance(dataset, RDD):
        return dataset.map(list)
    if isinstance(dataset, SparkDataFrame):
        return dataset.rdd.map(list)
    if hasattr(dataset, "to_numpy") and hasattr(dataset, "columns"):  # pandas DataFrame
        dataset = dataset.to_numpy(dtype=object).tolist()
    elif hasattr(dataset, "tolist"):  # NumPy array
        dataset = dataset.tolist()
    rows = [list(row) if isinstance(row, (list, tuple)) else [row] for row in dataset]
    return SparkContext.getOrCreate().parallelize(rows)


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


class SparkDMF:
    """Dissimilarity Mapping Filtering, distributed with Spark RDDs.

    Same algorithm and parameters as :class:`dmf.DMF`, for datasets that do not
    fit in memory. Comparing every pair of rows costs O(n²), so this is meant
    for a cluster, not for very large ``n`` on a single machine.

    :param dissimilarity: ``"euclidean"``, ``"dtw"``, ``"hamming"`` or a function ``f(a, b)``
    :param mapping: ``"sum"`` (the only mapping supported in Spark)
    :param filtering: ``"zscore"`` or ``"threshold"``
    :param alpha: filtering threshold
    """

    def __init__(self, dissimilarity="euclidean", mapping="sum", filtering="zscore", alpha=0.0):
        if mapping != "sum":
            raise ValueError("SparkDMF only supports mapping='sum'")
        if filtering not in FILTERINGS:
            raise ValueError(f"unknown filtering {filtering!r}; use one of {FILTERINGS}")
        self.dissimilarity = dissimilarity
        self.mapping = mapping
        self.filtering = filtering
        self.alpha = alpha

    def scores(self, signals):
        """Score of each row: the sum of its dissimilarities to every row.

        :param signals: RDD of ``(index, values)``, as returned by :func:`to_signals`
        :return: RDD of ``(index, score)``
        """
        measure = get_measure(self.dissimilarity)
        return (
            signals.cartesian(signals)
            .map(lambda pair: (pair[0][0], measure(pair[0][1], pair[1][1])))
            .reduceByKey(lambda a, b: a + b)
        )

    def detect(self, signals):
        """Rows flagged as anomalous.

        :param signals: RDD of ``(index, values)``, as returned by :func:`to_signals`
        :return: RDD of ``(index, score)`` with only the anomalous rows
        """
        scores = self.scores(signals).cache()
        alpha = self.alpha
        if self.filtering == "threshold":
            return scores.filter(lambda pair: pair[1] > alpha)

        stats = scores.values().stats()
        mean, std = stats.mean(), stats.stdev()  # population standard deviation, as dmf.DMF
        if std == 0:
            return scores.filter(lambda pair: False)
        return scores.filter(lambda pair: (pair[1] - mean) / std > alpha)
