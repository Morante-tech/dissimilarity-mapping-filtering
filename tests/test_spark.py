import numpy as np
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from dmf import DMF  # noqa: E402
from dmf.spark import SparkDMF, to_signals  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("dmf-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def sine_with_outliers():
    values = np.sin(np.arange(50))
    values[[10, 20, 30]] = [5, 19, 20]
    return values.reshape(-1, 1)


def test_matches_in_memory_implementation(spark):
    data = sine_with_outliers()
    found = sorted(index for index, _ in SparkDMF(alpha=0).detect(to_signals(data)).collect())
    assert found == list(DMF(alpha=0).fit(data).outlier_indices())


def test_scores_match_in_memory_implementation(spark):
    data = sine_with_outliers()
    scores = dict(SparkDMF().scores(to_signals(data)).collect())
    np.testing.assert_allclose([scores[i] for i in range(len(data))], DMF().fit(data).scores_)


def test_inputs_from_spark_and_pandas_dataframes(spark):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [0.0, 1.0, None], "b": [2.0, 3.0, 4.0]})
    expected = [(0, [0.0, 2.0]), (1, [1.0, 3.0]), (2, [-1.0, 4.0])]
    assert to_signals(frame, missing_numerical=-1.0).collect() == expected
    spark_frame = spark.createDataFrame([(0.0, 2.0), (1.0, 3.0), (None, 4.0)], ["a", "b"])
    assert to_signals(spark_frame, missing_numerical=-1.0).collect() == expected


def test_zero_is_not_treated_as_missing(spark):
    assert to_signals([[0, 5]], missing_numerical=99.0).collect() == [(0, [0.0, 5.0])]


def test_categorical_hamming(spark):
    rows = [["a", "b", "c"]] * 5 + [["x", None, "z"]]
    signals = to_signals(rows, kind="categorical")
    assert signals.collect()[-1] == (5, ["x", "NA", "z"])
    found = [index for index, _ in SparkDMF(dissimilarity="hamming", alpha=1).detect(signals).collect()]
    assert found == [5]


def test_threshold_filtering(spark):
    found = SparkDMF(filtering="threshold", alpha=100).detect(to_signals([[0], [1], [2], [50]])).keys()
    assert found.collect() == [3]


def test_rejects_unsupported_mapping():
    with pytest.raises(ValueError):
        SparkDMF(mapping="max")
