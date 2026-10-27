# Dissimilarity Mapping Filtering (DMF)

[![tests](https://github.com/Morante-tech/dissimilarity-mapping-filtering/actions/workflows/tests.yml/badge.svg)](https://github.com/Morante-tech/dissimilarity-mapping-filtering/actions/workflows/tests.yml)

DMF is an unsupervised method to find anomalous examples in a dataset. It was
developed during my PhD to select good robot demonstrations and relevant
features automatically, and later adapted to Apache Spark as a general anomaly
detector.

The algorithm has three steps:

1. **Dissimilarity**: compare every pair of examples (Euclidean, DTW or Hamming distance).
2. **Mapping**: reduce each example's dissimilarities to a single score (their sum).
3. **Filtering**: flag the examples whose score is too far from the rest (z-score or threshold).

Reference: S. Morante, J. G. Victores and C. Balaguer, "Automatic demonstration and
feature selection for robot learning", *2015 IEEE-RAS 15th International Conference
on Humanoid Robots (Humanoids)*, Seoul, 2015, pp. 428-433.
[doi:10.1109/HUMANOIDS.2015.7363569](https://doi.org/10.1109/HUMANOIDS.2015.7363569)

## Installation

```bash
pip install "git+https://github.com/Morante-tech/dissimilarity-mapping-filtering"
```

For the Spark version add the `spark` extra (it needs Java 17 or later):

```bash
pip install "dissimilarity-mapping-filtering[spark] @ git+https://github.com/Morante-tech/dissimilarity-mapping-filtering"
```

## Usage

```python
import numpy as np
from dmf import DMF

# a sine wave with three anomalous spikes
values = np.sin(np.arange(50))
values[[10, 20, 30]] = [5, 19, 20]

model = DMF(dissimilarity="euclidean", filtering="zscore", alpha=0)
model.fit(values.reshape(-1, 1))
model.outlier_indices()  # array([10, 20, 30])
```

Time series of different lengths can be compared with Dynamic Time Warping:

```python
series = [np.sin(np.linspace(0, 6, n)) for n in (40, 35, 38)] + [np.cos(np.linspace(0, 6, 40)) * 3]
DMF(dissimilarity="dtw", alpha=1).fit(series).outlier_indices()  # array([3])
```

After `fit`, the model keeps `dissimilarities_` (pairwise matrix), `scores_`,
`zscores_` and `outliers_` (boolean mask). The dissimilarity and the mapping can
also be your own functions.

### Spark

```python
from pyspark.sql import SparkSession
from dmf.spark import SparkDMF, to_signals

spark = SparkSession.builder.getOrCreate()
signals = to_signals(spark.read.table("my_table"))       # RDD of (row_index, values)
anomalies = SparkDMF(dissimilarity="euclidean", alpha=2).detect(signals)
anomalies.collect()                                      # [(row_index, score), ...]
```

`to_signals` also accepts RDDs, pandas DataFrames, NumPy arrays and lists of rows.
On a cluster, the package must be installed on the executors too. See
[examples/spark_anomaly_detection.ipynb](examples/spark_anomaly_detection.ipynb).

## Repository layout

| Folder | Content |
|---|---|
| [`src/dmf/`](src/dmf) | The `dmf` package: in-memory implementation (`dmf.DMF`) and Spark implementation (`dmf.spark`) |
| [`tests/`](tests) | Tests, including the reproduction of the demonstration selection published in the paper |
| [`examples/`](examples) | Example notebook for Spark |
| [`python/`](python) | Original scripts of the 2015 paper (ported to Python 3) and the dataset of experiment 1 |

`python/` comes from [phd-thesis-cgda](https://github.com/Morante-tech/phd-thesis-cgda) and the Spark code from
[distributed-machine-learning](https://github.com/Morante-tech/distributed-machine-learning); both keep their git history.

## Development

```bash
pip install -e ".[spark,legacy,test]"
pytest
```

## License

- The `dmf` package, tests and examples: [MIT](LICENSE).
- `python/src/`: [LGPL-3.0](python/src/LICENSE.TXT), co-authored with Juan G. Victores (Universidad Carlos III de Madrid).
- `python/datasets/`: [Open Database License (ODbL) 1.0](python/datasets/LICENSE.TXT).
