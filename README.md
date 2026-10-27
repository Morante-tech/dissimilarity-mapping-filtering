# Dissimilarity Mapping Filtering (DMF)

DMF is an unsupervised method to find anomalous examples in a dataset. It was
developed during my PhD to select good robot demonstrations and relevant
features automatically, and later adapted to Apache Spark as a general anomaly
detector.

The algorithm has three steps:

1. **Dissimilarity**: compare every pair of examples (Euclidean, DTW or Hamming distance).
2. **Mapping**: reduce each example's dissimilarities to a single value (e.g. their sum).
3. **Filtering**: discard the examples whose value is too far from the rest (z-score or threshold).

Reference: S. Morante, J. G. Victores and C. Balaguer, "Automatic demonstration and
feature selection for robot learning", *2015 IEEE-RAS 15th International Conference
on Humanoid Robots (Humanoids)*, Seoul, 2015, pp. 428-433.
[IEEE Xplore](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7363569)

## Repository layout

| Folder | Content |
|---|---|
| [`python/`](python) | Original implementation (2015) used in the paper: demonstration and feature selection, plus the dataset of experiment 1 |
| [`spark/`](spark) | Distributed implementation on PySpark (2016), with an example notebook |

Both folders keep their full git history. They come from
[phd-thesis-cgda](https://github.com/Morante-tech/phd-thesis-cgda) (`demonstration-feature-selection/`)
and [distributed-machine-learning](https://github.com/Morante-tech/distributed-machine-learning) (`anomaly_detection/`).

> **Status:** the code is from 2015-2016 and has not been updated yet
> (Python 2 in places, legacy Spark APIs). A modernization is planned.

## License

- `spark/` and the repository root: [MIT](LICENSE).
- `python/src/`: [LGPL-3.0](python/src/LICENSE.TXT), co-authored with Juan G. Victores (Universidad Carlos III de Madrid).
- `python/datasets/`: [Open Database License (ODbL) 1.0](python/datasets/LICENSE.TXT).
