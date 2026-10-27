# Original scripts of the Humanoids 2015 paper

These are the scripts used for the paper *"Automatic demonstration and feature
selection for robot learning"* (Humanoids 2015), ported from Python 2 to Python 3
in 2026. They use the `dmf` package for Dynamic Time Warping and reproduce the
published results (checked by [`tests/test_paper.py`](../tests/test_paper.py)).

For new work, use the `dmf` package instead: these scripts keep their original
structure, with the parameters written as constants at the top of each file.

## Running

From the repository root:

```bash
pip install -e ".[legacy]"
python python/src/main_demons_selector.py   # select good demonstrations
python python/src/main_feature_selector.py  # select relevant features
```

| Script | What it does |
|---|---|
| `src/main_demons_selector.py` | DMF over the raw demonstrations: DTW between them, sum, z-score. With `SAVE = True` it copies the selected ones to `datasets/.../selected-demonstrations/` |
| `src/main_feature_selector.py` | DMF over each feature of the selected demonstrations, to find the relevant ones |
| `src/viewers/` | Plots of the demonstrations (3D trajectories and single features) |
| `src/alternatives/` | Two approaches explored before DMF and not used in the paper: DTW + MDS + DBSCAN, and DTW + MDS + normal fit |

## Dataset

[`datasets/experiment-1`](datasets/experiment-1) contains 10 demonstrations of moving
two objects, recorded with a camera, and the demonstrations selected with each
configuration. See its [README](datasets/experiment-1/README.md) for the format.

## License

The code in `src/` is licensed under [LGPL-3.0](src/LICENSE.TXT) (Santiago Morante
and Juan G. Victores, Universidad Carlos III de Madrid). The dataset is licensed
under the [Open Database License 1.0](datasets/LICENSE.TXT).
