"""Reproduce the demonstration selection published with the Humanoids 2015 paper.

The dataset in python/datasets/experiment-1 has 10 raw demonstrations and, in
selected-demonstrations/, the ones that were kept with each configuration.
"""

from pathlib import Path

import numpy as np
import pytest

from dmf import DMF, dtw

EXPERIMENT = Path(__file__).resolve().parents[1] / "python" / "datasets" / "experiment-1"
RAW = sorted((EXPERIMENT / "raw").glob("*.csv"))


def normalized_dtw(a, b):
    # the `dtw` library used in 2015 divided the accumulated cost by both lengths
    return dtw(a, b) / (len(a) + len(b))


@pytest.mark.parametrize("collapse", ["cols", "rows"])
@pytest.mark.parametrize("alpha", [0.5, 1])
def test_selected_demonstrations(collapse, alpha):
    # drop the timestamp column, then collapse each demonstration to a 1-D series
    demonstrations = [np.loadtxt(path)[:, 1:] for path in RAW]
    axis = 1 if collapse == "cols" else 0
    series = [demonstration.sum(axis=axis) for demonstration in demonstrations]

    kept = DMF(dissimilarity=normalized_dtw, alpha=alpha).fit_predict(series) == False  # noqa: E712

    published = EXPERIMENT / "selected-demonstrations" / "NORM-NONE" / f"SUM-{collapse}" / f"ALPHA-{alpha}"
    expected = sorted(path.name for path in published.glob("*.csv"))
    assert [path.name for path, keep in zip(RAW, kept) if keep] == expected
