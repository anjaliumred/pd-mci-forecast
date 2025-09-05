import pytest
import numpy as np
from scripts.train_ts_en import plot_curves, bootstrap_auc_ci, ece

# Dummy data for testing
Y_TRUE = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
Y_PROB = [0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5, 0.4, 0.2]


def test_plot_curves(tmp_path):
    out_prefix = str(tmp_path / "test")
    plot_curves(Y_TRUE, Y_PROB, out_prefix)
    assert (tmp_path / "test_roc.png").exists()
    assert (tmp_path / "test_pr.png").exists()
    assert (tmp_path / "test_cal.png").exists()


def test_bootstrap_auc_ci():
    mean, ci = bootstrap_auc_ci(Y_TRUE, Y_PROB, n=100)
    assert isinstance(mean, float)
    assert isinstance(ci, list)
    assert len(ci) == 2
    assert ci[0] <= mean <= ci[1]


def test_ece():
    val = ece(Y_TRUE, Y_PROB, bins=5)
    assert isinstance(val, float)
    assert val >= 0
