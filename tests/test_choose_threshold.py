import pytest
import numpy as np
from scripts.choose_threshold import metrics_from_confusion, bootstrap_ci

def test_metrics_from_confusion():
    m = metrics_from_confusion(10, 20, 5, 3)
    assert isinstance(m, dict)
    assert "sensitivity" in m
    assert "specificity" in m
    assert "balanced_accuracy" in m
    assert "precision" in m
    assert "f1" in m
    assert "accuracy" in m


def test_bootstrap_ci():
    vals = np.random.rand(100)
    ci = bootstrap_ci(list(vals), n_boot=100)
    assert isinstance(ci, list)
    assert len(ci) == 3
    assert all(isinstance(x, float) or np.isnan(x) for x in ci)
