import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from scripts.perm_test import nested_auc

def test_nested_auc_binary():
    # Create a small synthetic binary classification dataset
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, random_state=42)
    auc = nested_auc(X, y, outer_splits=3, inner_splits=2, n_jobs=1, seed=123,
                     grid_C=(0.1, 1.0), grid_l1=(0.0, 0.5), calibration="sigmoid")
    assert 0.0 <= auc <= 1.0

def test_nested_auc_calibration_isotonic():
    X, y = make_classification(n_samples=80, n_features=5, n_informative=3,
                               n_redundant=1, n_classes=2, random_state=7)
    auc = nested_auc(X, y, outer_splits=2, inner_splits=2, n_jobs=1, seed=7,
                     grid_C=(0.1, 1.0), grid_l1=(0.0, 0.5), calibration="isotonic")
    assert 0.0 <= auc <= 1.0

def test_nested_auc_grid_search():
    X, y = make_classification(n_samples=60, n_features=4, n_informative=2,
                               n_redundant=1, n_classes=2, random_state=21)
    auc = nested_auc(X, y, outer_splits=2, inner_splits=2, n_jobs=1, seed=21,
                     grid_C=(0.01, 0.1), grid_l1=(0.0, 0.7), calibration="sigmoid")
    assert 0.0 <= auc <= 1.0
