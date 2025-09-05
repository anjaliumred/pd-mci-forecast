import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

def nested_auc(
    X: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_jobs: int = -1,
    seed: int = 42,
    grid_C: tuple = (0.1, 1.0),
    grid_l1: tuple = (0.0, 0.5),
    calibration: str = "sigmoid"
) -> float:
    """
    Compute OOF AUROC via nested CV (same recipe as train_ts_en.py, no PCA here).
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Binary labels
        outer_splits (int): Number of outer CV splits
        inner_splits (int): Number of inner CV splits
        n_jobs (int): Number of parallel jobs
        seed (int): Random seed
        grid_C (tuple): Grid for C hyperparameter
        grid_l1 (tuple): Grid for l1_ratio hyperparameter
        calibration (str): Calibration method
    Returns:
        float: AUROC score
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga", penalty="elasticnet", max_iter=5000,
            class_weight="balanced", random_state=42
        ))
    ])
    grid = {"clf__C": list(grid_C), "clf__l1_ratio": list(grid_l1)}
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    yhat, ytrue = [], []
    for k, (tr, te) in enumerate(outer.split(X, y), 1):
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=100 + k)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=n_jobs, refit=True)
        gs.fit(X[tr], y[tr])

        calib = CalibratedClassifierCV(gs.best_estimator_, method=calibration, cv=3)
        calib.fit(X[tr], y[tr])

        yhat.append(calib.predict_proba(X[te])[:, 1])
        ytrue.append(y[te])

    yhat = np.concatenate(yhat)
    ytrue = np.concatenate(ytrue)
    return roc_auc_score(ytrue, yhat)

def main() -> None:
    """
    Main entry point for permutation test script.
    Loads features and labels, computes observed AUROC, runs permutation test, and saves results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n_perms", type=int, default=200)
    ap.add_argument("--outer_splits", type=int, default=5)
    ap.add_argument("--inner_splits", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--calibration", choices=["sigmoid","isotonic"], default="sigmoid")
    ap.add_argument("--gridC", nargs="*", type=float, default=[0.1, 1.0])
    ap.add_argument("--gridL1", nargs="*", type=float, default=[0.0, 0.5])
    a = ap.parse_args()

    X = np.load(a.features)
    df = pd.read_csv(a.labels)
    df = df[df["group"].isin(["PD-NC", "PD-MCI"])].reset_index(drop=True)
    y = (df["group"] == "PD-MCI").astype(int).values

    # observed AUROC
    auc_obs = nested_auc(X, y, a.outer_splits, a.inner_splits, a.n_jobs, seed=42,
                         grid_C=tuple(a.gridC), grid_l1=tuple(a.gridL1), calibration=a.calibration)

    # permutation null
    rng = np.random.default_rng(a.seed)
    auc_null = []
    for i in range(a.n_perms):
        y_perm = rng.permutation(y)
        auc_null.append(nested_auc(X, y_perm, a.outer_splits, a.inner_splits, a.n_jobs,
                                   seed=42 + i, grid_C=tuple(a.gridC),
                                   grid_l1=tuple(a.gridL1), calibration=a.calibration))
    auc_null = np.array(auc_null)
    p = (1 + (auc_null >= auc_obs).sum()) / (1 + len(auc_null))

    out = {
        "auc_obs": float(auc_obs),
        "auc_null_mean": float(auc_null.mean()),
        "p_value": float(p),
        "settings": {
            "outer_splits": a.outer_splits,
            "inner_splits": a.inner_splits,
            "calibration": a.calibration,
            "grid_C": a.gridC,
            "grid_l1_ratio": a.gridL1,
            "n_perms": a.n_perms,
            "seed": a.seed
        }
    }
    print(out)
    with open("outputs/perm_test.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
