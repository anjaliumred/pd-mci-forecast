import argparse
import os
import json
import numbers
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                 # safe, non-GUI backend
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
import joblib


# ---------- helpers ----------

from typing import List, Optional

class SafePCA(BaseEstimator, TransformerMixin):
    """
    PCA wrapper that caps integer n_components to the max allowed per fold:
    min(n_samples-1, n_features). Also accepts float in (0,1) to keep variance.
    Exposes `n_components` (requested) and `n_components_` (effective after fit).
    Args:
        n_components (int, float, or None): Number of components or variance fraction
        random_state (int): Random seed
        svd_solver (str): SVD solver
        whiten (bool): Whether to whiten components
    """
    def __init__(self, n_components=None, random_state=42, svd_solver="auto", whiten=False):
        # IMPORTANT: expose init params as attributes with the same names
        self.n_components = n_components          # requested spec (int or float or None)
        self.random_state = random_state
        self.svd_solver = svd_solver
        self.whiten = whiten
        # set during fit
        self._pca = None
        self.n_components_ = None                 # effective int after fit

    def fit(self, X: np.ndarray, y=None) -> "SafePCA":
            """
            Fit PCA to data, capping n_components as needed.
            Args:
                X (np.ndarray): Input data
                y: Ignored
            Returns:
                self
            """
            n_samples, n_features = X.shape
            spec = self.n_components
            # determine effective n_components
            if spec is None:
                n_comp_eff = None
            elif isinstance(spec, numbers.Real) and 0 < float(spec) < 1:
                # variance target; PCA will choose k to reach this
                n_comp_eff = float(spec)
            else:
                # integer target; cap to valid upper bound for this fold
                n_comp_eff = int(spec)
                max_k = max(1, min(n_samples - 1, n_features))
                n_comp_eff = max(1, min(n_comp_eff, max_k))
            self._pca = PCA(n_components=n_comp_eff, svd_solver=self.svd_solver,
                            whiten=self.whiten, random_state=self.random_state)
            self._pca.fit(X)
            # sklearn PCA sets .n_components_ after fit (int)
            self.n_components_ = getattr(self._pca, "n_components_", None)
            return self

    def transform(self, X: np.ndarray) -> np.ndarray:
            """
            Transform data using fitted PCA.
            Args:


# ---------- ALL IMPORTS AT TOP ----------
import argparse
import os
import json
import numbers
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe, non-GUI backend
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
import joblib
from typing import List, Optional
                X (np.ndarray): Input data
            Returns:
                np.ndarray: Transformed data
            """
            return self._pca.transform(X)


def plot_curves(y_true: List[int], y_prob: List[float], out_prefix: str) -> None:
    """
    Plot ROC, PR, and calibration curves and save to disk.
    Args:
        y_true (array-like): True binary labels
        y_prob (array-like): Predicted probabilities
        out_prefix (str): Output file prefix
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC")
    plt.tight_layout(); plt.savefig(out_prefix + "_roc.png"); plt.close()

    pr, rc, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(); plt.plot(rc, pr)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.tight_layout(); plt.savefig(out_prefix + "_pr.png"); plt.close()

    bins = np.linspace(0, 1, 11)
    ids = np.digitize(y_prob, bins) - 1
    obs = [np.mean(y_true[ids == i]) if np.any(ids == i) else np.nan for i in range(10)]
    pred = [(bins[i] + bins[i + 1]) / 2 for i in range(10)]
    plt.figure(); plt.plot(pred, obs, marker="o"); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed fraction"); plt.title("Calibration")
    plt.tight_layout(); plt.savefig(out_prefix + "_cal.png"); plt.close()


def bootstrap_auc_ci(y_true: List[int], y_prob: List[float], n: int = 2000, seed: int = 7) -> tuple[float, List[float]]:
    """
    Compute bootstrap mean and confidence interval for ROC AUC.
    Args:
        y_true (array-like): True binary labels
        y_prob (array-like): Predicted probabilities
        n (int): Number of bootstrap samples
        seed (int): Random seed
    Returns:
        tuple: (mean, [lower, upper] percentiles)
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    stats = [roc_auc_score(y_true[idx], y_prob[idx])
             for idx in [rng.integers(0, len(y_true), len(y_true)) for _ in range(n)]]
    stats = np.array(stats)
    return float(np.mean(stats)), [float(np.percentile(stats, 2.5)),
                                   float(np.percentile(stats, 97.5))]


def ece(y_true: List[int], y_prob: List[float], bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Args:
        y_true (array-like): True binary labels
        y_prob (array-like): Predicted probabilities
        bins (int): Number of bins
    Returns:
        float: ECE value
    """
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.digitize(y_prob, edges) - 1
    e = 0.0
    for b in range(bins):
        m = ids == b
        if np.any(m):
            e += abs(y_true[m].mean() - y_prob[m].mean()) * m.mean()
    return float(e)


# ---------- main ----------

def main():
    """
    Main entry point for time series elastic net + PCA training script.
    Loads features, labels, metadata, runs cross-validation, and saves results.
    """
    ap = argparse.ArgumentParser(description="Train time series elastic net model with PCA and cross-validation.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels",   required=True)
    ap.add_argument("--meta",     required=True)
    ap.add_argument("--outdir",   default="outputs")

    # CV + compute
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=-1)

    # Model grid
    ap.add_argument("--grid-C",  nargs="*", type=float, default=[0.1, 1.0])
    ap.add_argument("--grid-l1", nargs="*", type=float, default=[0.0, 0.5])

    # Calibration
    ap.add_argument("--calibration", choices=["sigmoid","isotonic"], default="sigmoid")

    # PCA
    ap.add_argument("--pca-components", type=float, default=None,
                    help="If int>=1, uses that many comps (capped per fold to <= n_train-1). "
                         "If 0<val<1, keeps that fraction of variance. If omitted, no PCA.")

    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # Load data
    X = np.load(a.features)
    df = pd.read_csv(a.labels)
    df = df[df["group"].isin(["PD-NC", "PD-MCI"])].reset_index(drop=True)
    y = (df["group"] == "PD-MCI").astype(int).values

    # Build pipeline
    steps = [("scaler", StandardScaler())]
    if a.pca_components is not None:
        steps.append(("pca", SafePCA(n_components=a.pca_components, random_state=42)))
    steps.append(("clf", LogisticRegression(
        solver="saga", penalty="elasticnet", max_iter=5000,
        class_weight="balanced", random_state=42
    )))
    pipe = Pipeline(steps)

    grid = {"clf__C": a.grid_C, "clf__l1_ratio": a.grid_l1}
    outer = StratifiedKFold(n_splits=a.outer_splits, shuffle=True, random_state=42)

    yprob_all, ytrue_all, folds, preds_rows = [], [], [], []

    # Outer CV
    for k, (tr, te) in enumerate(outer.split(X, y), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        inner = StratifiedKFold(n_splits=a.inner_splits, shuffle=True, random_state=100 + k)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=a.n_jobs, refit=True)
        gs.fit(Xtr, ytr)

        calib = CalibratedClassifierCV(gs.best_estimator_, method=a.calibration, cv=3)
        calib.fit(Xtr, ytr)

        yprob = calib.predict_proba(Xte)[:, 1]
        ypred = (yprob >= 0.5).astype(int)

        auroc = roc_auc_score(yte, yprob)
        auprc = average_precision_score(yte, yprob)
        balacc = balanced_accuracy_score(yte, ypred)
        folds.append({"fold": int(k), "auroc": float(auroc),
                      "auprc": float(auprc), "bal_acc": float(balacc)})

        yprob_all.append(yprob); ytrue_all.append(yte)

        # Confusion matrix
        cm = confusion_matrix(yte, ypred, labels=[0, 1])
        plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title(f"Confusion Fold {k}")
        plt.colorbar(); tick = np.arange(2)
        plt.xticks(tick, ["PD-NC", "PD-MCI"], rotation=45); plt.yticks(tick, ["PD-NC", "PD-MCI"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
        plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
        plt.savefig(os.path.join(a.outdir, f"cm_ts_en_fold{k}.png")); plt.close()

        # OOF preds
        subj_ids = df.iloc[te]["subject_id"].astype(str).values if "subject_id" in df.columns else [str(i) for i in te]
        for sid, yt, yp in zip(subj_ids, yte, yprob):
            preds_rows.append({"subject_id": sid, "y_true": int(yt), "p_hat": float(yp), "fold": int(k)})

    # Aggregate OOF
    yprob_all = np.concatenate(yprob_all); ytrue_all = np.concatenate(ytrue_all)

    # Curves & summary
    plot_curves(ytrue_all, yprob_all, os.path.join(a.outdir, "ts_en"))
    mean_auc, ci = bootstrap_auc_ci(ytrue_all, yprob_all)
    summary = {
        "folds": folds,
        "overall": {
            "auroc_mean_bootstrap": float(mean_auc),
            "auroc_ci95": ci,
            "auprc": float(average_precision_score(ytrue_all, yprob_all)),
            "bal_acc": float(balanced_accuracy_score(ytrue_all, (yprob_all >= 0.5).astype(int))),
            "ece_10bin": float(ece(ytrue_all, yprob_all, 10))
        },
        "settings": {
            "pca_components_spec": a.pca_components,
            "calibration": a.calibration,
            "grid_C": a.grid_C,
            "grid_l1_ratio": a.grid_l1
        }
    }
    with open(os.path.join(a.outdir, "ts_en_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # OOF preds
    pd.DataFrame(preds_rows).to_csv(os.path.join(a.outdir, "preds_cv.csv"), index=False)

    # Final refit on all data
    final_gs = GridSearchCV(pipe, grid, cv=outer, scoring="roc_auc", n_jobs=a.n_jobs, refit=True)
    final_gs.fit(X, y)
    joblib.dump(final_gs.best_estimator_, os.path.join(a.outdir, "ts_en_model.joblib"))

    cal_final = CalibratedClassifierCV(final_gs.best_estimator_, method=a.calibration, cv=5)
    cal_final.fit(X, y)
    joblib.dump(cal_final, os.path.join(a.outdir, "ts_en_model_calibrated.joblib"))

    # Record actual PCA size if present
    final_pca_k = None
    if "pca" in final_gs.best_estimator_.named_steps:
        final_pca_k = getattr(final_gs.best_estimator_.named_steps["pca"], "n_components_", None)
    with open(os.path.join(a.outdir, "ts_en_meta.json"), "w") as f:
        json.dump({"final_pca_components_": None if final_pca_k is None else int(final_pca_k)}, f, indent=2)

    print("Training complete. Wrote:")
    print(" - outputs/ts_en_summary.json")
    print(" - outputs/preds_cv.csv")
    print(" - outputs/ts_en_model.joblib (uncalibrated)")
    print(" - outputs/ts_en_model_calibrated.joblib (calibrated)")
    if final_pca_k is not None:
        print(f" - PCA components used in final model: {final_pca_k}")


if __name__ == "__main__":
    main()
