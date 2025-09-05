import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe, non-GUI backend
import matplotlib.pyplot as plt
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
from typing import List

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


def main():
    """
    Main entry point for time series elastic net training script.
    Loads features, labels, metadata, runs cross-validation, and saves results.
    """
    ap = argparse.ArgumentParser(description="Train time series elastic net model with cross-validation.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels",   required=True)
    ap.add_argument("--meta",     required=True)
    ap.add_argument("--outdir",   default="outputs")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=-1)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    # Load data
    X = np.load(a.features)
    df = pd.read_csv(a.labels)
    # keep PD only for the task
    df = df[df["group"].isin(["PD-NC", "PD-MCI"])].reset_index(drop=True)
    y = (df["group"] == "PD-MCI").astype(int).values

    # Pipeline & grid
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga", penalty="elasticnet", max_iter=5000,
            class_weight="balanced", random_state=42
        ))
    ])
    grid = {"clf__C": [0.01, 0.1, 1, 10],
            "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75]}

    outer = StratifiedKFold(n_splits=a.outer_splits, shuffle=True, random_state=42)

    yprob_all, ytrue_all, folds = [], [], []
    preds_rows = []

    # Outer CV (OOF predictions)
    for k, (tr, te) in enumerate(outer.split(X, y), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        inner = StratifiedKFold(n_splits=a.inner_splits, shuffle=True, random_state=100 + k)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc",
                          n_jobs=a.n_jobs, refit=True)
        gs.fit(Xtr, ytr)

        calib = CalibratedClassifierCV(gs.best_estimator_, method="isotonic", cv=3)
        calib.fit(Xtr, ytr)

        yprob = calib.predict_proba(Xte)[:, 1]
        ypred = (yprob >= 0.5).astype(int)

        auroc = roc_auc_score(yte, yprob)
        auprc = average_precision_score(yte, yprob)
        balacc = balanced_accuracy_score(yte, ypred)

        folds.append({"fold": int(k), "auroc": float(auroc),
                      "auprc": float(auprc), "bal_acc": float(balacc)})

        yprob_all.append(yprob); ytrue_all.append(yte)

        # Confusion matrix per fold
        cm = confusion_matrix(yte, ypred, labels=[0, 1])
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Fold {k}")
        plt.colorbar()
        tick = np.arange(2)
        plt.xticks(tick, ["PD-NC", "PD-MCI"], rotation=45)
        plt.yticks(tick, ["PD-NC", "PD-MCI"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
        plt.ylabel("True"); plt.xlabel("Pred")
        plt.tight_layout()
        plt.savefig(os.path.join(a.outdir, f"cm_ts_en_fold{k}.png"))
        plt.close()

        # Save OOF predictions with subject IDs if available
        subj_ids = df.iloc[te]["subject_id"].astype(str).values if "subject_id" in df.columns else [str(i) for i in te]
        for sid, yt, yp in zip(subj_ids, yte, yprob):
            preds_rows.append({"subject_id": sid, "y_true": int(yt),
                               "p_hat": float(yp), "fold": int(k)})

    # Aggregate OOF
    yprob_all = np.concatenate(yprob_all)
    ytrue_all = np.concatenate(ytrue_all)

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
        }
    }
    with open(os.path.join(a.outdir, "ts_en_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Write OOF predictions
    pd.DataFrame(preds_rows).to_csv(os.path.join(a.outdir, "preds_cv.csv"), index=False)

    # Final refit on all data (save uncalibrated + calibrated models)
    final_gs = GridSearchCV(pipe, grid, cv=outer, scoring="roc_auc",
                            n_jobs=a.n_jobs, refit=True)
    final_gs.fit(X, y)
    joblib.dump(final_gs.best_estimator_, os.path.join(a.outdir, "ts_en_model.joblib"))

    cal_final = CalibratedClassifierCV(final_gs.best_estimator_, method="isotonic", cv=5)
    cal_final.fit(X, y)
    joblib.dump(cal_final, os.path.join(a.outdir, "ts_en_model_calibrated.joblib"))

    print("Training complete. Wrote:")
    print(" - outputs/ts_en_summary.json")
    print(" - outputs/preds_cv.csv")
    print(" - outputs/ts_en_model.joblib (uncalibrated)")
    print(" - outputs/ts_en_model_calibrated.joblib (calibrated)")


if __name__ == "__main__":
    main()
