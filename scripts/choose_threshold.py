
import os
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe on Windows/parallel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Dict, List

def metrics_from_confusion(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ba = 0.5 * (sens + spec)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": ba,
        "precision": prec,
        "f1": f1,
        "accuracy": acc
    }

def bootstrap_ci(vals: List[float], n_boot: int = 2000, seed: int = 7) -> List[float]:
    rng = np.random.default_rng(seed)
    vals = np.array(vals, float)
    N = len(vals)
    if N == 0:
        return [np.nan, np.nan, np.nan]
    stats = [np.mean(vals[rng.integers(0, N, N)]) for _ in range(n_boot)]
    q = np.percentile(stats, [2.5, 50, 97.5])
    return [float(q[0]), float(q[1]), float(q[2])]

def main():
    ap = argparse.ArgumentParser(description="Sweep thresholds and select optimal for classification.")
    ap.add_argument("--preds", default="outputs/preds_cv.csv",
                    help="OOF predictions CSV with columns: subject_id,y_true,p_hat,fold")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--metric", choices=["BA","F1","Youden"], default="BA",
                    help="Optimize Balanced Accuracy (BA), F1, or Youden J (sens+spec-1)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.preds).dropna(subset=["y_true","p_hat"])
    y = df["y_true"].astype(int).values
    p = df["p_hat"].astype(float).values

    thr = np.linspace(0.0, 1.0, 1001)
    ba_list, f1_list, j_list, confs = [], [], [], []
    for t in thr:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        m = metrics_from_confusion(tp, tn, fp, fn)
        ba_list.append(m["balanced_accuracy"])
        f1_list.append(m["f1"])
        j_list.append(m["sensitivity"] + m["specificity"] - 1.0)
        confs.append((tp, tn, fp, fn))

    if a.metric == "BA":
        idx = int(np.argmax(ba_list))
    elif a.metric == "F1":
        idx = int(np.argmax(f1_list))
    else:
        idx = int(np.argmax(j_list))
    t_star = float(thr[idx])
    tp, tn, fp, fn = confs[idx]
    m_star = metrics_from_confusion(tp, tn, fp, fn)

    rng = np.random.default_rng(11)
    N = len(y)
    ba_boot = []
    for _ in range(2000):
        idxb = rng.integers(0, N, N)
        yb, pb = y[idxb], p[idxb]
        yhatb = (pb >= t_star).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(yb, yhatb, labels=[0, 1]).ravel()
        ba_boot.append(metrics_from_confusion(_tp, _tn, _fp, _fn)["balanced_accuracy"])
    ba_ci = [float(np.percentile(ba_boot, 2.5)), float(np.percentile(ba_boot, 97.5))]

    with open(os.path.join(a.outdir, "ts_en_threshold.txt"), "w") as f:
        f.write(str(t_star))

    report = {
        "optimized_for": a.metric,
        "threshold": t_star,
        "oof_metrics_at_threshold": {
            **m_star, "support": int(N), "ba_ci95": ba_ci
        }
    }
    json.dump(report, open(os.path.join(a.outdir, "threshold_report.json"), "w"), indent=2)

    plt.figure()
    plt.plot(thr, ba_list, label="Balanced Accuracy")
    plt.axvline(t_star, ls="--", label=f"t*={t_star:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "threshold_sweep_ba.png"))
    plt.close()

    print("Optimized metric:", a.metric)
    print(f"Best threshold t* = {t_star:.3f}")
    print("Metrics at t*:", m_star)
    print("BA 95% CI at t*:", ba_ci)
    print(f"Wrote: ts_en_threshold.txt and threshold_report.json in {a.outdir}")

if __name__ == "__main__":
    main()
