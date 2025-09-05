
import argparse
import json
import numpy as np
import pandas as pd
import joblib

def main() -> None:
    """
    Main entry point for prediction script on new cohort.
    Loads model, threshold, features, predicts probabilities and labels, and saves results.
    """
    ap = argparse.ArgumentParser(description="Predict new cohort using trained model and threshold.")
    ap.add_argument("--model", default="outputs/ts_en_model_calibrated.joblib")
    ap.add_argument("--threshold", default="outputs/ts_en_threshold.txt")
    ap.add_argument("--features", required=True)  # connectomes.npy for new cohort
    ap.add_argument("--ids", default=None)        # optional: CSV with subject_id column
    ap.add_argument("--out", default="outputs/preds_new.csv")
    a = ap.parse_args()

    clf = joblib.load(a.model)
    t = float(open(a.threshold).read().strip())
    X = np.load(a.features)

    p = clf.predict_proba(X)[:,1]
    yhat = (p >= t).astype(int)

    if a.ids and a.ids.lower() != "none":
        ids = pd.read_csv(a.ids)["subject_id"].astype(str).values
        if len(ids) != len(p):
            raise ValueError("ids and features length mismatch")
    else:
        ids = [f"sub_{i:03d}" for i in range(len(p))]

    pd.DataFrame({"subject_id": ids, "p_hat": p, "pred": yhat}).to_csv(a.out, index=False)
    print(f"Wrote {a.out} with threshold {t:.3f}")

if __name__ == "__main__":
    main()
