import argparse
import os
import glob
import pandas as pd
import numpy as np

def norm_sub(s: str) -> str:
    """
    Normalize subject ID to BIDS format (sub-XXX).
    Args:
        s (str): Subject ID
    Returns:
        str: Normalized subject ID
    """
    s = str(s).strip()
    return s if s.startswith("sub-") else f"sub-{s}"

from typing import Optional

def norm_ses(s: str) -> Optional[str]:
    """
    Normalize session ID to BIDS format (ses-XXX), or None if missing.
    Args:
        s (str): Session ID
    Returns:
        str or None: Normalized session ID or None
    """
    s = str(s).strip()
    if not s or s.lower() in ["nan", "none", "null"]:
        return None
    return s if s.startswith("ses-") else f"ses-{s}"

def find_confounds(deriv: str, sub_id: str, ses_id: Optional[str] = None) -> str:
    """
    Find a confounds file for this subject/session.
    Matches both .tsv and .tsv.gz and supports nested layouts.
    Args:
        deriv (str): Derivatives directory
        sub_id (str): Subject ID
        ses_id (str or None): Session ID
    Returns:
        str: Path to confounds file or empty string if not found
    """
    base = os.path.join(deriv, sub_id)
    pats = []
    if ses_id:
        pats += [
            os.path.join(base, ses_id, "func",
                         f"{sub_id}_{ses_id}_task-rest*_desc-confounds_timeseries.tsv"),
            os.path.join(base, ses_id, "func",
                         f"{sub_id}_{ses_id}_task-rest*_desc-confounds_timeseries.tsv.gz"),
            os.path.join(base, ses_id, "**", "func",
                         f"{sub_id}_{ses_id}_task-rest*_desc-confounds_timeseries.tsv"),
            os.path.join(base, ses_id, "**", "func",
                         f"{sub_id}_{ses_id}_task-rest*_desc-confounds_timeseries.tsv.gz"),
        ]
    else:
        pats += [
            os.path.join(base, "func",
                         f"{sub_id}_task-rest*_desc-confounds_timeseries.tsv"),
            os.path.join(base, "func",
                         f"{sub_id}_task-rest*_desc-confounds_timeseries.tsv.gz"),
            os.path.join(base, "**", "func",
                         f"{sub_id}_task-rest*_desc-confounds_timeseries.tsv"),
            os.path.join(base, "**", "func",
                         f"{sub_id}_task-rest*_desc-confounds_timeseries.tsv.gz"),
        ]
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return ""

def main():
    """
    Main entry point for QC filter script.
    Loads labels, finds confounds, applies thresholds, and saves results.
    """
    ap = argparse.ArgumentParser(description="Filter subjects/sessions based on QC confounds.")
    ap.add_argument("--bids", required=True)     # kept for symmetry; not used to build confound paths
    ap.add_argument("--deriv", required=True)    # MUST point to the folder that contains sub-*/func/*
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fd-thresh", type=float, default=0.25)
    ap.add_argument("--min-tr", type=int, default=120)
    ap.add_argument("--debug", action="store_true", help="print where I'm looking")
    a = ap.parse_args()

    labels_df = pd.read_csv(a.labels)
    kept_rows, dropped_rows = [], []

    for _, row in labels_df.iterrows():
        sub_id = norm_sub(row["subject_id"])
        ses_id = norm_ses(row.get("session_id", ""))

        conf_path = find_confounds(a.deriv, sub_id, ses_id)
        if a.debug:
            print(f"[DEBUG] {sub_id} {ses_id or ''} -> {conf_path or 'NO_MATCH'}")

        if not conf_path:
            dropped_rows.append((sub_id, "no_confounds"))
            continue

        try:
            conf = pd.read_csv(conf_path, sep="\t")
        except Exception as e:
            dropped_rows.append((sub_id, f"conf_read_error:{e}"))
            continue

        # choose an FD column (fMRIPrep uses 'framewise_displacement')
        fdcol = next((c for c in conf.columns if "framewise_displacement" in c.lower()), None)
        fd = pd.to_numeric(conf[fdcol], errors="coerce").fillna(0.0) if fdcol else pd.Series([0.0]*len(conf))

        mean_fd = float(fd.mean())
        n_tr = int(len(conf))

        if (mean_fd <= a.fd_thresh) and (n_tr >= a.min_tr):
            kept_rows.append(row)
        else:
            dropped_rows.append((sub_id, f"fd={mean_fd:.3f}, n_tr={n_tr}"))

    kept_df = pd.DataFrame(kept_rows)
    kept_df.to_csv(a.out, index=False)

    if dropped_rows:
        pd.DataFrame(dropped_rows, columns=["subject_id", "reason"]).to_csv(
            os.path.join(os.path.dirname(a.out), "qc_dropped.tsv"),
            sep="\t", index=False
        )
    print(f"Kept {len(kept_rows)}; Dropped {len(dropped_rows)}; Wrote {a.out}")

if __name__ == "__main__":
    main()
