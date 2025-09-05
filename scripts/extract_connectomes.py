
import argparse
import os
import json
import glob
import yaml
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


from typing import Tuple, Optional, Any, List

# ------------------------- helpers -------------------------

def get_atlas(name: str, n_parcels: int = 200) -> Tuple[Any, Optional[List[str]]]:
    """
    Return atlas image and ROI names for Schaefer/AAL or a custom labels NIfTI path.
    Args:
        name (str): Atlas name or path
        n_parcels (int): Number of parcels (for Schaefer)
    Returns:
        tuple: (labels_img, roi_names)
    """
    from typing import Tuple, Optional, Any
    name_l = str(name).lower()
    if name_l == "schaefer":
        a = datasets.fetch_atlas_schaefer_2018(n_rois=int(n_parcels),
                                               yeo_networks=7, resolution_mm=2)
        return load_img(a["maps"]), list(a["labels"])
    if name_l == "aal":
        a = datasets.fetch_atlas_aal()
        return load_img(a["maps"]), list(a["labels"])
    # otherwise treat 'name' as path to a labels image
    return load_img(name), None


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


def norm_ses(s: str) -> Optional[str]:
    """
    Normalize session ID to BIDS format (ses-XXX), or None if missing.
    Args:
        s (str): Session ID
    Returns:
        str or None: Normalized session ID or None
    """
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    return s if s.startswith("ses-") else f"ses-{s}"


def first_hit(patterns: List[str]) -> str:
    """
    Return the first file hit from a list of glob patterns.
    Args:
        patterns (list): List of glob patterns
    Returns:
        str: First file path found or empty string
    """
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return ""


def find_bold(deriv: str, sub_id: str, ses_id: Optional[str], task: str, space: str, desc: str) -> str:
    """
    Find preprocessed BOLD file allowing run-*, echo-*, part-* and *_res-* token between 'space-...' and 'desc-...'.
    Args:
        deriv (str): Derivatives directory
        sub_id (str): Subject ID
        ses_id (str or None): Session ID
        task (str): Task name
        space (str): Space name
        desc (str): Description
    Returns:
        str: Path to BOLD file or empty string
    """
    base = os.path.join(deriv, sub_id)
    core = f"{sub_id}" + (f"_{ses_id}" if ses_id else "") + \
           f"_task-{task}_*space-{space}*desc-{desc}_bold.nii.gz"
    pats = []
    if ses_id:
        pats += [
            os.path.join(base, ses_id, "func", core),
            os.path.join(base, ses_id, "**", "func", core),
        ]
    else:
        pats += [
            os.path.join(base, "func", core),
            os.path.join(base, "**", "func", core),
        ]
    return first_hit(pats)


def find_confounds_for_bold(bold_path: str) -> str:
    """
    Prefer confounds file next to BOLD; support .tsv and .tsv.gz.
    Args:
        bold_path (str): Path to BOLD file
    Returns:
        str: Path to confounds file or empty string
    """
    if not bold_path:
        return ""
    func_dir = os.path.dirname(bold_path)
    hits = sorted(glob.glob(os.path.join(func_dir, "*desc-confounds_timeseries.tsv*")))
    if hits:
        return hits[0]
    # fallback: search upward within subject directory
    sub_dir = os.path.dirname(os.path.dirname(func_dir))
    hits = sorted(glob.glob(os.path.join(sub_dir, "**", "func", "*desc-confounds_timeseries.tsv*"),
                            recursive=True))
    return hits[0] if hits else ""


def bold_json_path(bold_path: str) -> str:
    """
    Return path to JSON sidecar for BOLD file if it exists.
    Args:
        bold_path (str): Path to BOLD file
    Returns:
        str: Path to JSON file or empty string
    """
    if not bold_path:
        return ""
    j = bold_path.replace(".nii.gz", ".json")
    return j if os.path.exists(j) else ""


def load_tr(bold_path: str) -> Optional[float]:
    """
    Return TR in seconds (prefer JSON; fallback to NIfTI header).
    Args:
        bold_path (str): Path to BOLD file
    Returns:
        float or None: TR value in seconds or None
    """
    j = bold_json_path(bold_path)
    if j:
        try:
            with open(j, "r") as f:
                meta = json.load(f)
            tr = meta.get("RepetitionTime", None)
            if tr:
                return float(tr)
        except Exception:
            pass
    try:
        hdr_tr = float(nib.load(bold_path).header.get_zooms()[3])
        if hdr_tr > 0:
            return hdr_tr
    except Exception:
        pass
    return None  # nilearn will accept None (no temporal filtering by frequency)


def make_scrub_mask(fd_series: pd.Series, fd_thresh: float = 0.5, n_before: int = 1, n_after: int = 1) -> List[int]:
    """
    Return indices to keep after scrubbing frames with FD > threshold, also removing n_before/n_after neighbors. No SciPy dependency.
    Args:
        fd_series (pd.Series): Framewise displacement values
        fd_thresh (float): FD threshold
        n_before (int): Number of frames before to remove
        n_after (int): Number of frames after to remove
    Returns:
        list: Indices to keep
    """
    fd = pd.to_numeric(fd_series, errors="coerce").fillna(0.0).values
    bad = fd > float(fd_thresh)
    mask = bad.copy()
    # add neighbors before
    for k in range(1, int(n_before) + 1):
        mask[k:] |= bad[:-k]
    # add neighbors after
    for k in range(1, int(n_after) + 1):
        mask[:-k] |= bad[k:]
    keep_idx = np.where(~mask)[0]
    return keep_idx.tolist()


def select_confounds(conf_df: pd.DataFrame, use_gsr: bool = False, max_acompcor: int = 5) -> Optional[np.ndarray]:
    """
    Pick a reasonable nuisance set from fMRIPrep confounds.
    Args:
        conf_df (pd.DataFrame): Confounds DataFrame
        use_gsr (bool): Whether to use global signal
        max_acompcor (int): Max number of aCompCor components
    Returns:
        np.ndarray or None: Selected confounds matrix or None
    """
    cols = []
    # 6 rigid-body + some standard signals
    base = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
            "white_matter", "csf", "wm_csf", "std_dvars", "dvars"]
    for c in base:
        if c in conf_df.columns:
            cols.append(c)
    if use_gsr and "global_signal" in conf_df.columns:
        cols.append("global_signal")
    # aCompCor components
    acomp = [c for c in conf_df.columns if c.lower().startswith("a_comp_cor")]
    acomp = acomp[:int(max_acompcor)]
    cols += acomp
    if not cols:
        return None
    X = conf_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    return X


def vec_upper(mat: np.ndarray) -> np.ndarray:
    """
    Return upper triangle of a matrix as a vector.
    Args:
        mat (np.ndarray): Input matrix
    Returns:
        np.ndarray: Upper triangle vector
    """
    iu = np.triu_indices_from(mat, 1)
    return mat[iu]


# ------------------------- main -------------------------

def main():
    """
    Main entry point for connectome extraction script.
    Loads config, finds BOLD and confounds, extracts connectomes, and saves results.
    """
    ap = argparse.ArgumentParser(description="Extract connectomes from BIDS derivatives using config YAML.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Paths
    bids_root  = cfg.get("bids_root")  # unused for path construction (kept for completeness)
    deriv      = cfg["deriv"]          # MUST be the folder that directly contains sub-*/
    labels_csv = cfg["labels_csv"]

    # What to extract
    atlas_name = cfg.get("atlas_name", "schaefer")
    n_parcels  = int(cfg.get("n_parcels", 200))
    fc_kind    = cfg.get("fc_kind", "tangent")
    space      = cfg.get("space", "MNI152NLin2009cAsym")
    task       = cfg.get("task", "rest")
    desc       = cfg.get("desc", "preproc")

    # Denoising / thresholds
    min_tr     = int(cfg.get("min_timepoints", 120))
    use_conf   = bool(cfg.get("use_confounds", True))
    gsr        = bool(cfg.get("global_signal_regression", False))

    filt       = cfg.get("filter", {}) or {}
    low_pass   = filt.get("low_pass", 0.1)
    high_pass  = filt.get("high_pass", 0.01)
    tr_fixed   = cfg.get("tr_s", None) or filt.get("t_r", None)

    scrub_cfg  = cfg.get("scrub", {}) or {}
    scrub_on   = bool(scrub_cfg.get("enabled", False))
    fd_thresh  = float(scrub_cfg.get("fd_thresh", 0.5))
    n_before   = int(scrub_cfg.get("n_before", 1))
    n_after    = int(scrub_cfg.get("n_after", 1))

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts_dir = os.path.join(out_dir, "timeseries")
    os.makedirs(ts_dir, exist_ok=True)

    # Atlas + masker template (we set t_r per subject, if needed)
    labels_img, roi_names = get_atlas(atlas_name, n_parcels=n_parcels)
    base_masker_kwargs = dict(
        labels_img=labels_img,
        standardize=True,
        detrend=True,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=None,
    )

    # Load labels; keep PD groups only if present
    df = pd.read_csv(labels_csv)
    if "group" in df.columns:
        df = df[df["group"].isin(["PD-NC", "PD-MCI"])].reset_index(drop=True)

    subjects_ts = []
    kept_rows = []

    for _, row in df.iterrows():
        sub_id = norm_sub(row["subject_id"])
        ses_id = norm_ses(row.get("session_id", ""))

        bold = find_bold(deriv, sub_id, ses_id, task, space, desc)
        if args.debug:
            print(f"[DEBUG] BOLD for {sub_id} {ses_id or ''}: {bold or 'NOT FOUND'}")
        if not bold:
            continue

        # TR
        tr_here = float(tr_fixed) if tr_fixed else load_tr(bold)

        # Confounds + optional scrubbing
        conf = None
        sample_mask = None
        if use_conf:
            conf_path = find_confounds_for_bold(bold)
            if args.debug:
                print(f"[DEBUG] Confounds for {sub_id}: {conf_path or 'NOT FOUND'}")
            if conf_path:
                try:
                    conf_df = pd.read_csv(conf_path, sep="\t")
                    if scrub_on:
                        fd_col = next((c for c in conf_df.columns
                                       if "framewise_displacement" in c.lower()), None)
                        if fd_col is not None:
                            keep_idx = make_scrub_mask(conf_df[fd_col],
                                                       fd_thresh=fd_thresh,
                                                       n_before=n_before,
                                                       n_after=n_after)
                            if len(keep_idx) == 0:
                                if args.debug:
                                    print(f"[DEBUG] {sub_id}: all frames scrubbed; skipping")
                                continue
                            sample_mask = np.array(keep_idx, dtype=int)
                    conf = select_confounds(conf_df, use_gsr=gsr, max_acompcor=5)
                except Exception as e:
                    if args.debug:
                        print(f"[DEBUG] Confounds read error for {sub_id}: {e}")

        # Masker (bind TR now)
        masker = NiftiLabelsMasker(**{**base_masker_kwargs, "t_r": tr_here})

        # Extract ROI time series
        try:
            ts = masker.fit_transform(bold, confounds=conf, sample_mask=sample_mask)
        except Exception as e:
            if args.debug:
                print(f"[DEBUG] Masker failed for {sub_id}: {e}")
            continue

        if ts is None or ts.shape[0] < min_tr:
            if args.debug:
                print(f"[DEBUG] {sub_id}: too few timepoints ({ts.shape[0] if ts is not None else 0})")
            continue

        # Optional: save per-subject timeseries (handy for debugging)
        np.save(os.path.join(ts_dir, f"{sub_id}{'_'+ses_id if ses_id else ''}.npy"), ts)

        subjects_ts.append(ts)
        kept_rows.append({
            "subject_id": sub_id,
            "session_id": ses_id or "",
            "group": row.get("group", "")
        })

    if len(subjects_ts) == 0:
        raise RuntimeError("No subjects passed extraction. Check paths/config or relax thresholds.")

    # Fit connectomes across subjects (proper 'tangent' behavior)
    conn = ConnectivityMeasure(kind=fc_kind)
    mats = conn.fit_transform(subjects_ts)  # (n_subj, n_rois, n_rois)

    # Vectorize upper triangle
    X = np.vstack([vec_upper(m) for m in mats])
    np.save(os.path.join(out_dir, "connectomes.npy"), X)
    pd.DataFrame(kept_rows).to_csv(os.path.join(out_dir, "labels.csv"), index=False)

    meta = {
        "atlas": atlas_name,
        "n_parcels": int(n_parcels),
        "fc_kind": fc_kind,
        "space": space,
        "task": task,
        "desc": desc,
        "vectorize_upper": True,
        "n_features": int(X.shape[1]),
        "use_confounds": bool(use_conf),
        "scrub": dict(enabled=bool(scrub_on), fd_thresh=float(fd_thresh),
                      n_before=int(n_before), n_after=int(n_after)),
        "filter": dict(low_pass=low_pass, high_pass=high_pass, t_r=tr_fixed),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if roi_names:
        with open(os.path.join(out_dir, "roi_names.txt"), "w") as f:
            for r in roi_names:
                f.write(f"{r}\n")

    print(f"Saved features to outputs\\connectomes.npy with shape {X.shape} "
          f"(n_subjects={len(subjects_ts)})")

if __name__ == "__main__":
    main()
