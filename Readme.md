# PD-MCI Forecast (rs-fMRI)

Progression-oriented neuroAI pilot: classify **Parkinson’s disease with normal cognition (PD‑NC)** vs **Parkinson’s disease with mild cognitive impairment (PD‑MCI)** from **resting‑state fMRI** functional connectivity. The pipeline is open, reproducible, and tuned for **small‑N** cohorts.

---

## TL;DR (current primary result)
- **Features:** rs‑fMRI **correlation** connectomes (Schaefer atlas), **GSR on**, fMRIPrep confounds + **scrubbing**
- **Model:** Elastic‑net logistic regression, **nested CV**, probability **calibration**
- **Performance (OOF, primary config “Step‑1”)**  
  AUROC ≈ **0.75** (95% CI ~ **0.55–0.91**), AUPRC ≈ **0.68**, Balanced Accuracy ≈ **0.71**, ECE ≈ **0.12**  
  Threshold (optimized on OOF for BA): **t\* = 0.612** → Sens 0.64, Spec 0.81, BA 0.73, Acc 0.73, F1 0.69
- **Cohort:** N≈30 PD subjects (PD‑NC / PD‑MCI) after QC
- **Notes:** small‑sample pilot → wide CIs; motion leakage checked; ablations (PCA, different atlas sizes) included.

---

## Repository structure
```
.
├─ configs/
│  └─ config.yaml                # paths + extraction settings
├─ outputs/                      # artifacts (created after runs)
│  ├─ connectomes.npy            # features
│  ├─ labels.csv                 # labels kept after extraction
│  ├─ meta.json                  # extraction metadata
│  ├─ preds_cv.csv               # out-of-fold predictions
│  ├─ ts_en_summary.json         # metrics summary
│  ├─ ts_en_model*.joblib        # trained models (calibrated & raw)
│  ├─ ts_en_threshold.txt        # chosen operating threshold t*
│  └─ figures *.png              # ROC/PR/calibration/CM/threshold sweep
├─ scripts/
│  ├─ qc_filter.py               # keeps subjects based on confounds FD + length
│  ├─ extract_connectomes.py     # robust BOLD+confounds → time series → FC
│  ├─ train_ts_en.py             # nested CV + calibration (+ optional PCA)
│  ├─ choose_threshold.py        # pick t* from OOF predictions
│  ├─ perm_test.py               # permutation test for AUROC (optional)
│  └─ predict_new.py             # apply trained model to new data
└─ README.md                     # this file
```

---

## 1) Environment setup (Windows + optional WSL)

### 1.1 Create Python env (Windows PowerShell)
```powershell
# Python 3.10+ recommended
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt   # or install packages below
```
**Minimal packages** if you don’t have `requirements.txt` yet:
```
numpy pandas scipy scikit-learn matplotlib nibabel nilearn pyyaml joblib
```

> **Tip (Windows plotting)**: set a non‑GUI backend to avoid Tk errors during parallel CV:  
> `setx MPLBACKEND Agg` (PowerShell) or export `MPLBACKEND=Agg` for the session.

### 1.2 (Optional) WSL + Docker for fMRIPrep
- Install **WSL2** and **Docker Desktop** (enable WSL2 backend).
- Get a **FreeSurfer license** (`license.txt`) and place it somewhere stable.

---

## 2) Data
- Source: OpenNeuro dataset **ds005892** (PD‑NC, PD‑MCI, HC).  
- BIDS root example (Windows): `C:/Users/<your path>`

> We analyze **PD only** for the primary task (PD‑NC vs PD‑MCI). HC can be used for normative checks, but is **not** part of the primary labels.

---

## 3) Preprocessing with fMRIPrep (Docker)
> Aim: preprocessed BOLD in **MNI152NLin2009cAsym**, confounds TSV, **skip FreeSurfer recon** (faster), **no CIFTI**.

**Create output folders**
```powershell
# Windows paths shown; mirror under /mnt/c/... if running via WSL bash
mkdir C:\...\derivatives
mkdir C:\...\workdir
Copy-Item C:\path\to\license.txt C:\...\license.txt
```

**WSL bash command (example)**
```bash
# Edit paths for your system
BIDS=/mnt/c/Users/<your_path>
DERIV=/mnt/c/Users/<your_path>
WORK=/mnt/c/Users/<your_path>
FS=/mnt/c/Users/<your_path>

sudo docker run --rm -it \
  -v $BIDS:/data:ro \
  -v $DERIV:/out \
  -v $WORK:/work \
  -v $FS:/opt/freesurfer/license.txt:ro \
  nipreps/fmriprep:23.2.0 \
    /data /out participant \
    --participant-label sub-MJF001 sub-MJF002 ... \
    --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --use-syn-sdc false \
    --nthreads 6 --omp-nthreads 2 --mem 12GB --work-dir /work
```
**Notes**
- Even with `--fs-no-reconall`, fMRIPrep expects a **license** file.
- For speed: run a few participants per call (parallelize across subjects).

Outputs you need (under `derivatives/sub-*/func/`):
- `..._space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`
- `..._desc-confounds_timeseries.tsv` (+ sidecar JSON with TR)

---

## 4) QC gate (framewise displacement + minimum timepoints)
Prepare a labels file (CSV) listing subjects and groups (PD‑NC / PD‑MCI). Then run:

```powershell
python .\scripts\qc_filter.py `
  --bids  "C:\...\pd" `
  --deriv "C:\...\derivatives" `
  --labels .\outputs\labels_raw.csv `
  --out    .\outputs\labels.csv `
  --fd-thresh 0.35 `
  --min-tr 90 `
  --debug
```
- Drops subjects with **no confounds** or **mean FD > threshold** or **< min TRs**.
- We later **scrub** high‑motion frames again during extraction.

> In our run, 2 high‑motion PD subjects were dropped; N≈30 PD retained.

---

## 5) Feature extraction: time series → FC
Configure `configs/config.yaml` (example):
```yaml
# paths
bids_root: "C:/.../pd"
deriv:     "C:/.../derivatives"
labels_csv: "outputs/labels.csv"

# what to extract
atlas_name: "schaefer"
n_parcels: 200              # try 100 for smaller N
fc_kind: "correlation"      # robust; no tangent leakage
space: "MNI152NLin2009cAsym"
task: "rest"
desc: "preproc"

# denoising
use_confounds: true
global_signal_regression: true
filter: {high_pass: 0.01, low_pass: 0.1, t_r: null}

# scrubbing (FD in confounds)
scrub:
  enabled: true
  fd_thresh: 0.5
  n_before: 1
  n_after: 1

# demand at least N timepoints **after** scrubbing
min_timepoints: 80
```
Run extraction:
```powershell
python .\scripts\extract_connectomes.py --config .\configs\config.yaml --debug
```
Artifacts:
- `outputs/connectomes.npy` (subjects × FC‑vector)
- `outputs/labels.csv` (subjects kept; PD‑NC/PD‑MCI)  
- `outputs/meta.json`, `outputs/timeseries/*.npy`

**If you see “No subjects passed extraction”:**
- Check filename patterns (we handle `_res-2_`, sessions, `.tsv.gz`),
- Relax `min_timepoints` temporarily to diagnose.

---

## 6) Model training (nested CV + calibration)
Train the elastic‑net LR with nested CV; saves OOF predictions and plots.

```powershell
$env:MPLBACKEND="Agg"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"

python .\scripts\train_ts_en.py `
  --features .\outputs\connectomes.npy `
  --labels   .\outputs\labels.csv `
  --meta     .\outputs\meta.json `
  --outdir   .\outputs `
  --calibration sigmoid `            # or isotonic
  --grid-C 0.1 1.0 `
  --grid-l1 0.0 0.5 `
  --pca-components 0.95              # optional: keep 95% variance (per-fold)
```
Outputs:
- `ts_en_summary.json`: AUROC, AUPRC, BA, ECE (+ CIs)
- `preds_cv.csv`: OOF subject‑level predictions (id, y_true, p_hat, fold)
- Plots: ROC, PR, calibration, confusion matrices
- Models: `ts_en_model.joblib` (uncalibrated), `ts_en_model_calibrated.joblib`

**Recommended small‑N defaults**
- **Atlas size:** start with **Schaefer‑100** (4,950 edges) if N is very small
- **Calibration:** `sigmoid` (more stable than isotonic at small N)
- **PCA:** keep variance (e.g., `0.95`) or fixed k (e.g., `25`); our `SafePCA` caps to ≤ n_train‑1 per fold.

---

## 7) Pick an operating threshold from OOF predictions
Choose a threshold **once** from OOF preds to maximize **Balanced Accuracy** (or F1 / Youden’s J). This avoids test leakage.

```powershell
python .\scripts\choose_threshold.py `
  --preds outputs\preds_cv.csv `
  --outdir outputs `
  --metric BA
```
Writes:
- `outputs/ts_en_threshold.txt` with **t\***  
- `outputs/threshold_report.json` with Sens/Spec/BA/Acc/Precision/F1 at t\* and BA 95% CI  
- `outputs/threshold_sweep_ba.png`

> **Our primary model (Step‑1):** t\* = **0.612**, Sens 0.64, Spec 0.81, BA 0.73, Acc 0.73.

---

## 8) Optional: permutation test (AUROC > chance?)
```powershell
python .\scripts\perm_test.py `
  --features outputs\connectomes.npy `
  --labels   outputs\labels.csv `
  --n_perms  200 `
  --n_jobs   -1
```
Saves `outputs/perm_test.json` with observed AUROC, null mean, and p‑value.

---

## 9) Apply the trained model to new subjects
```powershell
python .\scripts\predict_new.py `
  --model      outputs\ts_en_model_calibrated.joblib `
  --threshold  outputs\ts_en_threshold.txt `
  --features   path\to\new\connectomes.npy `
  --ids        path\to\new\ids.csv `
  --out        outputs\preds_new.csv
```

---

## 10) Reproducibility checklist
- [ ] Commit the exact **`configs/config.yaml`** used
- [ ] Pin package versions (`pip freeze > requirements.txt`)
- [ ] Save `ts_en_summary.json`, `preds_cv.csv`, `threshold_report.json`
- [ ] Record N, class counts, FD thresholds, min timepoints, scrubbing
- [ ] Log fMRIPrep image version + CLI flags in README

---

## 11) Troubleshooting
- **Tkinter / Tcl_AsyncDelete errors (Windows):** set `MPLBACKEND=Agg` (we force this in training).
- **Docker permission denied:** add your user to `docker-users` (Windows) or `sudo docker ...` under WSL.
- **fMRIPrep needs license:** mount `license.txt` to `/opt/freesurfer/license.txt`.
- **“No subjects passed extraction”:** verify `deriv` path points to the folder that **directly** contains `sub-*`; relax `min_timepoints` to diagnose; ensure patterns include `_res-2_`.
- **All “no_confounds”:** path bug → scripts should build paths from **`deriv` only** (fixed in repo); verify `.tsv`/`.tsv.gz` both allowed.
- **Training very slow:** set BLAS threads to 1; shrink grid; use Schaefer‑100 or PCA.

---

## 12) Interpretation, limits, ethics
- Pilot‑scale rs‑fMRI study (N≈30 PD) → **uncertain estimates**; replicate on larger cohorts (PPMI, etc.) when access permits.
- Report **AUROC + CI** (threshold‑free) and **thresholded metrics at t\*** (clinically interpretable); include a **permutation test**.
- Motion leakage controlled with confounds + scrubbing + GSR; we verify corr[p̂, mean FD] ≈ 0.
- Predictions are **not** diagnostic; use for research.

---

## 13) Cite tools & data
- fMRIPrep (nipreps), Nilearn, scikit‑learn, Schaefer 2018 atlas, OpenNeuro ds005892.

---

## 14) Quick commands (copy‑paste)

**Extract**
```powershell
python .\scripts\extract_connectomes.py --config .\configs\config.yaml
```
**Train**
```powershell
$env:MPLBACKEND="Agg"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python .\scripts\train_ts_en.py --features outputs\connectomes.npy --labels outputs\labels.csv --meta outputs\meta.json --outdir outputs --calibration sigmoid --grid-C 0.1 1.0 --grid-l1 0.0 0.5
```
**Choose threshold**
```powershell
python .\scripts\choose_threshold.py --preds outputs\preds_cv.csv --outdir outputs --metric BA
```
**(Optional) Permutation**
```powershell
python .\scripts\perm_test.py --features outputs\connectomes.npy --labels outputs\labels.csv --n_perms 200 --n_jobs -1
```
**Predict new**
```powershell
python .\scripts\predict_new.py --model outputs\ts_en_model_calibrated.joblib --threshold outputs\ts_en_threshold.txt --features path\to\new\connectomes.npy --ids path\to\new\ids.csv --out outputs\preds_new.csv
```

---

**Contact / Issues**: please open a GitHub issue with logs (`ts_en_summary.json`, errors) and environment details (`pip freeze`).

