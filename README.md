# WESAD Two-Stream Diffusion — Synthetic Physiological Signals

Generate realistic **ECG, Respiration, and EDA** windows conditioned on affective states from the **WESAD** dataset using a **two-stream diffusion** model (ECG @ 175 Hz, EDA/Resp @ 4 Hz). Includes a **Streamlit** app for interactive sampling and quick sanity checks, plus an optional **Docker** workflow so others can run it without local setup.

---

## TL;DR (Quickstart)

- Build the app image (from repo root):
  ```bash
  docker build -t wesad-app .
  ```

- Run the app — the container **downloads the inference bundle** from your GitHub Release:
  ```bash
  docker run --rm -p 8501:8501     -e WEIGHTS_ZIP_URL="https://github.com/jomoregie1/generative-health-models/releases/download/v1.0.0/inference_bundle.zip"     -e BUNDLE_SHA256="E68E9B2F2B32EC103B4F76E192EF5482E711A1EBA8B7E8EBD18F205DE3EF2811"     wesad-app
  ```

- Open **http://localhost:8501**.  
  Inside the container, defaults are:
  - Milestones dir: `/app/results/checkpoints/diffusion/milestones`
  - Checkpoint: `/app/results/checkpoints/diffusion/ckpt_epoch_130_WEIGHTS.pt`
  - Fold dir: `/app/data/processed/two_stream/fold_S10`

---

## 1) Overview

This repo implements a **two-branch conditional diffusion** model that synthesizes:
- **LOW stream:** EDA & Respiration at **4 Hz** (60 s windows → 120 steps)
- **ECG stream:** ECG at **175 Hz** (60 s windows → 5,250 steps)

The model is trained in a **subject-independent** fashion (LOSO) and deployed via a small **adapter** so both heads present a single “generator-like” interface.

---

## 2) Key Features

- **Two-stream diffusion** (separate UNets) with per-head sampling controls.
- **WESAD preprocessing** → fused arrays with channels **[ECG, Resp, EDA]**.
- **Conditioning** on affective states (baseline, stress, amusement; meditation optional).
- **Train-only normalization** saved as `norm_low.npz` / `norm_ecg.npz`.
- **Streamlit dashboard** for generation, quick PSD/ACF sanity metrics, and exports.
- **Docker** packaging so anyone can run the app without installing Python.

---

## 3) Repository Layout

```
src/
  app.py                         # Streamlit app (entry point)
  models/
    diffusion.py                 # diffusion heads
    diffusion_adapter.py         # two-stream generator adapter
  generate/
    core.py                      # WESADGenerator (loads checkpoint+milestone)
  evaluation/
    wesad_real.py                # RealWESADPreparer (loads/denorms/test split)
    calibration.py               # (optional) calibration utilities
  datasets/
    wesad.py                     # PyTorch Dataset + DataLoader (training)
Dockerfile
.dockerignore
requirements.txt
```

---

## 4) Requirements

- **Docker Desktop** (recommended), or Python **3.10/3.11** with:
  - `torch` (CPU wheels OK), `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`, `streamlit`
- Artifacts for inference (**not in git**): see next section.

---

## 5) Inference Bundle (what the app needs)

If you use auto-download, the bundle is fetched at runtime from your **Release**. For offline use, provide these files (zip or mount):

```
inference_bundle/
  results/checkpoints/diffusion/
    ckpt_epoch_XXX_WEIGHTS.pt               # chosen generator weights
    milestones/
      milestone_eXXX_....json               # manifest for that checkpoint
      milestone_eXXX_....pt                 # paired milestone weights (EMA copy)
      norm_low.npz
      norm_ecg.npz
  data/processed/two_stream/fold_S10/
    test_X_low.npy
    test_X_ecg.npy
    test_cond.npy            # optional; labels for display
    norm_low.npz
    norm_ecg.npz
    dataset_config.json
```

> **Why the fold?** The app uses a few real windows for PSD/ACF references and for channel stats; generation itself only needs the checkpoint+milestones.

---

## 6) Preprocessing (if you plan to re-create the fold)

- Preprocess raw WESAD pickles into two streams (LOW @4 Hz, ECG @175 Hz).
- Save train/test windows and **train-only** normalization (`norm_*.npz`).
- The **fold directory** must contain the files listed above.

*(If you only want to run the app, you just need the already-processed fold.)*

---

## 7) Model & Training (high-level)

- Two diffusion heads with the same condition vector:
  - **LOW head** consumes a per-timestep one-hot `(B, 120, K)`.
  - **ECG head** consumes a pooled one-hot `(B, K)`.
- Objective: sum of head losses with λ weights.
- **EMA** maintained for both heads; sampling uses EMA by default.
- Scheduler: either Cosine or ReduceLROnPlateau (validation-driven).
- LOSO evaluation across subjects (optional for the thesis; case-study also fine).

---

## 8) Evaluation (optional components)

- **Distribution metrics:** KS, Wasserstein, JSD (per channel), PSD/ACF similarity.
- **Linear probe** (frozen features) for label separability.
- **Leakage checks** (NN/Mahalanobis distances in simple feature spaces).

*(The app shows lightweight PSD/ACF correlations; full evaluation scripts are optional and not required to run the app.)*

---

## 9) Streamlit App (what it does)

- Loads your **checkpoint + milestone manifest** and **norm files**.
- Samples a batch for the selected **condition** and **steps/guidance**.
- **Fuses** outputs to target length (`T = 5250` or `T = 120`) and orders channels `[ECG, Resp, EDA]`.
- Displays **PSD/ACF similarity** vs. real references and lets you **download** CSV/NPZ.

---

## 10) How to Run the Application

### A) Docker (auto-download from Release) — **recommended**
```bash
docker run --rm -p 8501:8501   -e WEIGHTS_ZIP_URL="https://github.com/jomoregie1/generative-health-models/releases/download/v1.0.0/inference_bundle.zip"   -e BUNDLE_SHA256="E68E9B2F2B32EC103B4F76E192EF5482E711A1EBA8B7E8EBD18F205DE3EF2811"   wesad-app
```
Open: http://localhost:8501

**Container defaults**
- Milestones: `/app/results/checkpoints/diffusion/milestones`
- Checkpoint: `/app/results/checkpoints/diffusion/ckpt_epoch_130_WEIGHTS.pt`
- Fold dir: `/app/data/processed/two_stream/fold_S10`

### B) Docker (mount local folders) — offline alternative
Unzip the bundle locally and mount:
```bash
docker run --rm -p 8501:8501   -v "/absolute/path/inference_bundle/results/checkpoints/diffusion:/app/results/checkpoints/diffusion:ro"   -v "/absolute/path/inference_bundle/data/processed/two_stream/fold_S10:/app/data/processed/two_stream/fold_S10:ro"   wesad-app
```

### C) Local (without Docker)
1) Create a virtual env (Python 3.11 or 3.10)
2) Install deps (CPU torch OK)
3) `streamlit run src/app.py`  
Then set paths in the sidebar to your local bundle.

---

## 11) Configuration

- **Sampling**
  - ECG steps / LOW steps
  - Guidance (CFG scale)
  - Seed, batch size N
- **Calibration** (optional): enable/disable with a JSON of targets
- **Downloads**: NPZ and CSV export of generated batches

---

## 12) Troubleshooting

- **App starts but can’t see files**: Check that your bundle has top-level `results/` and `data/` folders (no extra parent dir).
- **“File not found” for milestone JSON**: The `.json` must sit next to the milestone `.pt` in the `milestones/` folder.
- **Port busy**: map another host port, e.g., `-p 8502:8501` → open http://localhost:8502.
- **Docker build uploads GBs**: Ensure `.dockerignore` excludes `data/`, `results/`, `*.pt`, `*.npz`, etc.

---

## 13) Security & Data Privacy

WESAD contains physiological data. Do not redistribute raw subject data. This repo only uses derived artifacts (processed windows and learned weights). Generated data is synthetic and should be clearly labeled as such.

---

## 14) Acknowledgements

- Schmidt, P., et al. **WESAD: A Multimodal Dataset for Wearable Stress and Affect Detection**.
- Diffusion model components inspired by recent conditional diffusion literature.
- Thanks to open-source contributors across PyTorch, SciPy, Streamlit.

---

## 15) License

### Code
This repository’s **code** is licensed under the MIT License.  
See [LICENSE](./LICENSE) for details.

### Model weights & generated data
The provided **model weights** and any **generated samples** are licensed under the terms in
[WEIGHTS_LICENSE.txt](./WEIGHTS_LICENSE.txt). In short: research and educational use is permitted; no clinical use or safety-critical deployment. Please read the file for the full terms.

> Note: The WESAD dataset is not redistributed in this repository. Users must obtain and use WESAD under its own license/terms.

---

### Appendix: Minimal `.dockerignore`

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
env/
ENV/
.git/
.DS_Store
.vscode/
.idea/

# Large artifacts kept out of the image (mount at runtime)
data/
results/
runs/
checkpoints/
*.pt
*.pth
*.npz
*.ckpt
*.zip
*.tar
*.gz
```
