# Streamlit WESAD Generator Dashboard
# ---------------------------------------------------------------
# Run with:  streamlit run app.py
# ---------------------------------------------------------------

import io
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

# --- your repo imports (expects this file to live anywhere inside the repo) ---
# If app.py sits at the repo root, this works out of the box. Otherwise, set REPO manually below.
import sys
REPO = None
here = Path.cwd()
for p in (here, *here.parents):
    if (p / "src").exists():
        REPO = p
        break
assert REPO is not None, "Couldn't find repo root (folder with src/). Place app.py somewhere inside your repo."
sys.path.insert(0, str(REPO / "src"))

from generate.core import WESADGenerator, _denorm  # type: ignore
from evaluation.wesad_real import RealWESADPreparer  # type: ignore
from evaluation.calibration import WESADCalibrator  # type: ignore

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="WESAD Generator",
    page_icon="💓",
    layout="wide",
)

# ----------------------------
# Constants & quick helpers
# ----------------------------
FS_ECG = 175.0
FS_LOW = 4.0
ORDER = ["baseline", "stress", "amusement"]
LAB2IDX = {"baseline": 0, "stress": 1, "amusement": 2}
IDX2LAB = {v: k for k, v in LAB2IDX.items()}
WESAD_IDS = {"baseline": 1, "stress": 2, "amusement": 3}  # ids in the raw labels

# Defaults – tweak for your machine if needed
DEFAULT_MILESTONES = REPO / "results/checkpoints/diffusion/milestones"
DEFAULT_CKPT = REPO / "results/checkpoints/diffusion/ckpt_epoch_130_WEIGHTS.pt"
DEFAULT_FOLD_DIR = REPO / "data/processed/two_stream/fold_S10"
DEFAULT_CAL_JSON = REPO / "results/evaluation/eval_ckpt136/calibration_targets.json"
DEFAULT_NORM_LOW = DEFAULT_FOLD_DIR / "norm_low.npz"
DEFAULT_NORM_ECG = DEFAULT_FOLD_DIR / "norm_ecg.npz"

# ----------------------------
# Caching layers
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_generator(milestones: Path, ckpt: Path) -> WESADGenerator:
    gen = WESADGenerator(milestones_dir=str(milestones), ckpt_path=str(ckpt))
    return gen

@st.cache_resource(show_spinner=False)
def load_calibrator(calibration_json: Path) -> WESADCalibrator:
    return WESADCalibrator.load(str(calibration_json))

@st.cache_data(show_spinner=False)
def load_real_by_condition(fold_dir: Path) -> Dict[str, np.ndarray]:
    prep = RealWESADPreparer(str(fold_dir))
    X, y = prep.prepare(target="ecg")  # (N, 5250, 3)
    assert y is not None and len(X) == len(y)
    keep = np.isin(y, list(WESAD_IDS.values()))
    X, y = X[keep], y[keep]
    out: Dict[str, np.ndarray] = {}
    for name in ORDER:
        cls = WESAD_IDS[name]
        out[name] = X[y == cls].astype(np.float32, copy=False)
    return out

@st.cache_data(show_spinner=False)
def real_refs_psd_acf(real_by_cond: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Return per-condition reference spectra and ACFs.
    dict[cond][ch] → (mean_psd_vector, mean_acf_vector)
    Channels named as 'ECG','Resp','EDA'.
    """
    def _mean_psd(x: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        psds = []
        for i in range(x.shape[0]):
            f, p = welch(x[i], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
            psds.append(p)
        psds = np.stack(psds, axis=0)
        return f, psds.mean(0), psds.std(0)

    def _mean_acf(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
        acfs = []
        for i in range(x.shape[0]):
            s = x[i] - x[i].mean()
            denom = (s * s).sum()
            if denom <= 1e-12:
                acfs.append(np.zeros(max_lag + 1, dtype=np.float32))
                continue
            c = np.correlate(s, s, mode="full")
            mid = len(c) // 2
            a = c[mid : mid + max_lag + 1] / denom
            acfs.append(a.astype(np.float32))
        return np.stack(acfs, axis=0).mean(0)

    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for cond, X in real_by_cond.items():
        out[cond] = {}
        for ch_idx, ch_name in enumerate(["ECG", "Resp", "EDA"]):
            fs = FS_ECG  # arrays are fused to ECG rate
            nperseg = 1024 if ch_name == "ECG" else 128
            f, mpsd, _ = _mean_psd(X[..., ch_idx], fs=fs, nperseg=nperseg)
            acf = _mean_acf(X[..., ch_idx])
            # trim low-streams to <1.5 Hz for respiratory/EDA when comparing
            if ch_name in ("Resp", "EDA"):
                mask = f <= 1.5
                out[cond][ch_name] = (mpsd[mask], acf)
            else:
                out[cond][ch_name] = (mpsd, acf)
    return out

# ----------------------------
# Sampling utility (ECG+LOW heads → fused [ECG,Resp,EDA])
# ----------------------------
def synth_match_real_fast(
    gen: WESADGenerator,
    condition: str,
    T: int,
    N: int,
    *,
    steps_ecg: int = 150,
    steps_low: int = 150,
    guidance_ecg: float = 0.5,
    guidance_low: float = 0.1,
    norm_low_path: Path = DEFAULT_NORM_LOW,
    norm_ecg_path: Path = DEFAULT_NORM_ECG,
) -> np.ndarray:
    import torch

    label_idx = LAB2IDX[condition]
    K = int(getattr(gen, "condition_dim", gen.bundle.manifest.get("condition_dim", 3)))

    # Load norms
    nl = np.load(str(norm_low_path), allow_pickle=False)
    ne = np.load(str(norm_ecg_path), allow_pickle=False)

    out = np.empty((N, T, 3), dtype=np.float32)
    device = gen.device

    for i in range(0, N, 16):
        b = min(16, N - i)
        cond = torch.zeros(b, K, device=device, dtype=torch.float32)
        cond[:, label_idx] = 1.0
        with torch.no_grad():
            x_low = gen.low.sample(cond, num_steps=int(steps_low), method="ddim", cfg_scale=float(guidance_low))
            x_ecg = gen.ecg.sample(cond, num_steps=int(steps_ecg), method="ddim", cfg_scale=float(guidance_ecg))
        x_low = _denorm(x_low.cpu().numpy(), nl)
        x_ecg = _denorm(x_ecg.cpu().numpy(), ne)

        # map low-stream to [Resp, EDA]
        mapped = False
        if "channels" in nl.files:
            try:
                chs = [str(c).lower() for c in nl["channels"].tolist()]
                resp_idx = chs.index("resp") if "resp" in chs else chs.index("respiration")
                eda_idx = chs.index("eda") if "eda" in chs else chs.index("electrodermal activity")
                x_low = x_low[..., [resp_idx, eda_idx]]
                mapped = True
            except Exception:
                mapped = False
        if not mapped:
            # heuristic by std
            stds = x_low.std(axis=(0, 1))
            idx_small = int(np.argmin(stds))  # Resp
            idx_large = 1 - idx_small  # EDA
            x_low = x_low[..., [idx_small, idx_large]]
        x_low[..., 1] = np.clip(x_low[..., 1], 0.0, None)  # EDA ≥ 0

        # upsample low to ECG length and fuse
        N_b, L_low, _ = x_low.shape
        L_ecg = x_ecg.shape[1]
        xp = np.linspace(0, 1, L_low)
        xq = np.linspace(0, 1, L_ecg)
        low_up = np.empty((N_b, L_ecg, 2), np.float32)
        for n in range(N_b):
            for c in range(2):
                low_up[n, :, c] = np.interp(xq, xp, x_low[n, :, c])
        fused = np.concatenate([x_ecg.astype(np.float32), low_up], axis=-1)
        out[i : i + b] = fused[:, :T]
    return out

# --- cached batch generator + session-state save ---
@st.cache_data(show_spinner=False)
def _generate_batch_cached(
    *, _gen: WESADGenerator, cond: str, N: int, T_target: int, seed: int,
    steps_ecg: int, steps_low: int, g_ecg: float, g_low: float,
    norm_low_p: Path, norm_ecg_p: Path, do_cal: bool, cal_json_p: Path
) -> np.ndarray:
    import torch
    gen = _gen 
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = synth_match_real_fast(
        gen, condition=cond, T=T_target, N=N,
        steps_ecg=steps_ecg, steps_low=steps_low,
        guidance_ecg=g_ecg, guidance_low=g_low,
        norm_low_path=norm_low_p, norm_ecg_path=norm_ecg_p,
    )
    if do_cal and cal_json_p.exists():
        cal = load_calibrator(cal_json_p)
        X = cal.apply(
            X, do_ecg=True, do_resp=True, do_eda=True,
            ecg_qmap=cal.has_ecg_qmap(), ecg_qmap_alpha=0.7, enforce_resp_std=True
        )
    return X


# ----------------------------
# Quick metrics against real references (sanity badge)
# ----------------------------

def quick_sanity_scores(X: np.ndarray, ref: Dict[str, Tuple[np.ndarray, np.ndarray]], fs: float = FS_ECG) -> Dict[str, float]:
    """Compute PSD & ACF similarity (corr of means) for each channel and return averages."""
    def _mean_psd(x: np.ndarray, fs: float, nperseg: int):
        psds = []
        for i in range(x.shape[0]):
            f, p = welch(x[i], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
            psds.append(p)
        return f, np.stack(psds).mean(0)

    ch_names = ["ECG", "Resp", "EDA"]
    psd_corrs, acf_corrs = [], []
    for ch_idx, ch in enumerate(ch_names):
        fs_loc = fs
        nperseg = 1024 if ch == "ECG" else 128
        f, mpsd = _mean_psd(X[..., ch_idx], fs_loc, nperseg)
        if ch in ("Resp", "EDA"):
            mask = f <= 1.5
            mpsd = mpsd[mask]
        ref_psd, ref_acf = ref[ch]
        psd_corrs.append(float(np.corrcoef(ref_psd, mpsd)[0, 1]))
        # ACF (mean over windows)
        # compute a small-lag acf
        def _mean_acf(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
            acfs = []
            for i in range(x.shape[0]):
                s = x[i] - x[i].mean()
                denom = (s * s).sum()
                if denom <= 1e-12:
                    acfs.append(np.zeros(max_lag + 1, dtype=np.float32))
                    continue
                c = np.correlate(s, s, mode="full")
                mid = len(c) // 2
                a = c[mid : mid + max_lag + 1] / denom
                acfs.append(a.astype(np.float32))
            return np.stack(acfs, axis=0).mean(0)
        macf = _mean_acf(X[..., ch_idx])
        acf_corrs.append(float(np.corrcoef(ref_acf, macf)[0, 1]))
    return {
        "PSD_sim": float(np.mean(psd_corrs)),
        "ACF_sim": float(np.mean(acf_corrs)),
    }

# ----------------------------
# Sidebar — Controls
# ----------------------------
st.sidebar.header("Inputs")

# Paths (collapsible, advanced)
with st.sidebar.expander("Paths & assets", expanded=False):
    milestones_p = Path(st.text_input("Milestones dir", str(DEFAULT_MILESTONES)))
    ckpt_p = Path(st.text_input("Checkpoint (.pt)", str(DEFAULT_CKPT)))
    fold_dir = Path(st.text_input("Fold dir (real windows)", str(DEFAULT_FOLD_DIR)))
    norm_low_p = Path(st.text_input("norm_low.npz", str(DEFAULT_NORM_LOW)))
    norm_ecg_p = Path(st.text_input("norm_ecg.npz", str(DEFAULT_NORM_ECG)))
    cal_json_p = Path(st.text_input("Calibration targets (optional)", str(DEFAULT_CAL_JSON)))

cond = st.sidebar.selectbox("Condition", ORDER, index=0)
N = int(st.sidebar.number_input("# windows", min_value=1, max_value=256, value=16, step=1))
seed = int(st.sidebar.number_input("Seed", min_value=0, value=42, step=1))

with st.sidebar.expander("Sampling (advanced)", expanded=False):
    steps_ecg = int(st.number_input("ECG steps", min_value=10, max_value=500, value=150, step=10))
    steps_low = int(st.number_input("LOW steps", min_value=10, max_value=500, value=150, step=10))
    g_ecg = float(st.slider("ECG guidance", 0.0, 3.0, 0.5, 0.05))
    g_low = float(st.slider("LOW guidance", 0.0, 3.0, 0.1, 0.05))
    do_cal = st.checkbox("Calibrate to real (recommended)", value=True)

st.sidebar.write("")
clicked = st.sidebar.button("Generate", type="primary")

# ----------------------------
# Main — Load assets
# ----------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with st.spinner("Loading generator & real references …"):
    gen = load_generator(milestones_p, ckpt_p)
    np.random.seed(seed)
    try:
        real_by_cond = load_real_by_condition(fold_dir)
    except Exception as e:
        st.error(f"Failed to load real windows from {fold_dir}: {e}")
        st.stop()
    refs = real_refs_psd_acf(real_by_cond)

# ----------------------------
# Generate on click
# ----------------------------
if clicked:
    try:
        T_target = int(real_by_cond[cond].shape[1])
        with st.spinner("Generating.."):
            X_syn = _generate_batch_cached(
                _gen=gen, cond=cond, N=N, T_target=T_target, seed=seed,
                steps_ecg=steps_ecg, steps_low=steps_low, g_ecg=g_ecg, g_low=g_low,
                norm_low_p=norm_low_p, norm_ecg_p=norm_ecg_p,
                do_cal=do_cal, cal_json_p=cal_json_p,
            )
        st.session_state["X_synth"] = X_syn
        st.session_state["gen_meta"] = dict(
            cond=cond, N=N, seed=seed, T_target=T_target,
            steps_ecg=steps_ecg, steps_low=steps_low,
            g_ecg=g_ecg, g_low=g_low, do_cal=do_cal,
            norm_low=str(norm_low_p), norm_ecg=str(norm_ecg_p),
            cal_json=str(cal_json_p),
        )
    except Exception as e:
        st.error(f"Generation failed: {e}")

# ----------------------------
# Always render preview/download from session
# ----------------------------
X_sess = st.session_state.get("X_synth")
meta   = st.session_state.get("gen_meta")

if X_sess is None or meta is None:
    st.info("Set parameters and click **Generate** to create samples.")
    st.stop()

# Warn if sidebar params differ from the stored batch
if (meta["cond"] != cond) or (meta["N"] != N) or (meta["seed"] != seed):
    st.warning(
        f"Showing batch generated for cond **{meta['cond']}**, N={meta['N']}, seed={meta['seed']}. "
        "Sidebar changed — click **Generate** to refresh."
    )

T_target = meta["T_target"]
N_gen    = int(X_sess.shape[0])
FS       = FS_ECG

# --- Quick sanity badge
scores = quick_sanity_scores(X_sess, refs[meta["cond"]])
with col_left:
    st.subheader("Quick sanity (vs real reference)")

    # show as % (friendlier in a dashboard), or raw correlation
    use_pct = st.toggle("Show as %", value=True, help="Display similarities as percentages")
    fmt = (lambda v: f"{100*v:.1f}%") if use_pct else (lambda v: f"{v:.3f}")

    c1, c2 = st.columns(2)
    c1.metric("PSD similarity", fmt(scores["PSD_sim"]))
    c2.metric("ACF similarity", fmt(scores["ACF_sim"]))

    # "Dropdown" details right below the metrics
    with st.expander("Details by channel", expanded=False):
        import pandas as pd
        from scipy.signal import welch

        ch_names = ["ECG", "Resp", "EDA"]

        def _mean_psd(x, fs, nperseg):
            psds = []
            for i in range(x.shape[0]):
                f, p = welch(x[i], fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend="constant")
                psds.append(p)
            return f, np.stack(psds, axis=0).mean(0)

        def _mean_acf(x, max_lag=100):
            acfs = []
            for i in range(x.shape[0]):
                s = x[i] - x[i].mean()
                den = (s*s).sum()
                if den <= 1e-12:
                    acfs.append(np.zeros(max_lag+1, np.float32)); continue
                c = np.correlate(s, s, mode="full"); mid = len(c)//2
                acfs.append((c[mid:mid+max_lag+1]/den).astype(np.float32))
            return np.stack(acfs, axis=0).mean(0)

        psd_vals, acf_vals = [], []
        for ci, ch in enumerate(ch_names):
            nperseg = 1024 if ch == "ECG" else 128
            f, mpsd = _mean_psd(X_sess[..., ci], FS, nperseg)
            if ch in ("Resp", "EDA"):
                mpsd = mpsd[f <= 1.5]
            ref_psd, ref_acf = refs[meta["cond"]][ch]
            psd_corr = float(np.corrcoef(ref_psd, mpsd)[0, 1])
            acf_corr = float(np.corrcoef(ref_acf, _mean_acf(X_sess[..., ci]))[0, 1])
            psd_vals.append(psd_corr); acf_vals.append(acf_corr)

        df = pd.DataFrame({"channel": ch_names, "PSD_sim": psd_vals, "ACF_sim": acf_vals})
        if use_pct:
            df_show = df.copy()
            df_show["PSD_sim"] = (100*df_show["PSD_sim"]).map(lambda v: f"{v:.1f}%")
            df_show["ACF_sim"] = (100*df_show["ACF_sim"]).map(lambda v: f"{v:.1f}%")
        else:
            df_show = df.round(3)

        st.dataframe(df_show, use_container_width=True)

# --- Plot
with col_right:
    st.subheader("Preview")
    idx = int(st.number_input("Preview window index", min_value=0, max_value=max(0, N_gen - 1), value=0))
    seconds = float(st.slider("Seconds to display", 2.0, min(12.0, T_target / FS), 8.0, 0.5))
    L = int(round(seconds * FS))
    s = max(0, (T_target - L) // 2); e = s + L
    t = np.arange(L) / FS

    xr = real_by_cond[meta["cond"]][0]    # single real example for overlay
    xs = X_sess[idx]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.2), sharex=True)
    names = ["ECG", "Resp", "EDA"]
    for i, name in enumerate(names):
        axes[i].plot(t, xr[s:e, i], color="#7f7f7f", lw=1.0, alpha=0.6, label="Real (example)")
        axes[i].plot(t, xs[s:e, i], color="#1f77b4", lw=1.4, label="Synth")
        axes[i].set_ylabel(name); axes[i].grid(alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)

# --- Downloads (CSV, NPZ) from stored batch
st.subheader("Download")
t_full = (np.arange(T_target) / FS).astype(np.float32)
window_ids = np.repeat(np.arange(N_gen, dtype=np.int32), T_target)
cond_col = np.full(N_gen * T_target, LAB2IDX[meta["cond"]], dtype=np.int32)
df = pd.DataFrame({
    "time": np.tile(t_full, N_gen),
    "ECG": X_sess[..., 0].reshape(-1),
    "Resp": X_sess[..., 1].reshape(-1),
    "EDA": X_sess[..., 2].reshape(-1),
    "condition": cond_col,
    "window_id": window_ids,
})
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes,
                   file_name=f"synth_{meta['cond']}_N{N_gen}.csv", mime="text/csv")

buf = io.BytesIO()
np.savez_compressed(
    buf,
    signals=X_sess.astype(np.float32, copy=False),
    channels=np.array(["ECG","Resp","EDA"], dtype=object),
    labels=np.full(N_gen, LAB2IDX[meta["cond"]], dtype=np.int32),
    condition=np.array([meta["cond"]]),
)
st.download_button("Download NPZ", data=buf.getvalue(),
                   file_name=f"synth_{meta['cond']}_N{N_gen}.npz", mime="application/octet-stream")

# ----------------------------
# Footer
# ----------------------------
st.caption(
    "This dashboard provides quick, non-exhaustive sanity checks (PSD/ACF similarities) and easy export. "
    "For full evaluation (Table 1/2), use your wesad_eval pipeline."
)
