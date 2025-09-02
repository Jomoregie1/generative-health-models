"""
Core two-stream WESAD preprocessing for diffusion:
- ECG @ 175 Hz (1 ch)
- EDA & RESP @ 4 Hz (2 ch; order: [EDA, RESP])
- Label masking (±5 s), min valid run 30 s
- Per-stream decimation, EDA high-pass (0.03 Hz)
- Windowing: constant-label 30s windows (50% overlap)
- Train-only z-scoring, save-ready outputs
"""
from pathlib import Path
import re
import pickle
import json
import numpy as np
from collections import Counter
from scipy.signal import butter, sosfiltfilt, decimate

# ---------------- I/O ----------------


def load_wesad_subject(pkl_path):
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("rb") as f:
        try:
            # most reliable for WESAD
            return pickle.load(f, encoding="latin1")
        except Exception:
            f.seek(0)
            return pickle.load(f)  # final fallback
        
        

def build_manifest(wesad_root: Path):
    """Return a list[{'subject': 'S2', 'path': '/abs/S2.pkl'}, ...]"""
    wesad_root = Path(wesad_root)
    pkls = sorted(p for p in wesad_root.rglob("*.pkl") if re.match(r"S\d+\.pkl$", p.name))
    rows = []
    for p in pkls:
        sid = re.match(r"(S\d+)", p.stem).group(1)
        rows.append({"subject": sid, "path": str(p.resolve())})
    return rows

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

# ------------- filters & helpers -------------

def butter_lowpass_zerophase(x, fs, cutoff_hz, order=4):
    nyq = fs * 0.5
    sos = butter(order, cutoff_hz/nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)

def butter_highpass_zerophase(x, fs, cutoff_hz, order=2):
    nyq = fs * 0.5
    sos = butter(order, cutoff_hz/nyq, btype="high", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)

def butter_bandpass_zerophase(x, fs, low_hz, high_hz, order=4):
    nyq = fs * 0.5
    sos = butter(order, [low_hz/nyq, high_hz/nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)

def _impute_linear_series(col):
    col = col.astype(np.float32, copy=False)
    n = col.shape[0]
    mask = np.isfinite(col)
    if mask.all(): return col
    if not mask.any(): return np.zeros_like(col)
    if mask.sum() == 1:
        col[~mask] = float(col[mask][0]); return col
    t = np.arange(n, dtype=np.float32)
    col[~mask] = np.interp(t[~mask], t[mask], col[mask])
    return col

def make_label_mask(labels, fs, valid_labels=(1,2,3,4), transition_pad_s=5.0, min_valid_run_s=30.0):
    y = np.asarray(labels)
    n = y.size
    keep = np.isin(y, np.array(list(valid_labels)))
    pad = int(round(transition_pad_s*fs))
    if pad > 0 and n > 1:
        chg = np.flatnonzero(np.diff(y) != 0)
        if chg.size:
            edge = np.zeros(n, dtype=bool)
            for i in chg:
                lo = max(0, i - pad + 1); hi = min(n, i + pad + 1)
                edge[lo:hi] = True
            keep &= ~edge
    min_len = int(round(min_valid_run_s*fs))
    if min_len > 1:
        i = 0
        while i < n:
            if keep[i]:
                j = i+1
                while j < n and keep[j]: j += 1
                if (j - i) < min_len: keep[i:j] = False
                i = j
            else:
                i += 1
    return keep

def block_mode_downsample(labels, in_len, factor=None, n_out=None):
    y = np.asarray(labels)
    if factor is None and n_out is None:
        raise ValueError("Provide factor or n_out")
    if factor is None: factor = int(np.floor(in_len / n_out))
    n_blocks = int(np.floor(in_len / factor))
    out = np.empty(n_blocks, dtype=y.dtype)
    for i in range(n_blocks):
        block = y[i*factor:(i+1)*factor]
        vals, counts = np.unique(block, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out

# ------------- extraction & per subject -------------

def extract_chest_ecg_eda_resp(subject_data, strict=True):
    """
    Return X (N,3) with columns [ECG, EDA, RESP] and labels (N,)
    """
    chest = subject_data["signal"]["chest"]
    labels = np.asarray(subject_data["label"]).reshape(-1)

    # case-normalize keys
    if "Resp" in chest and "RESP" not in chest: chest["RESP"] = chest.pop("Resp")
    if "Temp" in chest and "TEMP" not in chest: chest["TEMP"] = chest.pop("Temp")

    def col(k):
        x = np.asarray(chest[k])
        if x.ndim == 1: x = x[:,None]
        assert x.ndim == 2 and x.shape[1] == 1
        return x.astype(np.float32, copy=False)

    X = np.concatenate([col("ECG"), col("EDA"), col("RESP")], axis=1)  # (N,3)
    if strict:
        assert X.shape[0] == labels.shape[0]
    else:
        n = min(X.shape[0], labels.shape[0]); X, labels = X[:n], labels[:n]
    return X, labels

def preprocess_single_subject(subject_file, original_rate=700, target_low=4, target_ecg=175,
                              transition_pad_s=5.0, min_valid_run_s=30.0,
                              eda_hp_cutoff=0.03, eda_hp_order=2, verbose=False):
    data = load_wesad_subject(subject_file)
    X, y = extract_chest_ecg_eda_resp(data, strict=True)
    # filters at original fs
    Xf = np.empty_like(X, dtype=np.float32)
    Xf[:,0] = butter_bandpass_zerophase(X[:,0], original_rate, 0.5, 40.0)   # ECG
    Xf[:,1] = butter_lowpass_zerophase( X[:,1], original_rate, 5.0)         # EDA
    Xf[:,2] = butter_bandpass_zerophase(X[:,2], original_rate, 0.1, 0.35)   # RESP

    keep = make_label_mask(y, fs=original_rate, valid_labels=(1,2,3,4),
                           transition_pad_s=transition_pad_s, min_valid_run_s=min_valid_run_s)
    Xk, yk = Xf[keep], y[keep]

    # multi-rate exact alignment (LCM)
    fac_low = original_rate // target_low     # 700/4 = 175
    fac_ecg = original_rate // target_ecg     # 700/175 = 4
    lcm = np.lcm(fac_low, fac_ecg)
    n_keep = (len(yk) // lcm) * lcm
    Xk, yk = Xk[:n_keep], yk[:n_keep]

    # decimate
    Xd_ecg = decimate(Xk[:,0],  fac_ecg, ftype="fir", zero_phase=True)[:,None]
    low    = decimate(Xk[:,1:3], fac_low, ftype="fir",axis=0, zero_phase=True)      # (N_low,2) → [EDA, RESP]

    # EDA HP @ low fs
    if eda_hp_cutoff and eda_hp_cutoff > 0:
        low[:,0] = butter_highpass_zerophase(low[:,0], fs=target_low, cutoff_hz=eda_hp_cutoff, order=eda_hp_order)

    # labels downsampled by block mode
    yd_low = block_mode_downsample(yk, in_len=n_keep, factor=fac_low)
    yd_ecg = block_mode_downsample(yk, in_len=n_keep, factor=fac_ecg)

    # impute nans defensively
    for arr in (Xd_ecg, low):
        bad = ~np.isfinite(arr)
        if bad.any():
            for c in range(arr.shape[1]):
                if bad[:,c].any():
                    arr[:,c] = _impute_linear_series(arr[:,c])

    return dict(
        subject_id=Path(subject_file).stem,
        fs_low=target_low, fs_ecg=target_ecg,
        X_low=low.astype(np.float32, copy=False),       # (N_low, 2) [EDA, RESP]
        X_ecg=Xd_ecg.astype(np.float32, copy=False),    # (N_ecg, 1)
        y_low=yd_low.astype(np.int64, copy=False),
        y_ecg=yd_ecg.astype(np.int64, copy=False),
    )

# ------------- combine subjects & windowing -------------

def process_multiple_subjects(subject_files, target_rate_low=4, target_rate_ecg=175,
                              transition_pad_s=5.0, min_valid_run_s=30.0, verbose=False):
    # collect and stack per subject (streams kept separate)
    segments = []
    all_low, all_ecg = [], []
    all_y_low, all_y_ecg = [], []
    fs_low = fs_ecg = None

    for i, p in enumerate(subject_files, 1):
        res = preprocess_single_subject(
            p, original_rate=700,
            target_low=target_rate_low, target_ecg=target_rate_ecg,
            transition_pad_s=transition_pad_s, min_valid_run_s=min_valid_run_s,
            verbose=verbose
        )
        if res is None: continue

        if fs_low is None:
            fs_low, fs_ecg = res["fs_low"], res["fs_ecg"]

        low_start = 0 if not all_low else segments[-1]["low_end"]
        ecg_start = 0 if not all_ecg else segments[-1]["ecg_end"]

        all_low.append(res["X_low"]); all_ecg.append(res["X_ecg"])
        all_y_low.append(res["y_low"]); all_y_ecg.append(res["y_ecg"])

        segments.append(dict(
            subject_id=res["subject_id"],
            low_start=low_start, low_end=low_start + len(res["X_low"]),
            ecg_start=ecg_start, ecg_end=ecg_start + len(res["X_ecg"])
        ))

    X_low = np.vstack(all_low).astype(np.float32, copy=False)
    X_ecg = np.concatenate(all_ecg).astype(np.float32, copy=False)
    y_low = np.concatenate(all_y_low).astype(np.int64, copy=False)
    y_ecg = np.concatenate(all_y_ecg).astype(np.int64, copy=False)

    return dict(
        X_low=X_low, X_ecg=X_ecg, y_low=y_low, y_ecg=y_ecg,
        fs_low=fs_low, fs_ecg=fs_ecg,
        channels_low=["EDA","RESP"], channels_ecg=["ECG"],
        segments=segments
    )

def _constant_label_windows_slice(X, y_raw, T, step):
    """Windows fully inside constant-label runs."""
    Xw, yw = [], []
    n = len(y_raw)
    i = 0
    while i < n:
        j = i+1
        while j < n and y_raw[j] == y_raw[i]:
            j += 1
        run_len = j - i
        if run_len >= T:
            for t0 in range(i, j - T + 1, step):
                mid = t0 + T//2
                Xw.append(X[t0:t0+T])
                yw.append(int(y_raw[mid]))
        i = j
    if not Xw: return None, None
    return np.stack(Xw, 0).astype(np.float32), np.asarray(yw, dtype=np.int64)

def create_training_sequences_two_stream(ds, window_s=30, step_s=15, require_single_label=True):
    fs_low, fs_ecg = ds["fs_low"], ds["fs_ecg"]
    T_low  = int(window_s * fs_low)   # 120
    T_ecg  = int(window_s * fs_ecg)   # 5250
    step_low = int(step_s * fs_low)   # 60
    step_ecg = int(step_s * fs_ecg)   # 2625

    # per subject slice → windows, then concat
    Xl, yl, Xe, ye = [], [], [], []
    for seg in ds["segments"]:
        lo0, lo1 = seg["low_start"], seg["low_end"]
        ec0, ec1 = seg["ecg_start"], seg["ecg_end"]

        out_low = _constant_label_windows_slice(ds["X_low"][lo0:lo1], ds["y_low"][lo0:lo1], T_low, step_low)
        out_ecg = _constant_label_windows_slice(ds["X_ecg"][ec0:ec1], ds["y_ecg"][ec0:ec1], T_ecg, step_ecg)
        if out_low[0] is None or out_ecg[0] is None: continue
        # By construction, both should produce same count if segments aligned and label-consistent.
        n = min(out_low[0].shape[0], out_ecg[0].shape[0])
        Xl.append(out_low[0][:n]); yl.append(out_low[1][:n])
        Xe.append(out_ecg[0][:n]); ye.append(out_ecg[1][:n])

    X_low_w = np.concatenate(Xl, 0); y_w = np.concatenate(yl, 0)   # use low labels for cond
    X_ecg_w = np.concatenate(Xe, 0)

    # train/test split by subjects is handled by caller; here we just return windows + meta
    return dict(
        X_low=X_low_w, X_ecg=X_ecg_w, cond=y_w,
        fs_low=fs_low, fs_ecg=fs_ecg,
        T_low=T_low, T_ecg=T_ecg,
        channels_low=ds["channels_low"], channels_ecg=ds["channels_ecg"]
    )

def zscore_train_and_apply(X_train, X_test):
    mean = X_train.mean(axis=(0,1), dtype=np.float64)
    std  = X_train.std(axis=(0,1), dtype=np.float64); std[std==0] = 1.0
    Xtr = ((X_train - mean)/std).astype(np.float32)
    Xte = ((X_test  - mean)/std).astype(np.float32)
    return Xtr, Xte, mean.astype(np.float32), std.astype(np.float32)

def save_two_stream(out_dir: Path, train, test, meta, ):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_X_low.npy",  train["X_low"])
    np.save(out_dir / "train_X_ecg.npy",  train["X_ecg"])
    np.save(out_dir / "train_cond.npy",   train["cond"])
    np.save(out_dir / "test_X_low.npy",   test["X_low"])
    np.save(out_dir / "test_X_ecg.npy",   test["X_ecg"])
    np.save(out_dir / "test_cond.npy",    test["cond"])
    np.savez(out_dir / "norm_low.npz", mean=train["mean_low"], std=train["std_low"])
    np.savez(out_dir / "norm_ecg.npz", mean=train["mean_ecg"], std=train["std_ecg"])
    np.save(out_dir / "train_m1_seq.npy", train["m1_seq"])
    np.save(out_dir / "test_m1_seq.npy",  test["m1_seq"])
    save_json(meta, out_dir / "dataset_config.json")