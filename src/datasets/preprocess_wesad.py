
import pickle
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy import signal



# --- label config (4-class; change to {1,2,3} if you want 3-class) ---
LABEL_MAP = {1:"baseline", 2:"stress", 3:"amusement", 4:"meditation"}  # keep
VALID     = sorted(LABEL_MAP.keys())  # [1,2,3,4]
COND      = {lab:i for i, lab in enumerate(VALID)}  # {1:0,2:1,3:2,4:3}


def block_mode_downsample(labels, in_len, factor=None, n_out=None):
    """Mode of label in each block during downsampling."""
    y = np.asarray(labels)
    if factor is None and n_out is None:
        raise ValueError("Provide factor or n_out")
    if factor is None:
        factor = int(np.floor(in_len / n_out))
    n_blocks = int(np.floor(in_len / factor))
    out = np.empty(n_blocks, dtype=y.dtype)
    for i in range(n_blocks):
        block = y[i*factor:(i+1)*factor]
        vals, counts = np.unique(block, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out

def _impute_linear_series(col):
    # col: 1D array
    col = col.astype(np.float32, copy=False)
    n = col.shape[0]
    mask = np.isfinite(col)
    if mask.all():
        return col
    if not mask.any():
        return np.zeros_like(col)             # last-resort fallback
    if mask.sum() == 1:
        col[~mask] = col[mask][0]             # fill with single value
        return col
    idx = np.arange(n, dtype=np.float32)
    col[~mask] = np.interp(idx[~mask], idx[mask], col[mask])
    return col


def load_wesad_subject(pkl_path):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")
    with pkl_path.open("rb") as f:
        # encoding='latin1' is the safe choice for WESAD pickles
        return pickle.load(f, encoding="latin1")
    
def extract_chest_data_correct(subject_data, 
                               channels=("ECG","EDA","RESP"), 
                               strict=True, 
                               verbose=True, 
                               as_float32=True):
    """
    Extract selected chest channels from WESAD and stack to (N, C).
    channels: tuple of channel names in desired order.
              Valid names include: 'ECG','EDA','EMG','RESP','TEMP','ACC' (ACC is Nx3; not stacked here)
    strict:   raise if lengths mismatch; if False, truncate to the shortest length.
    """
    # 1) Access chest dict and labels
    chest = subject_data["signal"]["chest"]
    labels = np.asarray(subject_data["label"]).reshape(-1)

    # 2) Normalize key capitalization found in WESAD pickles
    #    (Resp→RESP, Temp→TEMP) while not overwriting if already present
    rename = {"Resp": "RESP", "Temp": "TEMP"}
    for old, new in rename.items():
        if old in chest and new not in chest:
            chest[new] = chest.pop(old)

    # 3) Gather requested channels in a fixed order
    arrays = []
    present = set(chest.keys())
    for ch in channels:
        if ch not in chest:
            # try case-insensitive match
            ci = {k.upper(): k for k in chest.keys()}
            if ch.upper() in ci:
                key = ci[ch.upper()]
            else:
                raise KeyError(f"Channel '{ch}' not found. Available: {sorted(present)}")
        else:
            key = ch

        x = np.asarray(chest[key])
        # Expect (N,1) or (N,), enforce (N,1)
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim == 2 and x.shape[1] != 1:
            raise ValueError(f"Channel '{ch}' expected shape (N,1) or (N,), got {x.shape}")
        elif x.ndim > 2:
            raise ValueError(f"Channel '{ch}' has invalid ndim={x.ndim}")

        arrays.append(x)

    # 4) Length checks (signals vs labels)
    lens = [a.shape[0] for a in arrays] + [labels.shape[0]]
    if len(set(lens)) != 1:
        if strict:
            raise ValueError(f"Length mismatch across channels/labels: {lens}")
        # else truncate to shortest
        n = min(lens)
        arrays = [a[:n] for a in arrays]
        labels = labels[:n]
    else:
        n = lens[0]

    # 5) Stack and cast
    X = np.concatenate(arrays, axis=1)
    if as_float32:
        X = X.astype(np.float32, copy=False)

    if verbose:
        ch_list = ", ".join(channels)
        print(f"📊 Extracted channels ({ch_list}) → X: {X.shape}, labels: {labels.shape}")

    return X, labels, list(channels)
    
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


def make_label_mask(labels, fs, valid_labels, transition_pad_s=5.0, min_valid_run_s=30.0):
    y = np.asarray(labels)
    n = y.size
    keep = np.isin(y, np.array(list(valid_labels)))
    # remove ±pad around transitions
    pad = int(round(transition_pad_s*fs))
    if pad > 0 and n > 1:
        chg = np.flatnonzero(np.diff(y) != 0)
        if chg.size:
            edge = np.zeros(n, dtype=bool)
            for i in chg:
                lo = max(0, i - pad + 1); hi = min(n, i + pad + 1)
                edge[lo:hi] = True
            keep &= ~edge
    # drop short runs
    min_len = int(round(min_valid_run_s*fs))
    if min_len > 1:
        i = 0
        while i < n:
            if keep[i]:
                j = i+1
                while j < n and keep[j]:
                    j += 1
                if (j - i) < min_len:
                    keep[i:j] = False
                i = j
            else:
                i += 1
    return keep

def preprocess_single_subject(subject_file, target_rate=4, original_rate=700, channels=("ECG","EDA","RESP"),
                              transition_pad_s=5.0, min_valid_run_s=30.0, verbose=True,
                              eda_hp_cutoff=0.03, eda_hp_order=2, eda_robust=False):
    sid = Path(subject_file).stem
    if verbose:
        print(f"\n📂 Processing: {sid}")

    try:
        # 1) Load + extract
        data = load_wesad_subject(subject_file)
        X, y, ch = extract_chest_data_correct(data, channels=channels, strict=True, verbose=False)
        n = len(X)
        if verbose:
            print(f"  Original: X {X.shape}, y {y.shape} (~{n/original_rate/60:.1f} min)")

        # 2) Channel-specific filtering at original fs
        Xf = np.empty_like(X, dtype=np.float32)
        for j, name in enumerate(ch):
            if name.upper() == "ECG":
                Xf[:, j] = butter_bandpass_zerophase(X[:, j], original_rate, low_hz=0.5, high_hz=40.0)
            elif name.upper() == "EDA":
                Xf[:, j] = butter_lowpass_zerophase(X[:, j], original_rate, cutoff_hz=5.0)
            elif name.upper() in ("RESP","RESPIRATION"):
                Xf[:, j] = butter_bandpass_zerophase(X[:, j], original_rate, low_hz=0.1, high_hz=0.35)
            else:
                Xf[:, j] = X[:, j]

        # 3) Label-based mask
        keep = make_label_mask(y, fs=original_rate, valid_labels=VALID,
                               transition_pad_s=transition_pad_s, min_valid_run_s=min_valid_run_s)
        Xk = Xf[keep]; yk = y[keep]
        if verbose:
            uniq, cnt = np.unique(yk, return_counts=True)
            print(f"  After mask (orig fs): {Xk.shape}, label dist: {dict(zip(uniq.tolist(), cnt.tolist()))}")

        # 4) Multi-rate
        TARGET_LOW, TARGET_ECG = target_rate, 175
        fac_low = original_rate // TARGET_LOW     # 175
        fac_ecg = original_rate // TARGET_ECG     # 4

        n_keep = (len(yk) // np.lcm(fac_low, fac_ecg)) * np.lcm(fac_low, fac_ecg)
        Xk, yk = Xk[:n_keep], yk[:n_keep]

        idx_ecg   = ch.index("ECG")
        idx_other = [j for j in range(len(ch)) if j != idx_ecg]

        # EDA + RESP @ 4 Hz
        Xd_low = signal.decimate(Xk[:, idx_other], fac_low, ftype="fir", axis=0, zero_phase=True)

        # Enforce [EDA, RESP] order
        cols_low = [ch[j] for j in idx_other]
        assert "EDA" in cols_low and "RESP" in cols_low, f"Low-rate cols missing: {cols_low}"
        perm = [cols_low.index("EDA"), cols_low.index("RESP")]
        Xd_low = Xd_low[:, perm]

        # EDA high-pass @ 4 Hz
        if verbose:
            from scipy.signal import welch
            nps = min(len(Xd_low), 1024)
            f0, P0 = welch(Xd_low[:,0], fs=TARGET_LOW, nperseg=nps)
            print(f"EDA BEFORE HP peak ~ {f0[P0.argmax()]:.4f} Hz")

        if eda_hp_cutoff and eda_hp_cutoff > 0:
            Xd_low[:,0] = butter_highpass_zerophase(Xd_low[:,0], fs=TARGET_LOW,
                                                    cutoff_hz=eda_hp_cutoff, order=eda_hp_order)

        if verbose:
            f1, P1 = welch(Xd_low[:,0], fs=TARGET_LOW, nperseg=nps)
            print(f"EDA AFTER  HP peak ~ {f1[P1.argmax()]:.4f} Hz")

        # Optional robust per-subject scale
        if eda_robust:
            def robust_z_1d(x):
                med = np.nanmedian(x)
                iqr = np.nanpercentile(x,75) - np.nanpercentile(x,25)
                scale = max(iqr/1.349, 1e-6)
                return (x - med) / scale
            Xd_low[:,0] = robust_z_1d(Xd_low[:,0])

        # ECG @ 175 Hz
        Xd_ecg = signal.decimate(Xk[:, idx_ecg], fac_ecg, ftype="fir", axis=0, zero_phase=True)[:, None]

        # Labels
        yd_low = block_mode_downsample(yk, in_len=n_keep, factor=fac_low)
        yd_ecg = block_mode_downsample(yk, in_len=n_keep, factor=fac_ecg)

        # Impute NaNs (defensive)
        for arr in (Xd_low, Xd_ecg):
            bad = ~np.isfinite(arr)
            if bad.any():
                for c in range(arr.shape[1]):
                    if bad[:, c].any():
                        arr[:, c] = _impute_linear_series(arr[:, c])

        # Sanity
        assert len(Xd_low) == len(yd_low)
        assert len(Xd_ecg) == len(yd_ecg)
        assert abs(len(Xd_low)/TARGET_LOW - len(Xd_ecg)/TARGET_ECG) < 1e-6

        # One-hot conditioning
        y_cond_low = np.vectorize(COND.get)(yd_low)
        K = len(COND)
        m1_low = np.zeros((len(y_cond_low), K), dtype=np.float32)
        m1_low[np.arange(len(y_cond_low)), y_cond_low] = 1.0

        if verbose:
            print(f"  ➜ X_low {Xd_low.shape} @ {TARGET_LOW} Hz | X_ecg {Xd_ecg.shape} @ {TARGET_ECG} Hz | m1 {m1_low.shape}")

        return {
            "subject_id": sid,
            "channels_low": ["EDA","RESP"],
            "channels_ecg": ["ECG"],
            "fs_low": TARGET_LOW,
            "fs_ecg": TARGET_ECG,
            "m2_low":   Xd_low.astype(np.float32, copy=False),
            "m2_ecg":   Xd_ecg.astype(np.float32, copy=False),
            "labels_low": yd_low.astype(np.int64, copy=False),
            "labels_ecg": yd_ecg.astype(np.int64, copy=False),
            "m1_low":   m1_low,
            "duration_minutes": len(Xd_low) / TARGET_LOW / 60.0,
        }
    except Exception as e:
        print(f"  ❌ Error processing {sid}: {e}")
        return None