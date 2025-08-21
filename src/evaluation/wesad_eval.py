from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy.signal import welch, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.signal import welch

try:
    import torch, torch.nn as nn, torch.optim as optim
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


@dataclass
class EvalConfig:
    # existing…
    T_target: int = 5250
    fs_ecg: float = 175.0
    fs_low: float = 4.0
    hist_bins: int = 50
    psd_nperseg_ecg: int = 1024
    psd_nperseg_low: int = 128
    acf_max_lag: int = 100
    results_dir: Path = Path("./results/evaluation")

    # classifier
    run_classifier: bool = True
    clf_labels: Tuple[int, ...] = (0, 1)
    clf_epochs: int = 8
    clf_batch_size: int = 64
    clf_seed: int = 0
    clf_val_frac: float = 0.3
    clf_lr: float = 1e-3

    # --- NEW: distribution & PSD knobs ---
    apply_eda_linfix: bool = False          # If True, linearly align synth EDA mean/std to real (affects Table 1 KS/W1/JSD only)
    eda_linfix_clip_nonneg: bool = True     # Clip EDA >= 0 after linfix
    psd_low_max_hz: float = 1.5  
    psd_log: bool = False                # NEW: log10 PSD before correlation
    psd_norm: Optional[str] = None       # NEW: None | "z" | "unit"           # Low-stream PSD max frequency (Hz) when T_target==ECG

    # --- NEW: classifier knobs ---
    clf_zscore: bool = False                # Z-score Xr/Xs using REAL stats before training/testing
    clf_class_weight: str | None = None     # "balanced" or None
    psd_low_max_hz: float = 1.5                 # (you already use this for Resp/EDA)
    psd_ecg_band: Optional[Tuple[float,float]] = (0.5, 40.0)  # ECG PSD band; set None to disable
    ecg_hp_cut_hz: Optional[float] = None  

# ---------- small helpers ----------
def _interp_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation along time; arr: (B,L,C) -> (B,target_len,C)."""
    B, L, C = arr.shape
    if L == target_len:
        return arr.astype(np.float32, copy=False)

    xp = np.linspace(0.0, 1.0, num=L, endpoint=True)
    xq = np.linspace(0.0, 1.0, num=target_len, endpoint=True)

    out = np.empty((B, target_len, C), dtype=np.float32)
    for n in range(B):
        for c in range(C):
            out[n, :, c] = np.interp(xq, xp, arr[n, :, c].astype(np.float64)).astype(np.float32)
    return out

def _butter_highpass(cut_hz: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    Wn = max(cut_hz / nyq, 1e-6)
    b, a = butter(order, Wn, btype="highpass")
    return b, a

def _hp_filter_batch(X: np.ndarray, fs: float, cut_hz: float) -> np.ndarray:
    """High-pass filter each row of X (B,T) with zero-phase filtfilt."""
    if cut_hz is None:
        return X
    b, a = _butter_highpass(float(cut_hz), float(fs), order=2)
    Y = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        Y[i] = filtfilt(b, a, X[i].astype(np.float32, copy=False))
    return Y

def _load_npz_or_npy(p: Path) -> np.ndarray:
    p = Path(p)
    if p.suffix.lower() == ".npz":
        d = np.load(p)
        if "signals" in d:
            return d["signals"].astype(np.float32)
        # fallback: first (N,T,3) entry
        for k in d.files:
            if d[k].ndim == 3 and d[k].shape[-1] == 3:
                return d[k].astype(np.float32)
        raise ValueError(f"No (N,T,3) array in {p}")
    elif p.suffix.lower() == ".npy":
        x = np.load(p).astype(np.float32)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected (N,T,3) in {p}, got {x.shape}")
        return x
    else:
        raise ValueError(f"Unsupported file: {p}")

def _stack_by_condition(file_map: Dict[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    lab = {"baseline":0, "stress":1, "amusement":2}
    xs, ys = [], []
    for cond, path in file_map.items():
        x = _load_npz_or_npy(path)
        xs.append(x)
        ys.append(np.full((x.shape[0],), lab[cond], dtype=np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def _jsd_from_hist(p, q, eps=1e-8):
    p = p.astype(np.float64); q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5*(p+q)
    def kl(a,b):
        a_ = np.clip(a, eps, None); b_ = np.clip(b, eps, None)
        return np.sum(a_ * np.log(a_/b_))
    return 0.5*(kl(p,m) + kl(q,m))

def _mu_sd_over_BT(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-channel mean/std over batch and time. x: (N,T,C) -> (C,), (C,)"""
    xt = x.reshape(-1, x.shape[-1]).astype(np.float64)
    mu = xt.mean(axis=0).astype(np.float32)
    sd = xt.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    return mu, sd

# ---------- evaluator ----------
class WESADEvaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # --- I/O & alignment ---
    def load_and_align(
        self,
        real_files: Dict[str, Path],
        synth_files: Dict[str, Path],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Xr, yr = _stack_by_condition(real_files)
        Xs, ys = _stack_by_condition(synth_files)
        if Xr.shape[1] != self.cfg.T_target:
            Xr = _interp_to_len(Xr, self.cfg.T_target)
        if Xs.shape[1] != self.cfg.T_target:
            Xs = _interp_to_len(Xs, self.cfg.T_target)
        return Xr, yr, Xs, ys

    # --- (i) distributional fidelity ---
    def distribution_metrics(self, Xr: np.ndarray, Xs: np.ndarray) -> Dict[str, Dict[str, float]]:
        out = {}
        names = ["ECG","Resp","EDA"]

        # --- NEW: compute global per-channel stats on REAL and SYNTH for optional EDA linfix ---
        if self.cfg.apply_eda_linfix:
            mu_r, sd_r = _mu_sd_over_BT(Xr)  # (3,)
            mu_s, sd_s = _mu_sd_over_BT(Xs)  # (3,)
            # precompute EDA linfix params
            a = (sd_r[2] / (sd_s[2] + 1e-12))
            b = mu_r[2] - a * mu_s[2]

        for c, name in enumerate(names):
            r = Xr[..., c].ravel()
            s = Xs[..., c].ravel()

            # --- NEW: EDA-only linear correction (synth aligned to real) ---
            if name == "EDA" and self.cfg.apply_eda_linfix:
                s = a * s + b
                if self.cfg.eda_linfix_clip_nonneg:
                    s = np.clip(s, 0.0, None)

            ks = ks_2samp(r, s, alternative="two-sided", mode="auto").statistic
            w1 = wasserstein_distance(r, s)
            lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
            h_r, edges = np.histogram(r, bins=self.cfg.hist_bins, range=(lo,hi))
            h_s, _     = np.histogram(s, bins=self.cfg.hist_bins, range=(lo,hi))
            jsd = _jsd_from_hist(h_r, h_s)
            out[name] = {"KS": float(ks), "W1": float(w1), "JSD": float(jsd)}
        return out

    # --- (ii) PSD similarity + figure ---
    def _mean_psd(self, x: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        psds = []
        for i in range(x.shape[0]):
            f, p = welch(x[i], fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend="constant")
            psds.append(p)
        psds = np.stack(psds, axis=0)
        return f, psds.mean(0), psds.std(0)

    def psd_similarity(self, Xr: np.ndarray, Xs: np.ndarray) -> Dict[str, Dict[str, float]]:
        out = {}
        names = ["ECG","Resp","EDA"]
        fs = self.cfg.fs_ecg if self.cfg.T_target == 5250 else self.cfg.fs_low
        nperseg = self.cfg.psd_nperseg_ecg if self.cfg.T_target == 5250 else self.cfg.psd_nperseg_low

        for c, name in enumerate(names):
            # Optionally high-pass ECG before PSD
            if name == "ECG" and (self.cfg.ecg_hp_cut_hz is not None):
                r = _hp_filter_batch(Xr[..., c], fs=fs, cut_hz=self.cfg.ecg_hp_cut_hz)
                s = _hp_filter_batch(Xs[..., c], fs=fs, cut_hz=self.cfg.ecg_hp_cut_hz)
            else:
                r = Xr[..., c]; s = Xs[..., c]

            # Mean PSDs
            f_r, m_r, _ = self._mean_psd(r, fs=fs, nperseg=nperseg)
            f_s, m_s, _ = self._mean_psd(s, fs=fs, nperseg=nperseg)

            # Band masks
            if name in ("Resp","EDA") and fs == self.cfg.fs_ecg and (self.cfg.psd_low_max_hz is not None):
                mask_r = f_r <= float(self.cfg.psd_low_max_hz)
                mask_s = f_s <= float(self.cfg.psd_low_max_hz)
                f_r, m_r = f_r[mask_r], m_r[mask_r]
                f_s, m_s = f_s[mask_s], m_s[mask_s]

            if name == "ECG" and (self.cfg.psd_ecg_band is not None):
                lo, hi = map(float, self.cfg.psd_ecg_band)
                mask_r = (f_r >= lo) & (f_r <= hi)
                mask_s = (f_s >= lo) & (f_s <= hi)
                f_r, m_r = f_r[mask_r], m_r[mask_r]
                f_s, m_s = f_s[mask_s], m_s[mask_s]
            

            eps = 1e-12
            if self.cfg.psd_log:
                m_r = np.log10(m_r + eps)
                m_s = np.log10(m_s + eps)

            if self.cfg.psd_norm == "z":
                mr_mu, mr_sd = m_r.mean(), (m_r.std() + 1e-12)
                ms_mu, ms_sd = m_s.mean(), (m_s.std() + 1e-12)
                m_r = (m_r - mr_mu) / mr_sd
                m_s = (m_s - ms_mu) / ms_sd
            elif self.cfg.psd_norm == "unit":
                m_r = m_r / (m_r.sum() + eps)
                m_s = m_s / (m_s.sum() + eps)

            corr = float(np.corrcoef(m_r, m_s)[0, 1]) if (m_r.size and m_s.size) else float("nan")
            out[name] = {"PSD_sim": corr}
            
        return out

    def figure_psd_overlay(self, Xr: np.ndarray, Xs: np.ndarray) -> Path:
        names = ["ECG","Resp","EDA"]
        fs = self.cfg.fs_ecg if self.cfg.T_target == 5250 else self.cfg.fs_low
        nperseg = self.cfg.psd_nperseg_ecg if self.cfg.T_target == 5250 else self.cfg.psd_nperseg_low

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
        for idx, name in enumerate(names):
            r = Xr[..., idx]; s = Xs[..., idx]
            if name == "ECG" and (self.cfg.ecg_hp_cut_hz is not None):
                r = _hp_filter_batch(r, fs=fs, cut_hz=self.cfg.ecg_hp_cut_hz)
                s = _hp_filter_batch(s, fs=fs, cut_hz=self.cfg.ecg_hp_cut_hz)

            f_r, m_r, s_r = self._mean_psd(r, fs=fs, nperseg=nperseg)
            f_s, m_s, s_s = self._mean_psd(s, fs=fs, nperseg=nperseg)

            if name in ("Resp","EDA") and fs == self.cfg.fs_ecg and (self.cfg.psd_low_max_hz is not None):
                mask_r = f_r <= float(self.cfg.psd_low_max_hz); mask_s = f_s <= float(self.cfg.psd_low_max_hz)
                f_r, m_r, s_r = f_r[mask_r], m_r[mask_r], s_r[mask_r]
                f_s, m_s, s_s = f_s[mask_s], m_s[mask_s], s_s[mask_s]

            if name == "ECG" and (self.cfg.psd_ecg_band is not None):
                lo, hi = map(float, self.cfg.psd_ecg_band)
                mask_r = (f_r >= lo) & (f_r <= hi); mask_s = (f_s >= lo) & (f_s <= hi)
                f_r, m_r, s_r = f_r[mask_r], m_r[mask_r], s_r[mask_r]
                f_s, m_s, s_s = f_s[mask_s], m_s[mask_s], s_s[mask_s]

            ax = axes[idx]
            ax.plot(f_r, m_r, label="real");  ax.fill_between(f_r, m_r - s_r, m_r + s_r, alpha=0.2)
            ax.plot(f_s, m_s, label="synth"); ax.fill_between(f_s, m_s - s_s, m_s + s_s, alpha=0.2)
            ax.set_ylabel(f"{name} PSD"); ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("frequency [Hz]"); axes[0].legend()
        fig.tight_layout()
        out = self.cfg.results_dir / "figure_psd_overlay.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        return out

    # --- (iii) ACF similarity + figure ---
    def _mean_acf(self, x: np.ndarray, max_lag: int) -> np.ndarray:
        acfs = []
        for i in range(x.shape[0]):
            s = x[i] - x[i].mean()
            denom = (s*s).sum()
            if denom <= 1e-12:
                acfs.append(np.zeros(max_lag+1, dtype=np.float32))
                continue
            c = np.correlate(s, s, mode="full")
            mid = len(c)//2
            a = c[mid:mid+max_lag+1] / denom
            acfs.append(a.astype(np.float32))
        return np.stack(acfs, axis=0).mean(0)

    def acf_similarity(self, Xr: np.ndarray, Xs: np.ndarray) -> Dict[str, Dict[str, float]]:
        out = {}
        names = ["ECG","Resp","EDA"]
        for c, name in enumerate(names):
            ar = self._mean_acf(Xr[..., c], self.cfg.acf_max_lag)
            as_ = self._mean_acf(Xs[..., c], self.cfg.acf_max_lag)
            out[name] = {"ACF_sim": float(np.corrcoef(ar, as_)[0,1])}
        return out

    def figure_acf_overlay(self, Xr: np.ndarray, Xs: np.ndarray) -> Path:
        names = ["ECG","Resp","EDA"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for idx, name in enumerate(names):
            ar = self._mean_acf(Xr[..., idx], self.cfg.acf_max_lag)
            as_ = self._mean_acf(Xs[..., idx], self.cfg.acf_max_lag)
            ax = axes[idx]
            ax.plot(np.arange(self.cfg.acf_max_lag+1), ar, label="real")
            ax.plot(np.arange(self.cfg.acf_max_lag+1), as_, label="synth")
            ax.set_ylabel(f"{name} ACF")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("lag")
        axes[0].legend()
        fig.tight_layout()
        out = self.cfg.results_dir / "figure_acf_overlay.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # --- (iv) classifier (optional) ---
    def _classifier_metrics(self, Xr: np.ndarray, yr: np.ndarray, Xs: np.ndarray, ys: np.ndarray):
        if not _TORCH_OK:
            return {"Real→Real":{"AUROC": np.nan, "F1": np.nan},
                    "Synth→Real":{"AUROC": np.nan, "F1": np.nan},
                    "Real+Synth→Real":{"AUROC": np.nan, "F1": np.nan}}

        from sklearn.metrics import roc_auc_score, f1_score
        import torch.nn as nn, torch.optim as optim

        target_labels = tuple(self.cfg.clf_labels)

        def select_labels(X, y, labels):
            m = np.isin(y, labels)
            X2, y2 = X[m], y[m]
            lab2idx = {lab:i for i, lab in enumerate(sorted(labels))}
            y2 = np.array([lab2idx[int(v)] for v in y2], dtype=np.int64)
            return X2, y2, len(lab2idx)

        Xr, yr, K = select_labels(Xr, yr, target_labels)
        Xs, ys, _ = select_labels(Xs, ys, target_labels)
        if K < 2:
            return {"Real→Real":{"AUROC": np.nan, "F1": np.nan},
                    "Synth→Real":{"AUROC": np.nan, "F1": np.nan},
                    "Real+Synth→Real":{"AUROC": np.nan, "F1": np.nan}}

        # --- NEW: Z-score (using REAL stats) before splitting ---
        if self.cfg.clf_zscore:
            mu, sd = _mu_sd_over_BT(Xr)  # (C,)
            Xr = (Xr - mu.reshape(1,1,-1)) / sd.reshape(1,1,-1)
            Xs = (Xs - mu.reshape(1,1,-1)) / sd.reshape(1,1,-1)

        # simple split on real
        rng = np.random.RandomState(self.cfg.clf_seed)
        idx = np.arange(len(Xr)); rng.shuffle(idx)
        n_val = max(1, int(self.cfg.clf_val_frac * len(Xr)))
        te_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xr_tr, yr_tr = Xr[tr_idx], yr[tr_idx]
        Xr_te, yr_te = Xr[te_idx], yr[te_idx]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class CNN1D(nn.Module):
            def __init__(self, in_ch=3, n_classes=2):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv1d(in_ch, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                    nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.fc = nn.Linear(32, n_classes)
                self.n_classes = n_classes
            def forward(self, x):
                x = x.permute(0,2,1)
                h = self.body(x).squeeze(-1)
                return self.fc(h)

        def fit_eval(Xtr, ytr, Xte, yte, K):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CNN1D(n_classes=K).to(device)
            opt = optim.AdamW(model.parameters(), lr=self.cfg.clf_lr)

            # --- NEW: optional class weights on training split ---
            weight_tensor = None
            if self.cfg.clf_class_weight and str(self.cfg.clf_class_weight).lower() == "balanced":
                counts = np.bincount(ytr.astype(np.int64), minlength=K).astype(np.float32)
                weights = counts.sum() / (K * np.maximum(counts, 1.0))
                weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

            Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(device)
            ytr_t = torch.from_numpy(ytr.astype(np.int64)).to(device)

            bs = self.cfg.clf_batch_size
            for _ in range(self.cfg.clf_epochs):
                model.train()
                for i in range(0, len(Xtr_t), bs):
                    xb = Xtr_t[i:i+bs]; yb = ytr_t[i:i+bs]
                    opt.zero_grad()
                    logits = model(xb)
                    l = loss_fn(logits, yb)  # uses weights if provided
                    l.backward(); opt.step()

            model.eval()
            Xte_t = torch.from_numpy(Xte.astype(np.float32)).to(device)
            with torch.no_grad():
                logits = model(Xte_t).cpu().numpy()

            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            yhat  = probs.argmax(axis=1)

            f1 = float(f1_score(yte, yhat, average="macro"))
            if K == 2:
                auroc = float(roc_auc_score(yte, probs[:,1]))
            else:
                from sklearn.preprocessing import label_binarize
                Yte = label_binarize(yte, classes=np.arange(K))
                auroc = float(roc_auc_score(Yte, probs, multi_class="ovr", average="macro"))
            return {"AUROC": auroc, "F1": f1}

        res_RR = fit_eval(Xr_tr, yr_tr, Xr_te, yr_te, K)
        res_SR = fit_eval(Xs, ys, Xr_te, yr_te, K)
        Xmix   = np.concatenate([Xr_tr, Xs], axis=0)
        ymix   = np.concatenate([yr_tr, ys], axis=0)
        res_RpS_R = fit_eval(Xmix, ymix, Xr_te, yr_te, K)

        return {"Real→Real": res_RR, "Synth→Real": res_SR, "Real+Synth→Real": res_RpS_R}

    # --- save tables ---
    def save_table1(self, dist: Dict, psd: Dict, acf: Dict) -> Path:
        out = self.cfg.results_dir / "table1_distribution_psd_acf.csv"
        with out.open("w", encoding="utf-8") as f:
            f.write("channel,KS,W1,JSD,PSD_sim,ACF_sim\n")
            for name in ["ECG","Resp","EDA"]:
                f.write(f"{name},{dist[name]['KS']},{dist[name]['W1']},{dist[name]['JSD']},{psd[name]['PSD_sim']},{acf[name]['ACF_sim']}\n")
        return out

    def save_table2(self, clf: Dict[str, Dict[str,float]]) -> Path:
        out = self.cfg.results_dir / "table2_classifier_metrics.csv"
        with out.open("w", encoding="utf-8") as f:
            f.write("setting,AUROC,F1\n")
            for k,v in clf.items():
                f.write(f"{k},{v['AUROC']},{v['F1']}\n")
        return out

    # --- one-shot orchestrator for the dashboard/backend ---
    def evaluate_all(
        self,
        real_files: Dict[str, Path],
        synth_files: Dict[str, Path],
    ) -> Dict[str, Any]:
        Xr, yr, Xs, ys = self.load_and_align(real_files, synth_files)

        dist = self.distribution_metrics(Xr, Xs)
        psd  = self.psd_similarity(Xr, Xs)
        acf  = self.acf_similarity(Xr, Xs)
        fig_psd = self.figure_psd_overlay(Xr, Xs)
        fig_acf = self.figure_acf_overlay(Xr, Xs)
        t1 = self.save_table1(dist, psd, acf)

        clf = {}
        t2  = None
        if self.cfg.run_classifier:
            clf = self._classifier_metrics(Xr, yr, Xs, ys)
            t2 = self.save_table2(clf)

        return {
            "table1_csv": str(t1),
            "table2_csv": (None if t2 is None else str(t2)),
            "figure_psd": str(fig_psd),
            "figure_acf": str(fig_acf),
            "metrics": {
                "distribution": dist,
                "psd": psd,
                "acf": acf,
                "classifier": clf,
            }
        }