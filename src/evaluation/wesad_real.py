from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

def _np_broadcast(arr, C):
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    if a.size == 1:
        a = np.full((C,), float(a[0]), dtype=np.float32)
    if a.size != C:
        raise ValueError(f"Normalization vector size {a.size} != channels {C}")
    return a.reshape(1, 1, C)

def _denorm(x: np.ndarray, stats_npz: np.lib.npyio.NpzFile) -> np.ndarray:
    files = set(stats_npz.files)
    y = x.astype(np.float32, copy=True)
    C = y.shape[-1]
    if {"mean","std"} <= files:
        y = y * _np_broadcast(stats_npz["std"], C) + _np_broadcast(stats_npz["mean"], C)
    elif {"min","max"} <= files:
        y = y * (_np_broadcast(stats_npz["max"], C) - _np_broadcast(stats_npz["min"], C)) + _np_broadcast(stats_npz["min"], C)
    elif {"scale","bias"} <= files:
        y = y * _np_broadcast(stats_npz["scale"], C) + _np_broadcast(stats_npz["bias"], C)
    # else: unknown keys → return as-is
    return y

def _interp_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    """arr: (N, L, C) → (N, target_len, C) via linear interpolation along time."""
    N, L, C = arr.shape
    if L == target_len:
        return arr.astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, num=L, endpoint=True)
    xq = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    out = np.empty((N, target_len, C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            out[n, :, c] = np.interp(xq, xp, arr[n, :, c].astype(np.float64)).astype(np.float32)
    return out

@dataclass
class RealWESADPreparer:
    """
    Loads real test windows for a WESAD fold and returns fused arrays ready for evaluation.

    Inputs expected in: <data_root>/processed/<fold>/
      - test_X_ecg.npy   (N, 5250, 1) or (N, 5250)
      - test_X_low.npy   (N,  120, 2)   [Resp + EDA, possibly in arbitrary order]
      - test_cond.npy    (N,  120, K)   one-hot labels (optional but recommended)
      - norm_ecg.npz, norm_low.npz

    Output:
      signals: (N, T, 3) float32, channels = [ECG, Resp, EDA]
      labels:  (N,) int (optional; None if test_cond.npy missing)

    Usage:
      prep = RealWESADPreparer(r".../data/processed/tc_multigan_fold_S10")
      X, y = prep.prepare(target="ecg")      # T=5250
      # or
      X_low, y = prep.prepare(target="low")  # T=120
      out = prep.save_npz(X, y, out_dir="./results", filename="real_test_ecgT_S10.npz")
    """
    fold_dir: Path | str
    prefer_prefused: bool = False  # if True and test_X.npy exists, try to use it

    def __post_init__(self):
        self.fold_dir = Path(self.fold_dir).resolve()
        if not self.fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {self.fold_dir}")
        self.norm_ecg_path = self.fold_dir / "norm_ecg.npz"
        self.norm_low_path = self.fold_dir / "norm_low.npz"
        if not self.norm_ecg_path.exists() or not self.norm_low_path.exists():
            raise FileNotFoundError("Missing norm files (norm_ecg.npz / norm_low.npz).")

    # ---------- public API ----------
    def prepare(self, target: str = "ecg") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        target: 'ecg' → fuse to (N, 5250, 3); 'low' → (N, 120, 3)
        Returns (signals, labels_or_None)
        """
        # Optional fast-path: pre-fused file (only if you explicitly prefer it)
        if self.prefer_prefused:
            X_pref = self._try_load_prefused(target)
            if X_pref is not None:
                return X_pref, self._load_labels()

        X_ecg, X_low = self._load_streams()
        X_ecg = self._denorm_ecg(X_ecg)
        X_low = self._denorm_low_and_reorder(X_low)
        X_low[..., 1] = np.clip(X_low[..., 1], 0.0, None)  # safety: EDA non-negative

        if target == "ecg":
            fused = np.concatenate([X_ecg.astype(np.float32),
                                    _interp_to_len(X_low, X_ecg.shape[1])], axis=-1)
        elif target == "low":
            fused = np.concatenate([_interp_to_len(X_ecg, X_low.shape[1]),
                                    X_low.astype(np.float32)], axis=-1)
        else:
            raise ValueError("target must be 'ecg' or 'low'.")

        # Final checks
        if fused.ndim != 3 or fused.shape[2] != 3:
            raise ValueError(f"Expected (N,T,3), got {fused.shape}")
        if not np.isfinite(fused).all():
            raise ValueError("Fused signals contain NaN/Inf")

        return fused.astype(np.float32, copy=False), self._load_labels()

    def save_npz(self, signals: np.ndarray, labels: Optional[np.ndarray],
                 out_dir: Path | str, filename: str) -> Path:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / filename
        np.savez_compressed(
            p,
            signals=signals.astype(np.float32, copy=False),
            channels=np.array(["ECG", "Resp", "EDA"], dtype=object),
            labels=(labels if labels is not None else np.array([], dtype=np.int32)),
        )
        return p

    # ---------- internals ----------
    def _load_streams(self) -> Tuple[np.ndarray, np.ndarray]:
        ecg_p = self.fold_dir / "test_X_ecg.npy"
        low_p = self.fold_dir / "test_X_low.npy"
        if not ecg_p.exists() or not low_p.exists():
            raise FileNotFoundError(f"Missing stream files:\n  {ecg_p}\n  {low_p}")

        X_ecg = np.load(ecg_p)  # (N, 5250) or (N, 5250, 1)
        X_low = np.load(low_p)  # (N, 120, 2) or (N, 120)
        if X_ecg.ndim == 2: X_ecg = X_ecg[:, :, None]
        if X_low.ndim == 2: X_low = X_low[:, :, None]
        if X_ecg.shape[-1] != 1:
            raise ValueError(f"ECG last-dim should be 1, got {X_ecg.shape}")
        if X_low.shape[-1] != 2:
            raise ValueError(f"LOW last-dim should be 2, got {X_low.shape}")
        return X_ecg.astype(np.float32), X_low.astype(np.float32)

    def _denorm_ecg(self, X: np.ndarray) -> np.ndarray:
        ne = np.load(self.norm_ecg_path)
        return _denorm(X, ne)

    def _denorm_low_and_reorder(self, X: np.ndarray) -> np.ndarray:
        nl = np.load(self.norm_low_path)
        X = _denorm(X, nl)
        # reorder low to [Resp, EDA] if channels info exists
        if "channels" in nl.files:
            try:
                chs = [str(c).lower() for c in nl["channels"].tolist()]
                resp_idx = chs.index("resp") if "resp" in chs else chs.index("respiration")
                eda_idx  = chs.index("eda")  if "eda"  in chs else chs.index("electrodermal activity")
                X = X[..., [resp_idx, eda_idx]]
            except Exception:
                # fallback: assume file is already [Resp, EDA]
                pass
        return X

    def _load_labels(self) -> Optional[np.ndarray]:
        """Return integer labels per window if available.

        Accepts any of:
        (N, 120, K) one-hot over time  → argmax at t=0
        (N, K)      one-hot            → argmax
        (N,)        integer class ids  → as-is
        """
        cond_p = self.fold_dir / "test_cond.npy"
        if not cond_p.exists():
            return None

        cond = np.load(cond_p)
        # (N,120,K) → take first timestep, argmax over K
        if cond.ndim == 3 and cond.shape[-1] > 1:
            return cond[:, 0, :].argmax(axis=1).astype(np.int32)
        # (N,K) one-hot → argmax over K
        if cond.ndim == 2 and cond.shape[-1] > 1:
            return cond.argmax(axis=1).astype(np.int32)
        # (N,) already class ids
        if cond.ndim == 1:
            return cond.astype(np.int32)

        # Unknown layout → treat as unavailable
        return None

    def _try_load_prefused(self, target: str) -> Optional[np.ndarray]:
        """Try to use a pre-fused 'test_X.npy' if it exists and matches desired T."""
        p = self.fold_dir / "test_X.npy"
        if not p.exists():
            return None
        X = np.load(p)
        if X.ndim != 3 or X.shape[2] != 3:
            return None
        # If it already has the target T, assume channels [ECG, Resp, EDA].
        if target == "ecg" and X.shape[1] == 5250:
            return X.astype(np.float32, copy=False)
        if target == "low" and X.shape[1] == 120:
            return X.astype(np.float32, copy=False)
        # Else, we don't try to resample a mystery fused array → safer to rebuild from streams
        return None