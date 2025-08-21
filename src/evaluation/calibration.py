# src/evaluation/calibration.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np

def _quick_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "finite_frac": float(np.isfinite(x).mean()),
        "mean": float(np.nanmean(x)),
        "std":  float(np.nanstd(x)),
    }

@dataclass
class _Targets:
    ecg_target_std: float
    resp_target_std: float
    eda_target_mean: float
    eda_target_std: float
    # reference quantiles for Resp (required)
    resp_qs: list
    resp_q:  list
    # optional reference quantiles for ECG histogram mapping
    ecg_qs: list | None = None
    ecg_q:  list | None = None


    def __post_init__(self):
        # coerce to arrays (handles when lists were passed)
        self.resp_qs = np.asarray(self.resp_qs, dtype=np.float64)
        self.resp_q  = np.asarray(self.resp_q,  dtype=np.float64)
        if self.ecg_qs is not None:
            self.ecg_qs = np.asarray(self.ecg_qs, dtype=np.float64)
        if self.ecg_q is not None:
            self.ecg_q  = np.asarray(self.ecg_q,  dtype=np.float64)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "version": "2.0",
            "ecg_target_std":  float(self.ecg_target_std),
            "resp_target_std": float(self.resp_target_std),
            "eda_target_mean": float(self.eda_target_mean),
            "eda_target_std":  float(self.eda_target_std),
            "resp_qs": np.asarray(self.resp_qs).tolist(),
            "resp_q":  np.asarray(self.resp_q).tolist(),
            "ecg_qs": (None if self.ecg_qs is None else np.asarray(self.ecg_qs).tolist()),
            "ecg_q":  (None if self.ecg_q  is None else np.asarray(self.ecg_q).tolist()),
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "_Targets":
        # back-compat with old keys if present
        if "resp_qs" not in d and "qs" in d:
            d["resp_qs"] = d.pop("qs")
        if "resp_q" not in d and "resp_q_ref" in d:
            d["resp_q"] = d.pop("resp_q_ref")
        if "ecg_q" not in d and "ecg_q_ref" in d:
            d["ecg_q"] = d.pop("ecg_q_ref")
        if "ecg_qs" not in d and d.get("ecg_q") is not None:
            d["ecg_qs"] = d.get("resp_qs")

        return cls(
            ecg_target_std=float(d["ecg_target_std"]),
            resp_target_std=float(d["resp_target_std"]),
            eda_target_mean=float(d["eda_target_mean"]),
            eda_target_std=float(d["eda_target_std"]),
            resp_qs=d["resp_qs"],
            resp_q=d["resp_q"],
            ecg_qs=d.get("ecg_qs"),
            ecg_q=d.get("ecg_q"),
        )
    

class WESADCalibrator:
    """
    Post-hoc calibration for (N, T, 3) arrays with channels [ECG, Resp, EDA].
    Steps implemented:
      - ECG amplitude scaling to match target std (optional ECG quantile-map)
      - Resp per-window quantile mapping to real distribution (+optional final std touch-up)
      - EDA global mean/std match + clamp to non-negative
    """

    def __init__(self, targets: _Targets):
        self.targets = targets
        # rehydrate to arrays for fast math
        self.resp_qs = np.asarray(targets.resp_qs, dtype=np.float64)
        self.resp_q  = np.asarray(targets.resp_q,  dtype=np.float64)
        self.ecg_qs  = None if targets.ecg_qs is None else np.asarray(targets.ecg_qs, dtype=np.float64)
        self.ecg_q   = None if targets.ecg_q  is None else np.asarray(targets.ecg_q,  dtype=np.float64)

    # ---------- factory methods ----------
    @classmethod
    def from_real(
        cls,
        X_real: np.ndarray,
        *,
        store_ecg_qmap: bool = False,
        resp_q_n: int = 2001,
        ecg_q_n: int = 2001,
        n_quantiles: int = 2001,   # legacy fallback; prefer resp_q_n/ecg_q_n
    ) -> "WESADCalibrator":
        """
        Build calibration targets from the real set (N, T, 3) with channels [ECG, Resp, EDA].

        Stores:
        - ECG target std (for amplitude scaling)
        - Resp target std (for optional std enforcement post mapping)
        - EDA target mean/std (for global shift/scale + clipping to >= 0)
        - Reference quantiles for Resp (always) and ECG (optional) to enable
            quantile mapping / histogram matching during calibration.
        """
        # --- validate ---
        if not (isinstance(X_real, np.ndarray) and X_real.ndim == 3 and X_real.shape[-1] == 3):
            raise ValueError("X_real must be a (N, T, 3) array with channels [ECG, Resp, EDA].")

        # ensure sensible quantile counts (odd is nice to include exact median)
        def _sanitize_qn(qn: int, fallback: int) -> int:
            try:
                qn = int(qn)
            except Exception:
                qn = int(fallback)
            qn = max(qn, 51)             # avoid super coarse mappings
            if qn % 2 == 0: qn += 1       # prefer odd (includes 0.5 exactly)
            return qn

        resp_q_n = _sanitize_qn(resp_q_n or n_quantiles, 2001)
        ecg_q_n  = _sanitize_qn(ecg_q_n  or n_quantiles, 2001)

        resp_qs = np.linspace(0.0, 1.0, int(resp_q_n), dtype=np.float64)
        resp_q  = np.quantile(X_real[...,1].ravel().astype(np.float64), resp_qs)

        Xr = X_real.astype(np.float64, copy=False)

        # --- per-channel targets ---
        ecg_target_std  = float(Xr[..., 0].std())
        resp_target_std = float(Xr[..., 1].std())
        eda_target_mean = float(Xr[..., 2].mean())
        eda_target_std  = float(Xr[..., 2].std())

        # NOTE: store lists so JSON serialization is trivial
        targets = _Targets(
            ecg_target_std=ecg_target_std,
            resp_target_std=resp_target_std,
            eda_target_mean=eda_target_mean,
            eda_target_std=eda_target_std,
            resp_qs=resp_qs, resp_q=resp_q
        )

        # Resp reference quantiles
        qs_r = np.linspace(0.0, 1.0, int(resp_q_n), dtype=np.float64)
        q_r  = np.quantile(X_real[..., 1].ravel().astype(np.float64), qs_r)
        targets.resp_qs = qs_r.tolist()
        targets.resp_q  = q_r.tolist()

        # (optional) ECG reference quantiles for histogram mapping
        if store_ecg_qmap:
            qs_e = np.linspace(0.0, 1.0, int(ecg_q_n), dtype=np.float64)
            q_e  = np.quantile(X_real[..., 0].ravel().astype(np.float64), qs_e)
            targets.ecg_qs = qs_e.tolist()
            targets.ecg_q  = q_e.tolist()


        return cls(targets)

    @classmethod
    def load(cls, path: str | Path) -> "WESADCalibrator":
        return cls(_Targets.from_json(json.loads(Path(path).read_text(encoding="utf-8"))))

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.targets.to_jsonable(), indent=2), encoding="utf-8")

    def has_ecg_qmap(self) -> bool:
        t = self.targets
        return (
            getattr(t, "ecg_qs", None) is not None and
            getattr(t, "ecg_q",  None) is not None and
            len(t.ecg_qs) > 0 and
            len(t.ecg_q)  > 0
        )

    # ---------- ops ----------
    @staticmethod
    def _quantile_match_1d(src: np.ndarray, qs: np.ndarray, ref_q: np.ndarray, clip: bool = True) -> np.ndarray:
        """Map src -> ref using quantiles. src is 1D view, qs/ref_q are global anchors."""
        s_q = np.quantile(src, qs)
        s_q, keep = np.unique(s_q, return_index=True)    # guard monotonicity
        r_q = ref_q[keep]
        mapped = np.interp(src, s_q, r_q)
        if clip:
            mapped = np.clip(mapped, r_q.min(), r_q.max())
        return mapped
    
     # --- NEW: duplicate-safe Resp quantile mapping (per window) ---
    def _map_resp_window(self, x_1d: np.ndarray) -> np.ndarray:
        qs_ref = self.resp_qs
        q_ref  = self.resp_q
        q_src  = np.quantile(x_1d.astype(np.float64), qs_ref)
        # guard against flat segments in the source
        q_src, keep = np.unique(q_src, return_index=True)
        q_ref = q_ref[keep]
        y = np.interp(x_1d, q_src, q_ref)
        y = np.clip(y, q_ref.min(), q_ref.max())
        return y.astype(np.float32)

    def apply(self, X_synth: np.ndarray, *,
              do_ecg: bool = True,
              do_resp: bool = True,
              do_eda: bool = True,
              ecg_qmap: bool = False,
              ecg_qmap_alpha: float = 1.0, 
              enforce_resp_std: bool = True) -> np.ndarray:
            
        Xc = X_synth.astype(np.float32, copy=True)

        # (1) ECG amplitude scale to real std
        if do_ecg:
            tgt = float(self.targets.ecg_target_std)
            sd  = float(Xc[..., 0].std()) + 1e-12
            Xc[..., 0] *= (tgt / sd)
        # (optional) ECG histogram/quantile map
        if ecg_qmap and self.has_ecg_qmap():
                qs  = np.asarray(self.targets.ecg_qs, dtype=np.float64)
                qrf = np.asarray(self.targets.ecg_q,  dtype=np.float64)

                def _map_ecg_window(x1d: np.ndarray) -> np.ndarray:
                    qsrc = np.quantile(x1d.astype(np.float64), qs)
                    qsrc, keep = np.unique(qsrc, return_index=True)
                    qref = qrf[keep]
                    y = np.interp(x1d, qsrc, qref).astype(np.float32)
                    if 0.0 < ecg_qmap_alpha < 1.0:
                        y = ecg_qmap_alpha * y + (1.0 - ecg_qmap_alpha) * x1d
                    # re-enforce target std after mapping
                    sd = float(y.std()) + 1e-12
                    y *= (tgt / sd)
                    return y

                for i in range(Xc.shape[0]):
                    Xc[i, :, 0] = _map_ecg_window(Xc[i, :, 0])

        # (2) Resp quantile map (window-wise)
        if do_resp:
            for i in range(Xc.shape[0]):
                Xc[i, :, 1] = self._map_resp_window(Xc[i, :, 1])
            if enforce_resp_std:
                s_t, s_c = float(self.targets.resp_target_std), float(Xc[..., 1].std())
                if s_c > 1e-12: Xc[..., 1] *= (s_t / s_c)

        # (3) EDA mean/std + non-negativity
        if do_eda:
            mu_t = float(self.targets.eda_target_mean)
            sd_t = float(self.targets.eda_target_std)
            mu_s = float(Xc[..., 2].mean())
            sd_s = float(Xc[..., 2].std()) + 1e-12
            Xc[..., 2] = (Xc[..., 2] - mu_s) * (sd_t / sd_s) + mu_t
            Xc[..., 2] = np.clip(Xc[..., 2], 0.0, None)


        return Xc

    # ---------- convenience ----------
    def summarize(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        return {
            "ECG":  _quick_stats(X[..., 0]),
            "Resp": _quick_stats(X[..., 1]),
            "EDA":  _quick_stats(X[..., 2]),
        }
