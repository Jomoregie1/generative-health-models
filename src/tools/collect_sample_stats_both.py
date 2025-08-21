import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- Helpers -------------------------- #

def _prefix(d, pfx):
    out = {}
    for k, v in d.items():
        if k in ("n_samples_ecg", "T_ecg", "n_samples_low", "T_low"):
            out[k] = v
        else:
            out[f"{pfx}_{k}"] = v
    return out

def _epoch_from_name(path: Path):
    m = re.search(r'epoch_(\d+)', path.stem)
    return int(m.group(1)) if m else None

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def _rfft_mag(x, fs, nfft=0):
    T = x.shape[-1]
    use_nfft = _next_pow2(T) if (nfft is None or nfft == 0) else max(nfft, T)
    X = np.fft.rfft(x, n=use_nfft)
    freqs = np.fft.rfftfreq(use_nfft, d=1.0 / fs)
    mag = np.abs(X)
    return freqs, mag

def _band_power_ratio_1d(x, fs, f_lo, f_hi, f_base_max, nfft=0):
    x = np.asarray(x)
    x = x - np.mean(x)  # detrend
    freqs, mag = _rfft_mag(x, fs, nfft)
    pw = mag ** 2
    band_mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    base_mask = (freqs >= 0.0) & (freqs < float(f_base_max))
    band = pw[band_mask].sum()
    base = pw[base_mask].sum() + 1e-12
    return float(band / base)

def _safe_percentile_abs_diff(x, q):
    x = np.asarray(x)
    d = np.abs(np.diff(x, axis=0))
    return float(np.percentile(d, q))

def _to_2d_first_channel(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (N,T) or 3D (N,T,1), got {arr.shape}")
    return arr

def _to_2d_two_channels(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] < 2:
        raise ValueError(f"Expected (N,T,>=2), got {arr.shape}")
    if arr.shape[-1] > 2:
        arr = arr[..., :2]
    return arr

def _clamp_ecg(x, margin):
    return np.clip(x, -float(margin), float(margin))

def _clamp_low(x, margin):
    return np.clip(x, -float(margin), float(margin))

def _renorm_ecg_from_denorm(x, mean, std):
    if std is None or std <= 0:
        raise ValueError("Invalid ecg_std for re-normalization")
    return (x - float(mean)) / float(std)

def _renorm_low_from_denorm(x, eda_mean, eda_std, resp_mean, resp_std):
    if eda_std is None or eda_std <= 0 or resp_std is None or resp_std <= 0:
        raise ValueError("Invalid EDA/RESP std for re-normalization")
    y = x.copy()
    y[..., 0] = (y[..., 0] - float(eda_mean)) / float(eda_std)
    y[..., 1] = (y[..., 1] - float(resp_mean)) / float(resp_std)
    return y

def _list_epoch_paths(sample_dir: Path, stem: str, prefer_denorm: bool):
    """Return only the chosen variant; never mix normalized with *_DENORM."""
    if prefer_denorm:
        paths = sorted(sample_dir.glob(f"{stem}_epoch_*_DENORM.npy"))
        if not paths:  # fallback if DENORM missing
            paths = sorted(p for p in sample_dir.glob(f"{stem}_epoch_*.npy")
                           if "_DENORM" not in p.stem)
    else:
        paths = sorted(p for p in sample_dir.glob(f"{stem}_epoch_*.npy")
                       if "_DENORM" not in p.stem)
    return paths

# -------------------------- Metrics per epoch -------------------------- #

def ecg_metrics_for_epoch(ecg_np, fs_ecg, nfft_ecg, clip_margin):
    N, T = ecg_np.shape
    clip_frac = float((np.abs(ecg_np) >= float(clip_margin)).mean())
    tv_mean = float(np.mean(np.abs(np.diff(ecg_np, axis=1))))
    d999 = float(np.mean([_safe_percentile_abs_diff(s, 99.9) for s in ecg_np]))
    br_list = []
    for s in ecg_np:
        br_list.append(_band_power_ratio_1d(s, fs_ecg, 5.0, 40.0, 5.0, nfft=nfft_ecg))
    band_ratio = float(np.mean(br_list))
    std = float(ecg_np.std())
    return dict(
        ecg_clip_frac=clip_frac,
        ecg_tv_mean=tv_mean,
        ecg_d999=d999,
        ecg_band_ratio=band_ratio,
        ecg_std=std,
        n_samples_ecg=N,
        T_ecg=T,
    )

def low_metrics_for_epoch(low_np, fs_low, nfft_low, clip_margin,
                          br_eda_fmin=0.03, br_eda_fmax=0.25, br_eda_base=0.03,
                          br_resp_fmin=0.1, br_resp_fmax=0.5, br_resp_base=0.1):
    N, T, _ = low_np.shape
    eda  = low_np[..., 0]
    resp = low_np[..., 1]
    tv_eda  = float(np.mean(np.abs(np.diff(eda,  axis=1))))
    tv_resp = float(np.mean(np.abs(np.diff(resp, axis=1))))
    thr = float(clip_margin)
    clip_eda  = float((np.abs(eda)  >= thr).mean())
    clip_resp = float((np.abs(resp) >= thr).mean())
    br_resp = float(np.mean([_band_power_ratio_1d(s, fs_low, br_resp_fmin, br_resp_fmax, br_resp_base, nfft=nfft_low) for s in resp]))
    br_eda  = float(np.mean([_band_power_ratio_1d(s, fs_low, br_eda_fmin,  br_eda_fmax,  br_eda_base,  nfft=nfft_low) for s in eda]))
    std_eda  = float(eda.std())
    std_resp = float(resp.std())
    return dict(
        tv_eda=tv_eda, tv_resp=tv_resp,
        clip_frac_eda=clip_eda, clip_frac_resp=clip_resp,
        br_resp=br_resp, br_eda=br_eda,
        std_eda=std_eda, std_resp=std_resp,
        n_samples_low=N, T_low=T
    )

# -------------------------- Main -------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_dir', type=str, required=True)
    ap.add_argument('--fs_ecg', type=float, default=175.0)
    ap.add_argument('--fs_low', type=float, default=4.0)
    ap.add_argument('--nfft_ecg', type=int, default=4096)
    ap.add_argument('--nfft_low', type=int, default=256)

    # clip margins in normalized units (these are what D/FM/spec see)
    ap.add_argument('--ecg_margin', type=float, default=0.98)
    ap.add_argument('--boundary_margin_low', type=float, default=0.92)

    # which views to compute
    ap.add_argument('--report_views', type=str, default='obs,raw')  # any of: obs,raw

    # prefer *_DENORM.npy files? (careful; see guard below)
    ap.add_argument('--prefer_denorm', action='store_true')

    # Optional: allow OBS when using DENORM by re-normalizing with dataset stats
    ap.add_argument('--renorm_denorm', action='store_true',
                    help='Use with --prefer_denorm to z-score back to normalized units for OBS metrics.')
    ap.add_argument('--ecg_mean', type=float, default=None)
    ap.add_argument('--ecg_std',  type=float, default=None)
    ap.add_argument('--eda_mean', type=float, default=None)
    ap.add_argument('--eda_std',  type=float, default=None)
    ap.add_argument('--resp_mean', type=float, default=None)
    ap.add_argument('--resp_std',  type=float, default=None)

    # Band-ratio windows (match training defaults)
    ap.add_argument('--br_eda_fmin', type=float, default=0.03)
    ap.add_argument('--br_eda_fmax', type=float, default=0.25)
    ap.add_argument('--br_eda_base', type=float, default=0.03)
    ap.add_argument('--br_resp_fmin', type=float, default=0.1)
    ap.add_argument('--br_resp_fmax', type=float, default=0.5)
    ap.add_argument('--br_resp_base', type=float, default=0.1)

    ap.add_argument('--out_csv', type=str, default=None)
    args = ap.parse_args()

    sample_dir = Path(args.sample_dir)
    assert sample_dir.exists(), f"sample_dir not found: {sample_dir}"

    views = [v.strip() for v in args.report_views.split(',') if v.strip()]
    if args.prefer_denorm and ('obs' in views) and not args.renorm_denorm:
        raise ValueError(
            "prefer_denorm + obs view is invalid: margins are in normalized units. "
            "Either drop 'obs' from --report_views or add --renorm_denorm with dataset mean/std."
        )

    # Pick files without mixing variants
    ecg_paths = _list_epoch_paths(sample_dir, "fake_ecg", args.prefer_denorm)
    low_paths = _list_epoch_paths(sample_dir, "fake_low", args.prefer_denorm)

    # Build dicts by epoch
    ecg_by_epoch, low_by_epoch = {}, {}
    for p in ecg_paths:
        ep = _epoch_from_name(p)
        if ep is None: 
            continue
        ecg_by_epoch[ep] = _to_2d_first_channel(np.load(p))  # (N, T)

    for p in low_paths:
        ep = _epoch_from_name(p)
        if ep is None:
            continue
        low_by_epoch[ep] = _to_2d_two_channels(np.load(p))   # (N, T, 2)

    # Quick sanity (helps catch unit mix-ups)
    if ecg_by_epoch:
        s = float(next(iter(ecg_by_epoch.values())).std())
        print(f"[check] ECG std@first epoch ~ {s:.3f}")
    if low_by_epoch:
        z = next(iter(low_by_epoch.values()))
        print(f"[check] LOW std EDA/RESP ~ {float(z[...,0].std()):.3f}/{float(z[...,1].std()):.3f}")

    # ECG stats
    rows_ecg = []
    for ep, ecg_np_raw in sorted(ecg_by_epoch.items()):
        row = dict(epoch=ep, n_samples_ecg=ecg_np_raw.shape[0], T_ecg=ecg_np_raw.shape[1])

        # For OBS on DENORM, re-normalize back to z-scores first
        if args.prefer_denorm and args.renorm_denorm and ('obs' in views):
            ecg_norm_for_obs = _renorm_ecg_from_denorm(ecg_np_raw, args.ecg_mean, args.ecg_std)
        else:
            ecg_norm_for_obs = ecg_np_raw

        for v in views:
            if v == 'obs':
                ecg_v = _clamp_ecg(ecg_norm_for_obs, args.ecg_margin)
                m = ecg_metrics_for_epoch(ecg_v, args.fs_ecg, args.nfft_ecg, clip_margin=args.ecg_margin)
                row.update(_prefix(m, 'obs'))
            elif v == 'raw':
                m = ecg_metrics_for_epoch(ecg_np_raw, args.fs_ecg, args.nfft_ecg, clip_margin=args.ecg_margin)
                row.update(_prefix(m, 'raw'))
        rows_ecg.append(row)
    df_ecg = pd.DataFrame(rows_ecg).sort_values("epoch").reset_index(drop=True)

    # LOW stats
    rows_low = []
    for ep, low_np_raw in sorted(low_by_epoch.items()):
        row = dict(epoch=ep, n_samples_low=low_np_raw.shape[0], T_low=low_np_raw.shape[1])

        if args.prefer_denorm and args.renorm_denorm and ('obs' in views):
            low_norm_for_obs = _renorm_low_from_denorm(
                low_np_raw, args.eda_mean, args.eda_std, args.resp_mean, args.resp_std
            )
        else:
            low_norm_for_obs = low_np_raw

        for v in views:
            if v == 'obs':
                low_v = _clamp_low(low_norm_for_obs, args.boundary_margin_low)
                m = low_metrics_for_epoch(
                    low_v, args.fs_low, args.nfft_low, clip_margin=args.boundary_margin_low,
                    br_eda_fmin=args.br_eda_fmin, br_eda_fmax=args.br_eda_fmax, br_eda_base=args.br_eda_base,
                    br_resp_fmin=args.br_resp_fmin, br_resp_fmax=args.br_resp_fmax, br_resp_base=args.br_resp_base
                )
                row.update(_prefix(m, 'obs'))
            elif v == 'raw':
                m = low_metrics_for_epoch(
                    low_np_raw, args.fs_low, args.nfft_low, clip_margin=args.boundary_margin_low,
                    br_eda_fmin=args.br_eda_fmin, br_eda_fmax=args.br_eda_fmax, br_eda_base=args.br_eda_base,
                    br_resp_fmin=args.br_resp_fmin, br_resp_fmax=args.br_resp_fmax, br_resp_base=args.br_resp_base
                )
                row.update(_prefix(m, 'raw'))
        rows_low.append(row)
    df_low = pd.DataFrame(rows_low).sort_values("epoch").reset_index(drop=True)

    # Merge + save
    df_both = pd.merge(df_ecg, df_low, on="epoch", how="outer").sort_values("epoch")
    for df in (df_ecg, df_low, df_both):
        for col in df.columns:
            if col not in ("epoch", "n_samples_ecg", "T_ecg", "n_samples_low", "T_low"):
                df[col] = df[col].astype(float)

    out_both = args.out_csv or str(sample_dir / "sample_stats_both.csv")
    out_ecg  = str(sample_dir / "sample_stats_ecg.csv")
    out_low  = str(sample_dir / "sample_stats_low.csv")

    df_both.round(6).to_csv(out_both, index=False)
    df_ecg.round(6).to_csv(out_ecg, index=False)
    df_low.round(6).to_csv(out_low, index=False)

    print(f"[saved] {out_both}")
    print(f"[saved] {out_ecg}")
    print(f"[saved] {out_low}")

    # ---------- Quick plots (OBS only, if present) ----------
    def _plot_xy(x, y, title, ylabel, fname):
        if len(x) == 0:
            return
        plt.figure(figsize=(7,4))
        plt.plot(x, y, marker='o')
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(sample_dir / fname, dpi=120)
        plt.close()

    def _maybe_plot(df, col, title, ylabel, fname):
        if col in df.columns and len(df):
            _plot_xy(df["epoch"].values, df[col].values, title, ylabel, fname)

    _maybe_plot(df_ecg, "obs_ecg_clip_frac", "ECG near-saturation fraction (OBS)", "clip_frac", "plot_ecg_clip_frac.png")
    _maybe_plot(df_ecg, "obs_ecg_d999",      "ECG |Δ| 99.9th percentile (OBS)",   "d999",      "plot_ecg_d999.png")
    _maybe_plot(df_ecg, "obs_ecg_band_ratio","ECG 5–40Hz / <5Hz (OBS)",            "band_ratio","plot_ecg_band_ratio.png")
    _maybe_plot(df_ecg, "obs_ecg_tv_mean",   "ECG mean |Δ| (TV, OBS)",             "tv_mean",   "plot_ecg_tv_mean.png")

    _maybe_plot(df_low, "obs_tv_eda",        "EDA mean |Δ| (TV, OBS)",            "tv_eda",        "plot_low_tv_eda.png")
    _maybe_plot(df_low, "obs_tv_resp",       "RESP mean |Δ| (TV, OBS)",           "tv_resp",       "plot_low_tv_resp.png")
    _maybe_plot(df_low, "obs_br_resp",       "RESP 0.1–0.5Hz / <0.1Hz (OBS)",     "br_resp",       "plot_low_br_resp.png")
    _maybe_plot(df_low, "obs_br_eda",        "EDA 0.03–0.25Hz / <0.03Hz (OBS)",   "br_eda",        "plot_low_br_eda.png")
    _maybe_plot(df_low, "obs_clip_frac_eda", "EDA near-saturation frac (OBS)",     "clip_frac_eda", "plot_low_clip_eda.png")
    _maybe_plot(df_low, "obs_clip_frac_resp","RESP near-saturation frac (OBS)",    "clip_frac_resp","plot_low_clip_resp.png")

if __name__ == "__main__":
    main()