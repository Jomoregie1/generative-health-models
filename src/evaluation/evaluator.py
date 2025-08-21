from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats

# --------------------- helpers ---------------------

def _ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _empirical_cdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys

def _js_div_from_hist(x: np.ndarray, y: np.ndarray, bins='fd', eps=1e-12, base=2) -> float:
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float('nan')

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    if bins == 'fd':
        comb = np.concatenate([x, y])
        q75, q25 = np.percentile(comb, [75, 25])
        iqr = q75 - q25
        if iqr <= 0:
            nbins = int(np.sqrt(max(comb.size, 1)))
        else:
            bw = 2 * iqr * (comb.size ** (-1/3))
            nbins = int(np.clip(np.ceil((hi - lo) / max(bw, 1e-12)), 10, 512))
    else:
        nbins = int(bins) if isinstance(bins, (int, np.integer)) else 64
    if not np.isfinite(hi - lo) or (hi <= lo):
        lo, hi = -1.0, 1.0
    edges = np.linspace(lo, hi, nbins + 1)

    p, _ = np.histogram(x, bins=edges, density=True)
    q, _ = np.histogram(y, bins=edges, density=True)
    p = (p + eps); q = (q + eps)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return np.sum(a * (np.log(a) - np.log(b))) / np.log(base)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))

def _plot_overlays(real_flat: np.ndarray, fake_flat: np.ndarray, name: str, out_dir: Path):
    out_dir = _ensure_dir(out_dir)
    xr, yr = _empirical_cdf(real_flat)
    xf, yf = _empirical_cdf(fake_flat)

    # CDF
    plt.figure()
    if xr.size: plt.plot(xr, yr, label='Real')
    if xf.size: plt.plot(xf, yf, label='Fake')
    plt.title(f'Empirical CDF — {name}')
    plt.xlabel(name); plt.ylabel('F(x)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / f'cdf_{name}.png', dpi=180); plt.close()

    # Histogram
    comb = np.concatenate([real_flat.ravel(), fake_flat.ravel()])
    comb = comb[~np.isnan(comb)]
    if comb.size == 0: 
        return
    nbins = int(np.clip(np.sqrt(comb.size), 30, 200))
    lo, hi = comb.min(), comb.max()
    edges = np.linspace(lo, hi, nbins + 1) if hi > lo else np.linspace(lo - 1, hi + 1, nbins + 1)
    plt.figure()
    plt.hist(real_flat, bins=edges, density=True, alpha=0.55, label='Real')
    plt.hist(fake_flat, bins=edges, density=True, alpha=0.55, label='Fake')
    plt.title(f'Density — {name}')
    plt.xlabel(name); plt.ylabel('Density')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / f'hist_{name}.png', dpi=180); plt.close()

def _channel_flatten(batch_low: torch.Tensor, batch_ecg: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    batch_low: (B, T_low, 2)  -> {'eda': (B*T_low,), 'resp': (B*T_low,)}
    batch_ecg: (B, T_ecg, 1)  -> {'ecg': (B*T_ecg,)}
    """
    d = {}
    if batch_low is not None and batch_low.ndim == 3 and batch_low.size(-1) >= 2:
        d["eda"]  = batch_low[..., 0].reshape(-1).cpu().numpy()
        d["resp"] = batch_low[..., 1].reshape(-1).cpu().numpy()
    if batch_ecg is not None and batch_ecg.ndim == 3 and batch_ecg.size(-1) >= 1:
        d["ecg"]  = batch_ecg[..., 0].reshape(-1).cpu().numpy()
    return d


# ----------------- distribution-level eval -----------------

@torch.no_grad()
def eval_distribution_epoch(
    G, 
    val_loader, 
    cfg,
    device: torch.device,
    epoch: int,
    out_root: str | Path,
    channels: Sequence[str] = ("eda", "resp", "ecg"),
    n_batches: int = 8,
    standardize: bool = False,
    bootstrap: int = 0,
    viz_n: int = 4,
    # --- new, all optional; read sensible defaults from cfg if None ---
    use_clamp: Optional[bool] = None,           # mirror D's observed view
    use_denorm: Optional[bool] = None,          # convert back to physical units using norm_*.npz
    stats_low_path: Optional[str] = None,
    stats_ecg_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute KS, W1, JS per channel using val batches. 
    We generate fakes conditioned on the batch conditions and compare to the 
    corresponding real windows (same batches). Metrics are computed **per-channel**.

    New options:
      • use_clamp   : if True, clamp real/fake to cfg.boundary_margin_low / cfg.ecg_margin
      • use_denorm  : if True, de-normalize real/fake using norm_low.npz / norm_ecg.npz
                      BEFORE flattening (so W1 can be read in physical units)
    """
    # --- defaults from cfg if not provided ---
    if use_clamp is None:
        use_clamp = bool(getattr(cfg, "eval_use_clamp", True))
    if use_denorm is None:
        use_denorm = bool(getattr(cfg, "eval_denorm", False))
    margin_low = float(getattr(cfg, "boundary_margin_low", 0.92))
    margin_ecg = float(getattr(cfg, "ecg_margin", 0.98))

    # Stats for de-normalization (if requested)
    if stats_low_path is None:
        stats_low_path = str(Path(cfg.data_root) / cfg.fold / "norm_low.npz")
    if stats_ecg_path is None:
        stats_ecg_path = str(Path(cfg.data_root) / cfg.fold / "norm_ecg.npz")

    mu_low = sd_low = mu_ecg = sd_ecg = None
    if use_denorm:
        try:
            sL = np.load(stats_low_path); sE = np.load(stats_ecg_path)
            mu_low = torch.tensor(sL["mean"][:2], dtype=torch.float32, device=device)  # (2,)
            sd_low = torch.tensor(sL["std"] [:2], dtype=torch.float32, device=device)
            mu_ecg = torch.tensor(sE["mean"][:1], dtype=torch.float32, device=device)  # (1,)
            sd_ecg = torch.tensor(sE["std"] [:1], dtype=torch.float32, device=device)
        except Exception as e:
            print(f"[eval_dist] warn: could not load denorm stats ({e}); continuing in normalized units.")
            use_denorm = False

    def _apply_view(x_low: torch.Tensor, x_ecg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) optional clamp (observed view)
        if use_clamp:
            if x_low is not None:
                x_low = x_low.clamp(-margin_low, margin_low)
            if x_ecg is not None:
                x_ecg = x_ecg.clamp(-margin_ecg, margin_ecg)
        # 2) optional de-normalization to physical units
        if use_denorm:
            if (x_low is not None) and (mu_low is not None) and (sd_low is not None) and (x_low.size(-1) >= 2):
                # only first two channels (EDA, RESP)
                x_low = x_low.clone()
                x_low[..., 0] = x_low[..., 0] * sd_low[0] + mu_low[0]
                x_low[..., 1] = x_low[..., 1] * sd_low[1] + mu_low[1]
            if (x_ecg is not None) and (mu_ecg is not None) and (sd_ecg is not None) and (x_ecg.size(-1) >= 1):
                x_ecg = x_ecg.clone()
                x_ecg[..., 0] = x_ecg[..., 0] * sd_ecg[0] + mu_ecg[0]
        return x_low, x_ecg

    G.eval()
    out_dir = _ensure_dir(Path(out_root) / f"eval_epoch_{epoch:03d}")
    plots_dir = _ensure_dir(out_dir / "plots")

    # collect real/fake flattened arrays per channel
    buf_real: Dict[str, list] = {c: [] for c in channels}
    buf_fake: Dict[str, list] = {c: [] for c in channels}

    n_done = 0
    for batch in val_loader:
        if n_done >= n_batches: 
            break
        sig_low  = batch["signal_low"].to(device).float()    # (B, T_low, 2)
        sig_ecg  = batch["signal_ecg"].to(device).float()    # (B, T_ecg, 1)
        cond_low = batch["condition"].to(device).float()     # (B, T_low, K)

        # generate fakes for this batch
        z = torch.randn(sig_low.size(0), cfg.z_dim, device=device)
        fake_low, fake_ecg = G(z, cond_low)  # raw G output in training scale

        # apply evaluation view (clamp / denorm) BEFORE flatten
        r_low, r_ecg = _apply_view(sig_low,  sig_ecg)
        f_low, f_ecg = _apply_view(fake_low, fake_ecg)

        # flatten per channel
        real_flat = _channel_flatten(r_low, r_ecg)
        fake_flat = _channel_flatten(f_low, f_ecg)

        for ch in channels:
            if ch in real_flat and ch in fake_flat:
                r = real_flat[ch].astype(np.float64)
                f = fake_flat[ch].astype(np.float64)
                if standardize:
                    mu = np.nanmean(r)
                    sd = np.nanstd(r) + 1e-12
                    r = (r - mu) / sd
                    f = (f - mu) / sd
                buf_real[ch].append(r)
                buf_fake[ch].append(f)

        n_done += 1

    # compute metrics
    results: Dict[str, Dict[str, float]] = {}
    for ch in channels:
        if not buf_real[ch] or not buf_fake[ch]:
            continue
        r = np.concatenate(buf_real[ch], axis=0)
        f = np.concatenate(buf_fake[ch], axis=0)

        ks_stat, ks_p = stats.ks_2samp(r, f, alternative='two-sided', method='auto')
        w1 = stats.wasserstein_distance(r, f)
        js = _js_div_from_hist(r, f, bins='fd')

        res = dict(ks=float(ks_stat), ks_p=float(ks_p), w1=float(w1), js=float(js))

        # optional bootstrap CI
        if bootstrap and min(r.size, f.size) >= 20:
            rng = np.random.default_rng(42)
            n   = int(min(r.size, f.size))
            ks_b, w1_b, js_b = [], [], []
            for _ in range(int(bootstrap)):
                idx_r = rng.choice(r.size, n, replace=True)
                idx_f = rng.choice(f.size, n, replace=True)
                rb, fb = r[idx_r], f[idx_f]
                ksb, _ = stats.ks_2samp(rb, fb, alternative='two-sided', method='auto')
                w1b    = stats.wasserstein_distance(rb, fb)
                jsb    = _js_div_from_hist(rb, fb, bins='fd')
                ks_b.append(ksb); w1_b.append(w1b); js_b.append(jsb)
            def ci(a):
                return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
            res["ks_ci95"] = ci(ks_b)
            res["w1_ci95"] = ci(w1_b)
            res["js_ci95"] = ci(js_b)

        results[ch] = res
        _plot_overlays(r, f, name=ch, out_dir=plots_dir)

    # persist (CSV + JSON + README)
    rows = [{"channel": ch, **metrics} for ch, metrics in results.items()]
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    except Exception:
        # fallback CSV
        with open(out_dir / "metrics.csv", "w") as fh:
            if rows:
                keys = ["channel"] + [k for k in rows[0].keys() if k != "channel"]
                fh.write(",".join(keys) + "\n")
                for r in rows:
                    fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    with open(out_dir / "README.txt", "w") as f:
        f.write(
            "Channel-wise distribution metrics (KS, W1, JS).\n"
            "KS: largest CDF gap; W1: mean shift; JS: bounded divergence (mode coverage).\n"
            f"Options used — clamp={use_clamp}, denorm={use_denorm}, standardize={standardize}, "
            f"n_batches={n_batches}, bootstrap={bootstrap}.\n"
        )
    return results



# ------------- label-structure probe (linear classifier) -------------

class LinearProbe(torch.nn.Module):
    """Single linear layer probe: h -> logits (num_classes)."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

@torch.no_grad()
def _safe_labels_from_batch(batch, K: int, device: torch.device) -> Optional[torch.Tensor]:
    labels = batch.get("label")
    if labels is None:
        return None
    y = (labels.to(device).long() if labels.ndim == 1 else labels.to(device).argmax(dim=-1).long())
    if y.min() >= 1 and y.max() == K:
        y = y - 1  # map 1..K -> 0..K-1 if needed
    return y

def _pooled_features(D, sig_low, sig_ecg, cond):
    """
    Use the discriminator trunk to extract pooled features. Works with your
    D.extract_features return type (pooled or (pooled, pooled_proj)).
    """
    feat = D.extract_features(sig_low, sig_ecg, cond)
    if isinstance(feat, tuple):
        feat = feat[0]
    return feat

def _acc(pred, target):
    return float((pred.argmax(dim=1) == target).float().mean().item())

@torch.no_grad()
def _confusion(pred, target, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    y = target.cpu().numpy()
    p = pred.argmax(dim=1).cpu().numpy()
    for yi, pi in zip(y, p):
        cm[yi, pi] += 1
    return cm

def _balanced_accuracy(cm: np.ndarray) -> float:
    # recall per class = TP / (TP + FN) = diag / row_sum
    row_sum = cm.sum(axis=1, keepdims=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = np.where(row_sum > 0, np.diag(cm) / row_sum, 0.0)
    return float(np.mean(rec))

def _macro_f1(cm: np.ndarray) -> float:
    # precision per class = TP / (TP + FP) = diag / col_sum
    col_sum = cm.sum(axis=0, keepdims=False)
    row_sum = cm.sum(axis=1, keepdims=False)
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(col_sum > 0, tp / col_sum, 0.0)
        rec  = np.where(row_sum > 0, tp / row_sum, 0.0)
        f1   = np.where((prec + rec) > 0, 2.0 * prec * rec / (prec + rec), 0.0)
    return float(np.mean(f1))

def eval_label_probe(
    G, D, 
    train_loader, val_loader, 
    cfg, device: torch.device,
    epoch: int,
    out_root: str | Path,
    n_train = None, 
    n_val = None,
    n_train_batches: int = 8,
    n_val_batches: int   = 8,
) -> Dict[str, float]:
    """
    Train a tiny linear probe on *real* pooled features (from D) to predict stress labels,
    then evaluate on (a) real val features and (b) synthetic features generated with
    the same condition sequences. This tests whether G preserves label structure.
    """

    if n_train is None:
        n_train = int(getattr(cfg, "probe_n_train", 2000))
    if n_val is None:
        n_val = int(getattr(cfg, "probe_n_val", 1000))

    out_dir = _ensure_dir(Path(out_root) / f"eval_epoch_{epoch:03d}" / "probe")
    plots_dir = _ensure_dir(out_dir / "plots")
    K = int(cfg.condition_dim)

    X_train, y_train = [], []
    D.eval(); G.eval()
    got_labels = False

    # ---- REAL train features ----
    tdone = 0
    for batch in train_loader:
        if tdone >= n_train_batches: break
        y = _safe_labels_from_batch(batch, K, device)
        if y is None:
            continue
        sig_low  = batch["signal_low"].to(device).float()
        sig_ecg  = batch["signal_ecg"].to(device).float()
        cond_low = batch["condition"].to(device).float()

        feat = _pooled_features(D, sig_low, sig_ecg, cond_low)
        X_train.append(feat.detach().cpu()); y_train.append(y.detach().cpu())
        got_labels = True
        tdone += 1

    if not got_labels:
        with open(out_dir / "README.txt", "w") as f:
            f.write("Skipped label probe: dataset/batches did not include 'label'.\n")
        return {"skipped": 1.0}

    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    # ---- REAL + SYNTH val features ----
    X_val_real, y_val = [], []
    X_val_synth = []
    vdone = 0
    for batch in val_loader:
        if vdone >= n_val_batches: break
        y = _safe_labels_from_batch(batch, K, device)
        if y is None:
            continue

        sig_low  = batch["signal_low"].to(device).float()
        sig_ecg  = batch["signal_ecg"].to(device).float()
        cond_low = batch["condition"].to(device).float()

        fr = _pooled_features(D, sig_low, sig_ecg, cond_low)     # real pooled
        X_val_real.append(fr.detach().cpu())
        y_val.append(y.detach().cpu())

        z = torch.randn(sig_low.size(0), cfg.z_dim, device=device)  # synth pooled
        fake_low, fake_ecg = G(z, cond_low)
        fs = _pooled_features(D, fake_low, fake_ecg, cond_low)
        X_val_synth.append(fs.detach().cpu())

        vdone += 1

    X_val_real  = torch.cat(X_val_real,  dim=0)
    X_val_synth = torch.cat(X_val_synth, dim=0)
    y_val       = torch.cat(y_val,       dim=0)

    # ---- tiny linear probe on REAL ----
    in_dim = X_train.shape[1]
    probe  = LinearProbe(in_dim, K).to(device)
    opt    = torch.optim.Adam(probe.parameters(), lr=1e-3)
    epochs = 10

    X_train_d = X_train.to(device); y_train_d = y_train.to(device)
    for _ in range(epochs):
        probe.train()
        logits = probe(X_train_d)
        loss   = F.cross_entropy(logits, y_train_d)
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- evaluate on REAL val and SYNTH val ----
    probe.eval()
    with torch.no_grad():
        pr_real  = probe(X_val_real.to(device))
        pr_synth = probe(X_val_synth.to(device))

        acc_real  = _acc(pr_real,  y_val.to(device))
        acc_synth = _acc(pr_synth, y_val.to(device))

        cm_real   = _confusion(pr_real,  y_val, K)
        cm_synth  = _confusion(pr_synth, y_val, K)

        bacc_real  = _balanced_accuracy(cm_real)
        bacc_synth = _balanced_accuracy(cm_synth)
        f1_real    = _macro_f1(cm_real)
        f1_synth   = _macro_f1(cm_synth)

    # save confusion matrices
    def _plot_cm(cm: np.ndarray, title: str, path: Path):
        plt.figure(figsize=(3.6, 3.2))
        plt.imshow(cm, interpolation='nearest', aspect='auto')
        plt.title(title); plt.colorbar()
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()

    _plot_cm(cm_real,  "Confusion — real features",  plots_dir / "cm_real.png")
    _plot_cm(cm_synth, "Confusion — synth features", plots_dir / "cm_synth.png")

    # persist numbers (JSON + CSV)
    out = dict(
        acc_real=float(acc_real), acc_synth=float(acc_synth),
        bacc_real=float(bacc_real), bacc_synth=float(bacc_synth),
        f1_real=float(f1_real), f1_synth=float(f1_synth),
    )
    with open(out_dir / "probe_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    try:
        import pandas as pd
        pd.DataFrame([out]).to_csv(out_dir / "probe_metrics.csv", index=False)
    except Exception:
        with open(out_dir / "probe_metrics.csv", "w") as fh:
            keys = list(out.keys())
            fh.write(",".join(keys) + "\n")
            fh.write(",".join(str(out[k]) for k in keys) + "\n")

    with open(out_dir / "README.txt", "a") as f:
        f.write(
            "\nLinear probe trained on REAL pooled features from D.extract_features.\n"
            "Evaluated on REAL and SYNTH (generated under same condition sequences).\n"
            "acc/bacc/f1 on SYNTH close to REAL indicates label structure is preserved.\n"
        )
    return out


def run_epoch_evaluations(
    G, D,
    train_loader, val_loader,
    cfg, device,
    epoch: int, ema=None
) -> Dict[str, float]:
    """
    Run distribution metrics and label-structure probe for this epoch,
    save artifacts under: sample_dir/eval_epoch_XXX/,
    and return a compact dict with headline numbers.
    """
    # --- Distribution metrics (optionally with EMA weights) ---
    if bool(getattr(cfg, "use_ema", False)) and (ema is not None):
        ema.apply_to(G)
    dist_results = eval_distribution_epoch(
        G=G,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        epoch=epoch,
        out_root=cfg.sample_dir,
        channels=getattr(cfg, "eval_channels", ("eda","resp","ecg")),
        n_batches=int(getattr(cfg, "eval_n_batches", 8)),
        standardize=bool(getattr(cfg, "eval_standardize", False)),
        bootstrap=int(getattr(cfg, "eval_bootstrap", 0)),
        viz_n=int(getattr(cfg, "eval_viz_n", 4)),
        # new knobs read from cfg (optional)
        use_clamp=bool(getattr(cfg, "eval_use_clamp", True)),
        use_denorm=bool(getattr(cfg, "eval_denorm", False)),
    )
    if bool(getattr(cfg, "use_ema", False)) and (ema is not None):
        ema.restore(G)

    # extract headline numbers directly from dict (no pandas needed)
    headline = {}
    def _pick(ch, key):
        try:
            return float(dist_results.get(ch, {}).get(key))
        except Exception:
            return None
    headline["ks_eda"]  = _pick("eda",  "ks")
    headline["w1_ecg"]  = _pick("ecg",  "w1")
    headline["js_resp"] = _pick("resp", "js")

    # --- Label-structure probe (also with EMA weights for G) ---
    if bool(getattr(cfg, "use_ema", False)) and (ema is not None):
        ema.apply_to(G)
    probe_out = eval_label_probe(
        G=G, D=D,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg, device=device,
        epoch=epoch,
        out_root=cfg.sample_dir,
        # keep default batch counts unless you want to override here
        n_train=None, n_val=None,
        n_train_batches=int(getattr(cfg, "probe_train_batches", 8)),
        n_val_batches=int(getattr(cfg, "probe_val_batches", 8)),
    )
    if bool(getattr(cfg, "use_ema", False)) and (ema is not None):
        ema.restore(G)

    if isinstance(probe_out, dict):
        headline.update({
            "probe_acc_real":  probe_out.get("acc_real"),
            "probe_acc_fake":  probe_out.get("acc_synth"),
            "probe_bacc_real": probe_out.get("bacc_real"),
            "probe_bacc_fake": probe_out.get("bacc_synth"),
            "probe_f1_real":   probe_out.get("f1_real"),
            "probe_f1_fake":   probe_out.get("f1_synth"),
        })

    # persist one-line summary for this epoch
    out_dir = Path(cfg.sample_dir) / f"eval_epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(headline, f, indent=2)

    return headline