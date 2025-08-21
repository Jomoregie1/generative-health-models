from __future__ import annotations
import argparse, json, hashlib, os
from pathlib import Path
from typing import Tuple
import numpy as np
import types

# ------------- Utilities ------------- #

def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def find_matching_milestone(milestones_dir: Path, final_ckpt_sha: str) -> Tuple[Path, Path]:
    """
    Find the milestone .pt whose SHA-256 matches final.ckpt, and return (pt_path, json_path).
    This works for both copies and hardlinks (Windows) since the bytes are identical.
    """
    candidates = sorted(milestones_dir.glob("milestone_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No milestone .pt files in: {milestones_dir}")
    for pt in candidates:
        if sha256_file(pt) == final_ckpt_sha:
            js = pt.with_suffix(".json")
            if not js.exists():
                raise FileNotFoundError(f"Found matching .pt but missing JSON: {js}")
            return pt, js
    raise RuntimeError("No milestone .pt matches final.ckpt by SHA-256; ensure you picked the right milestones folder.")

def ensure_norms(milestones_dir: Path) -> Tuple[Path, Path]:
    low = milestones_dir / "norm_low.npz"
    ecg = milestones_dir / "norm_ecg.npz"
    if not low.exists():
        raise FileNotFoundError(f"Missing norm_low.npz in {milestones_dir}")
    if not ecg.exists():
        raise FileNotFoundError(f"Missing norm_ecg.npz in {milestones_dir}")
    return low, ecg

# ------------- YOUR PROJECT INTEGRATION ------------- #
def sample_wesad_window(*,
                        final_ckpt: Path,
                        manifest: dict,
                        norm_low: Path,
                        norm_ecg: Path,
                        condition: str,
                        T: int,
                        steps: int,
                        guidance: float,
                        seed: int) -> np.ndarray:
    import numpy as np
    import torch

    # === EDIT THIS IMPORT to your path ===
    # If diffusion.py is at src/models/diffusion.py:
    from src.models.diffusion import DiffusionLow, DiffusionECG  # <-- edit if needed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Instantiate heads with training-time hyperparams (fallbacks if not in manifest) ----
    condition_dim   = int(manifest.get("condition_dim", 4))
    diffusion_steps = int(manifest.get("diffusion_steps", 1000))
    beta_schedule   = str(manifest.get("beta_schedule", "cosine"))
    cond_drop_prob  = float(manifest.get("cond_drop_prob", 0.0))
    x0_clip_q       = float(manifest.get("x0_clip_q", 0.0))

    low = DiffusionLow(
        condition_dim=condition_dim,
        diffusion_steps=diffusion_steps,
        beta_schedule=beta_schedule,
        cond_drop_prob=cond_drop_prob,
        x0_clip_q=x0_clip_q,
        device=torch.device(device),
    ).to(device)

    ecg = DiffusionECG(
        condition_dim=condition_dim,
        diffusion_steps=diffusion_steps,
        beta_schedule=beta_schedule,
        cond_drop_prob=cond_drop_prob,
        x0_clip_q=x0_clip_q,
        device=torch.device(device),
    ).to(device)

    low_len = getattr(low, "Tlen", getattr(low, "seq_length", 120))
    ecg_len = getattr(ecg, "Tlen", getattr(ecg, "seq_length", 5250))

    # ---- Load checkpoint weights (robust to various layouts) ----
    # PyTorch 2.6+ compat: allow full unpickling for trusted checkpoints
    # 1) Load with PyTorch 2.6+ compatibility
    try:
        from torch.serialization import safe_globals
        with safe_globals([types.SimpleNamespace]):
            state = torch.load(final_ckpt, map_location="cpu", weights_only=True)
    except Exception:
        try:
            state = torch.load(final_ckpt, map_location="cpu", weights_only=False)  # trusted file
        except TypeError:
            state = torch.load(final_ckpt, map_location="cpu")  # older torch
    
    

    def _norm(obj):
        """Return a mapping-like view for dict / SimpleNamespace; else return obj unchanged."""
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, types.SimpleNamespace):
            return vars(obj)
        return obj

    def _maybe_sd(x):
        """Extract a state_dict from various container forms."""
        x = _norm(x)
        # nested {'state_dict': {...}}
        if isinstance(x, dict):
            if "state_dict" in x and isinstance(x["state_dict"], dict):
                return x["state_dict"]
            # heuristic: flat state_dict (str keys -> tensors)
            if x and all(isinstance(k, str) for k in x.keys()) and any(hasattr(v, "shape") for v in x.values()):
                return x
        # nn.Module
        if hasattr(x, "state_dict"):
            try:
                return x.state_dict()
            except Exception:
                return None
        return None

    sd_root = _norm(state)
    loaded = False
    flat_sd = None  # keep a flat dict if we find one

    # 2) Try obvious paired containers first (EMA preferred)
    for a, b in [("ema_low", "ema_ecg"), ("low", "ecg"), ("diff_low", "diff_ecg"), ("model_low", "model_ecg")]:
        if isinstance(sd_root, dict) and (a in sd_root) and (b in sd_root):
            lsd = _maybe_sd(sd_root[a]); esd = _maybe_sd(sd_root[b])
            if lsd and esd:
                low.load_state_dict(lsd, strict=False)
                ecg.load_state_dict(esd, strict=False)
                print(f"[load] loaded from pair containers: {a}/{b}")
                loaded = True
                break

    # 3) Try nested 'ema' container
    if not loaded and isinstance(sd_root, dict) and ("ema" in sd_root):
        ema = _norm(sd_root["ema"])
        if isinstance(ema, dict):
            for a, b in [("low", "ecg"), ("diff_low", "diff_ecg")]:
                if (a in ema) and (b in ema):
                    lsd = _maybe_sd(ema[a]); esd = _maybe_sd(ema[b])
                    if lsd and esd:
                        low.load_state_dict(lsd, strict=False)
                        ecg.load_state_dict(esd, strict=False)
                        print(f"[load] loaded from ema.{a}/{b}")
                        loaded = True
                        break

    # 4) Try flat prefixes from a single state_dict
    if not loaded:
        flat_sd = _maybe_sd(sd_root)
        if flat_sd:
            for lp, ep in [
                ("ema_low.", "ema_ecg."),
                ("low.", "ecg."),
                ("diff_low.", "diff_ecg."),
                ("module.ema_low.", "module.ema_ecg."),
                ("module.low.", "module.ecg."),
                ("module.diff_low.", "module.diff_ecg."),
            ]:
                lsd = {k[len(lp):]: v for k, v in flat_sd.items() if k.startswith(lp)}
                esd = {k[len(ep):]: v for k, v in flat_sd.items() if k.startswith(ep)}
                if lsd and esd:
                    low.load_state_dict(lsd, strict=False)
                    ecg.load_state_dict(esd, strict=False)
                    print(f"[load] loaded from flat prefixes: {lp} / {ep}")
                    loaded = True
                    break

    # 5) Last resort: try the same flat SD for both (strict=False)
    if not loaded and flat_sd:
        low.load_state_dict(flat_sd, strict=False)
        ecg.load_state_dict(flat_sd, strict=False)
        print("[load] loaded with shared flat state_dict (strict=False)")
        loaded = True

    if not loaded:
        # Helpful debug: print top-level keys so we know what’s inside
        keys = list(sd_root.keys()) if isinstance(sd_root, dict) else type(sd_root)
        raise RuntimeError(f"Could not locate low/ecg weights in checkpoint. Top-level: {keys}")


    low.eval(); ecg.eval()

    # ---- Determinism ----
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Conditioning: one-hot [baseline, stress, amusement] of size condition_dim ----
    label_idx = {"baseline": 0, "stress": 1, "amusement": 2}.get(condition)
    if label_idx is None:
        raise ValueError(f"Unknown condition '{condition}'")
    if label_idx >= condition_dim:
        raise ValueError(f"condition_dim={condition_dim} incompatible with label index {label_idx}")
    cond_vec = torch.zeros(1, condition_dim, device=device, dtype=torch.float32)
    cond_vec[0, label_idx] = 1.0

    # ---- Sampler settings from manifest ----
    method = str(manifest.get("sampling_method", "ddim")).lower()
    num_steps = int(steps) if method == "ddim" else None
    # Per-head DDIM alphā start + debug flags (if present)
    if "ddim_alpha_bar_start_low" in manifest:
        low.diff.ddim_alpha_bar_start = float(manifest["ddim_alpha_bar_start_low"])
    if "ddim_alpha_bar_start_ecg" in manifest:
        ecg.diff.ddim_alpha_bar_start = float(manifest["ddim_alpha_bar_start_ecg"])
    if "debug_clip_x0_low" in manifest:
        low.diff.debug_clip_x0 = bool(manifest["debug_clip_x0_low"])
    if "debug_clip_x0_ecg" in manifest:
        ecg.diff.debug_clip_x0 = bool(manifest["debug_clip_x0_ecg"])

    # ---- Per-head sampling (wrappers return (B,T,C)) ----
    with torch.no_grad():
        x_low = low.sample(cond_vec, num_steps=num_steps, method=method, cfg_scale=float(guidance))  # (1, low_len, 2)
        x_ecg = ecg.sample(cond_vec, num_steps=num_steps, method=method, cfg_scale=float(guidance))  # (1, ecg_len, 1)

    # ---- De-normalization using provided norms ----
    def _broadcast(arr, C):
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if a.size == 1: a = np.full((C,), float(a[0]), dtype=np.float32)
        if a.size != C: raise ValueError(f"norm size {a.size} != channels {C}")
        return a.reshape(1, 1, C)

    def _denorm(x: np.ndarray, npz: np.lib.npyio.NpzFile) -> np.ndarray:
        files = set(npz.files)
        y = x.astype(np.float32, copy=True)
        C = y.shape[-1]
        if {"mean", "std"} <= files:
            y = y * _broadcast(npz["std"], C) + _broadcast(npz["mean"], C)
        elif {"min", "max"} <= files:
            y = y * (_broadcast(npz["max"], C) - _broadcast(npz["min"], C)) + _broadcast(npz["min"], C)
        elif {"scale", "bias"} <= files:
            y = y * _broadcast(npz["scale"], C) + _broadcast(npz["bias"], C)
        else:
            raise KeyError(f"Unknown norm keys in {npz.files}")
        return y

    x_low = _denorm(x_low if isinstance(x_low, np.ndarray) else x_low.cpu().numpy(), np.load(norm_low))  # (1, L, 2)
    x_ecg = _denorm(x_ecg if isinstance(x_ecg, np.ndarray) else x_ecg.cpu().numpy(), np.load(norm_ecg))  # (1, E, 1)

    # Optional: enforce [Resp, EDA] order if norm_low has 'channels'
    nl = np.load(norm_low)
    if "channels" in nl.files:
        try:
            chs = [str(c).lower() for c in nl["channels"].tolist()]
            resp_idx = chs.index("resp") if "resp" in chs else chs.index("respiration")
            eda_idx  = chs.index("eda")  if "eda"  in chs else chs.index("electrodermal activity")
            x_low = x_low[..., [resp_idx, eda_idx]]
        except Exception:
            pass

    # Safety: clamp EDA to non-negative after de-norm
    x_low[..., 1] = np.clip(x_low[..., 1], 0.0, None)

    # ---- Fuse to requested T (choose ECG or LOW native length) ----
    def _interp_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
        B, L, C = arr.shape
        if L == target_len: return arr.astype(np.float32)
        xp = np.linspace(0.0, 1.0, num=L, endpoint=True)
        xq = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
        out = np.empty((B, target_len, C), dtype=np.float32)
        for c in range(C):
            out[0, :, c] = np.interp(xq, xp, arr[0, :, c].astype(np.float64)).astype(np.float32)
        return out

    if T == ecg_len:
        fused = np.concatenate([x_ecg.astype(np.float32), _interp_to_len(x_low, ecg_len)], axis=-1)
    elif T == low_len:
        fused = np.concatenate([_interp_to_len(x_ecg, low_len), x_low.astype(np.float32)], axis=-1)
    else:
        raise ValueError(
            f"--duration (T={T}) must equal ECG length ({ecg_len}) or LOW length ({low_len}). "
            f"Use {ecg_len} to keep ECG at its training rate."
        )

    # Final checks
    fused = fused.astype(np.float32, copy=False)
    if fused.shape != (1, T, 3):
        raise ValueError(f"Fused output shape {fused.shape} != (1, {T}, 3)")
    if not np.isfinite(fused).all():
        raise ValueError("Output contains NaN/Inf")

    return fused

# ------------- Main ------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestones", type=Path, required=True,
                    help="Path to .../results/checkpoints/diffusion/milestones")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Path to final.ckpt (default: <milestones>/.. /final.ckpt)")
    ap.add_argument("--condition", required=True, choices=["baseline","stress","amusement"])
    ap.add_argument("--duration", type=int, required=True, help="T_test (timesteps per window)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--override_steps", type=int, default=None, help="Override sampling_steps from manifest")
    ap.add_argument("--override_guidance", type=float, default=None, help="Override cfg_scale from manifest")
    ap.add_argument("--plot_dir", type=Path, default=None, help="Optional: save quick preview plots here")
    args = ap.parse_args()

    milestones_dir = args.milestones.resolve()
    final_ckpt = (args.ckpt or (milestones_dir.parent / "final.ckpt")).resolve()

    if not final_ckpt.exists():
        raise FileNotFoundError(f"final.ckpt not found: {final_ckpt}")

    # Compute SHA for final.ckpt and locate the matching milestone
    ckpt_sha = sha256_file(final_ckpt)
    pt_path, json_path = find_matching_milestone(milestones_dir, ckpt_sha)
    json_sha = sha256_file(json_path)

    # Norm files from the SAME milestone folder
    norm_low, norm_ecg = ensure_norms(milestones_dir)
    norm_low_sha = sha256_file(norm_low)
    norm_ecg_sha = sha256_file(norm_ecg)

    # Read config from manifest
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    sampling_method = manifest.get("sampling_method", "ddim")
    steps = int(args.override_steps if args.override_steps is not None else manifest.get("sampling_steps", 50))
    guidance = float(args.override_guidance if args.override_guidance is not None else manifest.get("cfg_scale", 0.5))

    # --- Generate one window ---
    x = sample_wesad_window(
        final_ckpt=final_ckpt,
        manifest=manifest,
        norm_low=norm_low,
        norm_ecg=norm_ecg,
        condition=args.condition,
        T=int(args.duration),
        steps=steps,
        guidance=guidance,
        seed=int(args.seed),
    )

    # --- Checks ---
    if not isinstance(x, np.ndarray):
        raise TypeError("Sampler must return a NumPy array")
    if x.shape != (1, args.duration, 3):
        raise ValueError(f"Bad output shape {x.shape}; expected (1, {args.duration}, 3)")
    if x.dtype != np.float32:
        raise TypeError(f"Bad dtype {x.dtype}; expected float32")
    if not np.isfinite(x).all():
        raise ValueError("Output contains NaN/Inf")
    # Non-flat quick check
    stds = x.std(axis=(0,1))
    if not (stds > 1e-6).all():
        raise ValueError(f"Non-flat check failed; per-channel std={stds}")

    # Optional plots
    if args.plot_dir is not None:
        import matplotlib.pyplot as plt  # only if requested
        outdir = args.plot_dir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        L = min(1000, args.duration)
        t = np.arange(L)
        for i, ch in enumerate(["ECG","Resp","EDA"]):
            plt.figure()
            plt.plot(t, x[0, :L, i])
            plt.title(f"{ch} (first {L} samples)")
            plt.xlabel("sample")
            plt.ylabel("a.u.")
            plt.tight_layout()
            plt.savefig(outdir / f"smoke_{ch.lower()}.png", dpi=120)
            plt.close()

    # --- Provenance log ---
    print("\nSmoke Test — Final Checkpoint")
    print("Artifacts")
    print(f"- final.ckpt:     {final_ckpt}")
    print(f"- milestone.json: {json_path}")
    print(f"- norm_low.npz:   {norm_low}")
    print(f"- norm_ecg.npz:   {norm_ecg}")

    print("\nSHA-256")
    print(f"- final.ckpt:     {ckpt_sha}")
    print(f"- milestone.json: {json_sha}")
    print(f"- norm_low.npz:   {norm_low_sha}")
    print(f"- norm_ecg.npz:   {norm_ecg_sha}")

    print("\nConfig (from manifest)")
    print(f"- sampling_method: {sampling_method}")
    print(f"- sampling_steps:  {steps}")
    print(f"- cfg_scale:       {guidance}")

    print("\nGeneration settings")
    print(f"- condition:       {args.condition}")
    print(f"- T_test:          {args.duration}")
    print(f"- seed:            {args.seed}")

    print("\nResults")
    print(f"- output shape:    {x.shape}  [PASS]")
    print(f"- channel order:   [ECG, Resp, EDA]  [ASSUMED BY CONTRACT]  [PASS]")
    print(f"- dtype:           {x.dtype}  [PASS]")
    print(f"- finite values:   no NaNs/Infs  [PASS]")
    print(f"- non-flat std:    {stds}")

if __name__ == "__main__":
    main()
