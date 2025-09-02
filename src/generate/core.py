from __future__ import annotations

import json, hashlib, os, math, types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import torch

# Your project models (same import pattern used in smoke_test.py)
from models.diffusion import DiffusionLow, DiffusionECG


# ---------------------------
# Small utilities
# ---------------------------
def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _np_broadcast(arr, C):
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    if a.size == 1:
        a = np.full((C,), float(a[0]), dtype=np.float32)
    if a.size != C:
        raise ValueError(f"Normalization vector size {a.size} != channels {C}")
    return a.reshape(1, 1, C)


def _denorm(x: np.ndarray, stats: np.lib.npyio.NpzFile) -> np.ndarray:
    """Apply inverse-normalization using keys in npz."""
    files = set(stats.files)
    y = x.astype(np.float32, copy=True)
    C = y.shape[-1]
    if {"mean", "std"} <= files:
        y = y * _np_broadcast(stats["std"], C) + _np_broadcast(stats["mean"], C)
    elif {"min", "max"} <= files:
        y = y * (_np_broadcast(stats["max"], C) - _np_broadcast(stats["min"], C)) + _np_broadcast(stats["min"], C)
    elif {"scale", "bias"} <= files:
        y = y * _np_broadcast(stats["scale"], C) + _np_broadcast(stats["bias"], C)
    else:
        raise KeyError(f"Unknown norm keys: {files}")
    return y



def _interp_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    B, L, C = arr.shape
    if L == target_len:
        return arr.copy()
    xp = np.linspace(0.0, 1.0, L, endpoint=True)
    xq = np.linspace(0.0, 1.0, target_len, endpoint=True)
    out = np.empty((B, target_len, C), dtype=np.float32)
    for n in range(B):
        for c in range(C):
            out[n, :, c] = np.interp(xq, xp, arr[n, :, c].astype(np.float64)).astype(np.float32)
    return out


@dataclass
class CheckpointBundle:
    ckpt_path: Path
    milestone_json: Path
    norm_low: Path
    norm_ecg: Path
    manifest: dict
    sha_ckpt: str
    sha_manifest: str
    sha_norm_low: str
    sha_norm_ecg: str


# ---------------------------
# WESAD Generator
# ---------------------------
class WESADGenerator:
    """
    Reusable generator for WESAD-like two-stream diffusion models.

    Key APIs:
      - sample_one(condition, T, steps=None, guidance=None, seed=...)
      - sample_batch(condition, T, n_samples, base_seed)
      - write_npz(signals, condition, base_seed, out_dir=...)
      - write_csv(signals, condition, base_seed, out_dir=...)

    Output shape contract: (N, T, 3) with channels [ECG, Resp, EDA].
    """

    def __init__(
        self,
        milestones_dir: Path,
        ckpt_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.milestones_dir = Path(milestones_dir).resolve()
        self.ckpt_path = Path(ckpt_path).resolve() if ckpt_path is not None else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if not self.milestones_dir.exists():
            raise FileNotFoundError(f"Milestones dir not found: {self.milestones_dir}")

        self.bundle = self._assemble_checkpoint_bundle()
        self.low, self.ecg = self._build_and_load_models(self.bundle.manifest)

        # cache native lengths from wrappers
        self.low_len = getattr(self.low, "Tlen", 120)
        self.ecg_len = getattr(self.ecg, "Tlen", 5250)

    # ---------- bundle discovery ----------
    def _assemble_checkpoint_bundle(self) -> CheckpointBundle:
        """
        Choose a checkpoint and find the matching milestone JSON + norms.
        If ckpt_path is provided, try to match by SHA to a milestone *.pt; otherwise, use it directly
        and pick a reasonable milestone JSON (same epoch or single best guess).
        """
        # 1) Resolve ckpt
        if self.ckpt_path is None:
            # default to .../diffusion/final.ckpt
            default = (self.milestones_dir.parent / "final.ckpt").resolve()
            if not default.exists():
                raise FileNotFoundError("No --ckpt provided and final.ckpt not found.")
            ckpt = default
        else:
            ckpt = self.ckpt_path
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        sha_ckpt = sha256_file(ckpt)

        # 2) Try to find a milestone whose .pt matches this SHA (works for hardlinks/copies)
        match_pt: Optional[Path] = None
        match_json: Optional[Path] = None
        for pt in self.milestones_dir.glob("milestone_*.pt"):
            if sha256_file(pt) == sha_ckpt:
                match_pt = pt
                js = pt.with_suffix(".json")
                match_json = js if js.exists() else None
                break

        # 3) Fallback: if no SHA match, pick a milestone json heuristically (largest epoch)
        if match_json is None:
            jsons = sorted(self.milestones_dir.glob("milestone_*.json"))
            if not jsons:
                raise FileNotFoundError(f"No milestone JSONs in {self.milestones_dir}")
            match_json = jsons[-1]

        manifest = _load_json(match_json)
        sha_manifest = sha256_file(match_json)

        # 4) Norms come from the SAME milestones_dir (milestones are self-contained)
        norm_low = self.milestones_dir / "norm_low.npz"
        norm_ecg = self.milestones_dir / "norm_ecg.npz"
        if not norm_low.exists() or not norm_ecg.exists():
            raise FileNotFoundError(f"Missing norm files in {self.milestones_dir} (need norm_low.npz & norm_ecg.npz)")

        sha_norm_low = sha256_file(norm_low)
        sha_norm_ecg = sha256_file(norm_ecg)

        return CheckpointBundle(
            ckpt_path=ckpt,
            milestone_json=match_json,
            norm_low=norm_low,
            norm_ecg=norm_ecg,
            manifest=manifest,
            sha_ckpt=sha_ckpt,
            sha_manifest=sha_manifest,
            sha_norm_low=sha_norm_low,
            sha_norm_ecg=sha_norm_ecg,
        )
    
    def _iter_state_dicts(self, state):
        """Yield plausible state_dicts found inside a loaded checkpoint object."""
        root = self._norm_container(state)
        # Paired containers (common)
        if isinstance(root, dict):
            for key in ("ema_low", "low", "diff_low", "model_low",
                        "ema_ecg", "ecg", "diff_ecg", "model_ecg"):
                if key in root:
                    sd = self._maybe_state_dict(root[key])
                    if sd:
                        yield sd
            # Nested 'ema'
            if "ema" in root:
                ema = self._norm_container(root["ema"])
                if isinstance(ema, dict):
                    for key in ("low", "diff_low", "ecg", "diff_ecg"):
                        if key in ema:
                            sd = self._maybe_state_dict(ema[key])
                            if sd:
                                yield sd
            # Flat map at top-level
            flat = self._maybe_state_dict(root)
            if flat:
                yield flat

    def _infer_condition_dim_from_state(self, state) -> Optional[int]:
        """Scan any found state_dict for cond_mlp.0.weight and return its in_features."""
        for sd in self._iter_state_dicts(state):
            for k, v in sd.items():
                if isinstance(k, str) and k.endswith("cond_mlp.0.weight"):
                    try:
                        return int(v.shape[1])  # Linear(out, in): in == condition_dim
                    except Exception:
                        pass
        return None

    # ---------- model load ----------
    def _build_and_load_models(self, manifest: dict) -> Tuple[DiffusionLow, DiffusionECG]:

            # Load once (we'll reuse this object)
        state = self._torch_load_safe(self.bundle.ckpt_path)
        meta  = state.get("meta", {}) if isinstance(state, dict) else {}

        # Strongest signal: read condition_dim from actual weights if present
        cond_from_ckpt = self._infer_condition_dim_from_state(state)

        # Order of preference: inferred from weights → checkpoint meta → milestone manifest → default(3)
        condition_dim   = int(
            (cond_from_ckpt
            if cond_from_ckpt is not None
            else meta.get("condition_dim", manifest.get("condition_dim", 3)))
        )
        diffusion_steps = int(meta.get("diffusion_steps", manifest.get("diffusion_steps", 1000)))
        beta_schedule   = str(meta.get("beta_schedule",   manifest.get("beta_schedule", "cosine")))
        cond_drop_prob  = float(meta.get("cond_drop_prob", manifest.get("cond_drop_prob", 0.0)))
        x0_clip_q       = float(meta.get("x0_clip_q",       manifest.get("x0_clip_q", 0.0)))

        low = DiffusionLow(
            condition_dim=condition_dim,
            diffusion_steps=diffusion_steps,
            beta_schedule=beta_schedule,
            cond_drop_prob=cond_drop_prob,
            x0_clip_q=x0_clip_q,
            device=self.device,
        ).to(self.device)

        ecg = DiffusionECG(
            condition_dim=condition_dim,
            diffusion_steps=diffusion_steps,
            beta_schedule=beta_schedule,
            cond_drop_prob=cond_drop_prob,
            x0_clip_q=x0_clip_q,
            device=self.device,
        ).to(self.device)

        # Load checkpoint with PyTorch 2.6+ safety compatibility
        state = self._torch_load_safe(self.bundle.ckpt_path)

        # Heuristic: prefer EMA weights if present; otherwise use regular
        loaded = False
        sd_root = self._norm_container(state)
        flat_sd = None

        # Paired containers
        for a, b in [("ema_low", "ema_ecg"), ("low", "ecg"), ("diff_low", "diff_ecg"), ("model_low", "model_ecg")]:
            if isinstance(sd_root, dict) and (a in sd_root) and (b in sd_root):
                lsd = self._maybe_state_dict(sd_root[a]); esd = self._maybe_state_dict(sd_root[b])
                if lsd and esd:
                    low.load_state_dict(lsd, strict=False)
                    ecg.load_state_dict(esd, strict=False)
                    loaded = True
                    break

        # Nested 'ema'
        if not loaded and isinstance(sd_root, dict) and ("ema" in sd_root):
            ema = self._norm_container(sd_root["ema"])
            if isinstance(ema, dict):
                for a, b in [("low", "ecg"), ("diff_low", "diff_ecg")]:
                    if (a in ema) and (b in ema):
                        lsd = self._maybe_state_dict(ema[a]); esd = self._maybe_state_dict(ema[b])
                        if lsd and esd:
                            low.load_state_dict(lsd, strict=False)
                            ecg.load_state_dict(esd, strict=False)
                            loaded = True
                            break

        # Flat prefixes
        if not loaded:
            flat_sd = self._maybe_state_dict(sd_root)
            if flat_sd:
                for lp, ep in [
                    ("ema_low.", "ema_ecg."),
                    ("low.", "ecg."),
                    ("diff_low.", "diff_ecg."),
                    ("module.ema_low.", "module.ema_ecg."),
                    ("module.low.", "module.ecg."),
                    ("module.diff_low.", "module.diff_ecg."),
                ]:
                    lsd = {k[len(lp):]: v for k, v in flat_sd.items() if isinstance(k, str) and k.startswith(lp)}
                    esd = {k[len(ep):]: v for k, v in flat_sd.items() if isinstance(k, str) and k.startswith(ep)}
                    if lsd and esd:
                        low.load_state_dict(lsd, strict=False)
                        ecg.load_state_dict(esd, strict=False)
                        loaded = True
                        break

        # Last fallback: same map into both (strict=False)
        if not loaded and flat_sd:
            low.load_state_dict(flat_sd, strict=False)
            ecg.load_state_dict(flat_sd, strict=False)
            loaded = True

        if not loaded:
            keys = list(sd_root.keys()) if isinstance(sd_root, dict) else type(sd_root)
            raise RuntimeError(f"Could not locate low/ecg weights in checkpoint. Top-level: {keys}")




        low.eval(); ecg.eval()

        try:
            self.condition_dim = int(low.diff.model.cond_mlp[0].in_features)
        except Exception:
            # fallback if in_features isn't available
            try:
                self.condition_dim = int(low.diff.model.cond_mlp[0].weight.shape[1])
            except Exception:
                # last resort: fall back to manifest
                self.condition_dim = int(manifest.get("condition_dim", 3))



        return low, ecg

    @staticmethod
    def _torch_load_safe(path: Path) -> Any:
        # PyTorch 2.6 tightened defaults; allow SimpleNamespace if present
        try:
            from torch.serialization import safe_globals
            with safe_globals([types.SimpleNamespace]):
                return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")

    @staticmethod
    def _norm_container(obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, types.SimpleNamespace):
            return vars(obj)
        return obj

    @staticmethod
    def _maybe_state_dict(x):
        x = WESADGenerator._norm_container(x)
        if isinstance(x, dict):
            if "state_dict" in x and isinstance(x["state_dict"], dict):
                return x["state_dict"]
            # flat map: strings->tensors
            if x and all(isinstance(k, str) for k in x.keys()) and any(hasattr(v, "shape") for v in x.values()):
                return x
        if hasattr(x, "state_dict"):
            try:
                return x.state_dict()
            except Exception:
                return None
        return None

    # ---------- public sampling ----------
    def sample_one(
        self,
        condition: str,                 # "baseline" | "stress" | "amusement"
        T: int,                         # must equal self.ecg_len or self.low_len
        steps: Optional[int] = None,    # if None, use manifest["sampling_steps"]
        guidance: Optional[float] = None,  # if None, use manifest["cfg_scale"]
        seed: int = 123,
    ) -> np.ndarray:
        """
        Returns: np.ndarray of shape (1, T, 3), dtype float32 — channels [ECG, Resp, EDA].
        """
        # Determinism
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        # Manifest sampling knobs
        manifest = self.bundle.manifest
        method = str(manifest.get("sampling_method", "ddim")).lower()
        num_steps = int(steps if steps is not None else manifest.get("sampling_steps", 50))
        cfg_scale = float(guidance if guidance is not None else manifest.get("cfg_scale", 0.0))

        # Per-head schedule flags mirrored from manifest (if present)
        if "ddim_alpha_bar_start_low" in manifest:
            self.low.diff.ddim_alpha_bar_start = float(manifest["ddim_alpha_bar_start_low"])
        if "ddim_alpha_bar_start_ecg" in manifest:
            self.ecg.diff.ddim_alpha_bar_start = float(manifest["ddim_alpha_bar_start_ecg"])
        if "debug_clip_x0_low" in manifest:
            self.low.diff.debug_clip_x0 = bool(manifest["debug_clip_x0_low"])
        if "debug_clip_x0_ecg" in manifest:
            self.ecg.diff.debug_clip_x0 = bool(manifest["debug_clip_x0_ecg"])

        # Label → one-hot
        label_idx = {"baseline": 0, "stress": 1, "amusement": 2}.get(condition)
        if label_idx is None:
            raise ValueError(f"Unknown condition '{condition}'")
        K = getattr(self, "condition_dim",int(manifest.get("condition_dim", 3)))  # falls back if not set
        if label_idx >= K:
            raise ValueError(f"condition_dim={K} incompatible with label index {label_idx}")
        cond_vec = torch.zeros(1, K, device=self.device, dtype=torch.float32)
        cond_vec[0, label_idx] = 1.0

        # Sample each head at its native rate
        with torch.no_grad():
            x_low = self.low.sample(cond_vec, num_steps=num_steps, method=method, cfg_scale=cfg_scale)   # (1, 120, 2)
            x_ecg = self.ecg.sample(cond_vec, num_steps=num_steps, method=method, cfg_scale=cfg_scale)   # (1, 5250, 1)

        # De-normalize with milestone stats
        nl = np.load(self.bundle.norm_low)
        ne = np.load(self.bundle.norm_ecg)
        x_low_np = _denorm(x_low.cpu().numpy(), nl)   # (1, 120, 2) -> [Resp, EDA] ordering handled below
        x_ecg_np = _denorm(x_ecg.cpu().numpy(), ne)   # (1, 5250, 1)

        # --- Ensure LOW is ordered as [Resp, EDA] ---
        mapped = False
        # Prefer explicit metadata if present
        if "channels" in nl.files:
            try:
                chs = [str(c).lower() for c in nl["channels"].tolist()]
                resp_idx = chs.index("resp") if "resp" in chs else chs.index("respiration")
                eda_idx  = chs.index("eda")  if "eda"  in chs else chs.index("electrodermal activity")
                x_low_np = x_low_np[..., [resp_idx, eda_idx]]
                mapped = True
            except Exception:
                pass

        # Heuristic fallback (when no metadata): smaller-std → Resp, larger-std → EDA
        if not mapped:
            stds = x_low_np.std(axis=(0, 1))
            # idx_small ~ Resp, idx_large ~ EDA
            idx_small = int(np.argmin(stds))
            idx_large = 1 - idx_small
            x_low_np = x_low_np[..., [idx_small, idx_large]]

        # Safety: clamp EDA non-negative after mapping
        x_low_np[..., 1] = np.clip(x_low_np[..., 1], 0.0, None)

        # Fuse to requested T
        if T == self.ecg_len:
            fused = np.concatenate([x_ecg_np.astype(np.float32), _interp_to_len(x_low_np, self.ecg_len)], axis=-1)
        elif T == self.low_len:
            fused = np.concatenate([_interp_to_len(x_ecg_np, self.low_len), x_low_np.astype(np.float32)], axis=-1)
        else:
            raise ValueError(
                f"T={T} must equal ECG length ({self.ecg_len}) or LOW length ({self.low_len}). "
                f"Use {self.ecg_len} to preserve ECG morphology."
            )

        # Final checks
        fused = fused.astype(np.float32, copy=False)
        if fused.shape != (1, T, 3):
            raise ValueError(f"Output shape {fused.shape} != (1,{T},3)")
        if not np.isfinite(fused).all():
            raise ValueError("Output contains NaN/Inf")

        return fused

    def sample_batch(
        self,
        condition: str,
        T: int,
        n_samples: int,
        base_seed: int,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> np.ndarray:
        """
        Returns: np.ndarray of shape (N, T, 3), dtype float32.
        Seeds are incremented deterministically: base_seed, base_seed+1, ...
        """
        xs = []
        for i in range(int(n_samples)):
            x = self.sample_one(condition, T, steps=steps, guidance=guidance, seed=int(base_seed) + i)
            xs.append(x)
        out = np.concatenate(xs, axis=0).astype(np.float32)
        # non-flat quick check
        if not (out.std(axis=(0, 1)) > 1e-6).all():
            raise ValueError("Non-flat check failed: per-channel std too small.")
        return out

    # ---------- save utilities ----------
    def write_npz(
        self,
        signals: np.ndarray,           # (N,T,3) float32
        condition: str,
        base_seed: int,
        out_dir: Path | str = "./synthetic",
        filename: Optional[str] = None,
        window_ids: Optional[np.ndarray] = None,
    ) -> Path:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"synthetic_{condition}_seed{int(base_seed)}.npz"
        p = out_dir / filename

        N, T, C = signals.shape
        if window_ids is None:
            window_ids = np.arange(N, dtype=np.int32)

        np.savez_compressed(
            p,
            signals=signals.astype(np.float32, copy=False),
            channels=np.array(["ECG", "Resp", "EDA"], dtype=object),
            condition=str(condition),
            window_ids=window_ids.astype(np.int32, copy=False),
        )

        if p.stat().st_size <= 0:
            raise IOError("NPZ written but file is empty.")
        return p

    def write_csv(
        self,
        signals: np.ndarray,   # (N,T,3) float32
        condition: str,
        base_seed: int,
        out_dir: Path | str = "./synthetic",
        filename: Optional[str] = None,
        time_mode: str = "index",   # "index" or "seconds"
    ) -> Path:
        """
        CSV columns: time, ECG, Resp, EDA, condition, window_id
        time_mode:
          - "index": 0..T-1
          - "seconds": derives from native timeline (5250@175Hz ≈ 30 s; 120@4Hz ≈ 30 s)
        """
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"synthetic_{condition}_seed{int(base_seed)}.csv"
        p = out_dir / filename

        N, T, _ = signals.shape
        # choose fs based on T
        if T == self.ecg_len:
            fs = 175.0
        elif T == self.low_len:
            fs = 4.0
        else:
            # unknown; fall back to index
            fs = None

        with p.open("w", encoding="utf-8") as f:
            f.write("time,ECG,Resp,EDA,condition,window_id\n")
            for n in range(N):
                if time_mode == "seconds" and fs is not None:
                    t_vec = np.arange(T, dtype=np.float32) / fs
                else:
                    t_vec = np.arange(T, dtype=np.float32)
                ecg = signals[n, :, 0]
                resp = signals[n, :, 1]
                eda = signals[n, :, 2]
                for t_i in range(T):
                    f.write(f"{t_vec[t_i]:.6f},{ecg[t_i]:.6f},{resp[t_i]:.6f},{eda[t_i]:.6f},{condition},{n}\n")

        if p.stat().st_size <= 0:
            raise IOError("CSV written but file is empty.")
        return p

    # ---------- convenience: full dataset ----------
    def generate_dataset(
        self,
        condition: str,
        T: int,
        n_samples: int,
        base_seed: int,
        out_format: str = "npz",       # "npz" or "csv"
        out_dir: Path | str = "./synthetic",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> Path:
        """
        End-to-end dataset generation with quality checks and file write.
        Returns path to the written file and prints a DONE line.
        """
        signals = self.sample_batch(
            condition=condition, T=T, n_samples=int(n_samples), base_seed=int(base_seed),
            steps=steps, guidance=guidance
        )

        # Validate shape/channels/dtype/finite
        if signals.ndim != 3 or signals.shape[2] != 3:
            raise ValueError(f"Expected (N,T,3), got {signals.shape}")
        if signals.dtype != np.float32:
            signals = signals.astype(np.float32, copy=False)
        if not np.isfinite(signals).all():
            raise ValueError("Signals contain NaN/Inf.")

        # Write
        if out_format.lower() == "npz":
            path = self.write_npz(signals, condition, base_seed, out_dir=out_dir)
        elif out_format.lower() == "csv":
            path = self.write_csv(signals, condition, base_seed, out_dir=out_dir, time_mode="index")
        else:
            raise ValueError("out_format must be 'npz' or 'csv'.")

        print(f"DONE: synthetic dataset generated at {path}")
        return path
    
    