# src/datasets/wesad.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ---------- helpers ----------
def normalize_channelwise(data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Standardize last-dimension channels: (N, T, C) -> z = (x - mu)/sigma."""
    return (data - means[None, None, :]) / (stds[None, None, :] + 1e-8)

def _looks_standardized(arr: np.ndarray, tol_mean: float = 0.2, tol_std: float = 0.3) -> bool:
    """
    Heuristic: returns True if per-channel mean ~ 0 and std ~ 1.
    """
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    m = flat.mean(axis=0)
    s = flat.std(axis=0)
    return (np.all(np.abs(m) < tol_mean) and np.all(np.abs(s - 1.0) < tol_std))

def _get_stat(d: np.lib.npyio.NpzFile, primary: str, fallbacks: list[str]) -> np.ndarray:
    if primary in d:
        return d[primary].astype(np.float32)
    for k in fallbacks:
        if k in d:
            return d[k].astype(np.float32)
    raise KeyError(f"Missing '{primary}' in stats (tried { [primary]+fallbacks })")


# ---------- dataset ----------
class WESADSequenceDataset(Dataset):
    """
    Returns per sample:
      • 'signal_low':  FloatTensor (T_low, 2)   # EDA, RESP @ 4 Hz
      • 'signal_ecg':  FloatTensor (T_ecg, 1)   # ECG @ 175 Hz
      • 'condition' :  FloatTensor (T_low, K)   # per-timestep one-hot
      • optional 'label': LongTensor () if scalar labels present
    """

    def __init__(
        self,
        root_dir: str | Path,
        fold: str = "tc_multigan_fold_S10",
        split: str = "train",
        window_size_low: int = 240,      # steps @4 Hz
        condition_dim: int = 4,
        # --- normalization / augmentation controls ---
        normalize: bool = True,
        normalize_ecg: bool = True,
        stats_low_path: Optional[str] = None,   # train stats (low): npz with mean,std
        stats_ecg_path: Optional[str] = None,   # train stats (ecg): npz with mean,std
        force_use_stats: bool = False,          # ignore heuristic; always use provided stats if paths exist
        use_split_stats_if_needed: bool = True, # if not standardized and no stats provided → compute split stats
        augment: bool = False,
        aug_jitter: float = 0.01,
        aug_scale: float = 0.0,
        expected_ecg_len: Optional[int] = None, # assert/crop if set
        debug_print: bool = False,

        


    ) -> None:
        root = Path(root_dir).expanduser().resolve()
        self.fold = fold
        self.split = split
        self.window_size_low = window_size_low
        self.condition_dim   = condition_dim

        self.normalize = normalize
        self.normalize_ecg = normalize_ecg
        self.force_use_stats = force_use_stats
        self.use_split_stats_if_needed = use_split_stats_if_needed
        self.augment = augment
        self.aug_jitter = aug_jitter
        self.aug_scale = aug_scale
        self.expected_ecg_len = expected_ecg_len
        self.debug_print = debug_print

        # ---- file paths ----
        files = {
            "X_low": root / fold / f"{split}_X_low.npy",
            "X_ecg": root / fold / f"{split}_X_ecg.npy",
            "m1":    root / fold / f"{split}_m1_seq.npy",
        }

        for name, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} file: {path}")
        # optional scalar window labels
        file_cond = root / fold / f"{split}_cond.npy"

        # ---- load arrays (N, T, C) ----
        self.x_low_np  = np.load(files["X_low"]).astype(np.float32)   # (N, T_low, 2)
        self.x_ecg_np  = np.load(files["X_ecg"]).astype(np.float32)   # (N, T_ecg, 1)
        self.m1_low_np = np.load(files["m1"]).astype(np.float32)      # (N, T_low, K)

        # sanity check: condition dimensionality matches config
        if self.m1_low_np.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition_dim mismatch: m1_seq last dim={self.m1_low_np.shape[-1]} "
                f"but dataset was constructed with condition_dim={self.condition_dim}"
        )

        # optional scalar window labels
        self.cond_np: Optional[np.ndarray] = None
        self.cond_scalar = False
        if file_cond.exists():
            raw = np.load(file_cond)
            self.cond_scalar = (raw.ndim == 1)
            self.cond_np     = raw

        # ---- optional ECG length check/crop ----
        if self.expected_ecg_len is not None and self.x_ecg_np.shape[1] != self.expected_ecg_len:
            # simple center-crop if longer; assert if shorter
            T = self.x_ecg_np.shape[1]
            target = int(self.expected_ecg_len)
            if T > target:
                start = (T - target) // 2
                self.x_ecg_np = self.x_ecg_np[:, start:start+target, :]
            else:
                raise ValueError(f"ECG length {T} < expected {target}")

        # ----------------- NORMALIZATION BLOCK (deterministic) -----------------
        # Always prefer provided stats; optionally compute split stats; otherwise raise.
        # Store stats on the instance for debugging and reproducibility.
        self.low_mean: Optional[np.ndarray]  = None
        self.low_std:  Optional[np.ndarray]  = None
        self.ecg_mean: Optional[np.ndarray]  = None
        self.ecg_std:  Optional[np.ndarray]  = None

        def _coerce_vec(v: np.ndarray, C: int, name: str) -> np.ndarray:
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size == 1:
                v = np.repeat(v, C)
            if v.size != C:
                raise ValueError(f"{name}: expected {C} channels, got size {v.size} (shape {v.shape})")
            return v

        def _load_stats(path_like: Optional[str | Path], C: int, tag: str):
            if path_like is None:
                raise FileNotFoundError(f"[{tag}] stats path not provided.")
            p = Path(path_like).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"[{tag}] stats file not found: {p}")
            with np.load(p) as d:
                mu  = _get_stat(d, "mean", ["mu", f"{tag}_mean", "avg", "m"])
                std = _get_stat(d, "std",  ["sigma", f"{tag}_std", "sd", "stddev", "s"])
            mu  = _coerce_vec(mu,  C, f"[{tag}] mean")
            std = _coerce_vec(std, C, f"[{tag}] std")
            std = np.where(std <= 0, 1.0, std).astype(np.float32)
            return mu.astype(np.float32), std.astype(np.float32)

        # ---- low-rate (EDA, RESP; C=2) ----
        if self.normalize:
            if self.force_use_stats:
                mu_low, sd_low = _load_stats(stats_low_path, 2, "low")
            elif stats_low_path:
                mu_low, sd_low = _load_stats(stats_low_path, 2, "low")
            elif self.use_split_stats_if_needed:
                flat = self.x_low_np.reshape(-1, self.x_low_np.shape[-1]).astype(np.float32)
                mu_low = flat.mean(axis=0)
                sd_low = flat.std(axis=0)
                sd_low = np.where(sd_low <= 0, 1.0, sd_low).astype(np.float32)
            else:
                raise FileNotFoundError("[low] No stats provided and split-normalization disabled.")
            self.low_mean, self.low_std = mu_low, sd_low
            self.x_low_np = normalize_channelwise(self.x_low_np, self.low_mean, self.low_std)
            if self.debug_print:
                f = self.x_low_np.reshape(-1, self.x_low_np.shape[-1])
                print(f"[{self.split}] low after norm mean={f.mean(0)} std={f.std(0)}")

        # ---- ECG (C=1) ----
        if self.normalize_ecg:
            if self.force_use_stats:
                mu_ecg, sd_ecg = _load_stats(stats_ecg_path, 1, "ecg")
            elif stats_ecg_path:
                mu_ecg, sd_ecg = _load_stats(stats_ecg_path, 1, "ecg")
            elif self.use_split_stats_if_needed:
                flat = self.x_ecg_np.reshape(-1, self.x_ecg_np.shape[-1]).astype(np.float32)
                mu_ecg = flat.mean(axis=0)
                sd_ecg = flat.std(axis=0)
                sd_ecg = np.where(sd_ecg <= 0, 1.0, sd_ecg).astype(np.float32)
            else:
                raise FileNotFoundError("[ecg] No stats provided and split-normalization disabled.")
            self.ecg_mean, self.ecg_std = mu_ecg, sd_ecg
            self.x_ecg_np = normalize_channelwise(self.x_ecg_np, self.ecg_mean, self.ecg_std)
            if self.debug_print:
                f = self.x_ecg_np.reshape(-1, self.x_ecg_np.shape[-1])
                print(f"[{self.split}] ecg after norm mean={f.mean(0)} std={f.std(0)}")

        if self.debug_print:
            print(f"normalize flags: {self.normalize} {self.normalize_ecg} force: {self.force_use_stats}")
            print(f"loaded low_std: {None if self.low_std is None else self.low_std.tolist()}")
            print(f"loaded ecg_std: {None if self.ecg_std is None else self.ecg_std.tolist()}")
            fL = self.x_low_np.reshape(-1, self.x_low_np.shape[-1])
            fE = self.x_ecg_np.reshape(-1, self.x_ecg_np.shape[-1])
            print(f"dataset arrays (post-init) stds low: {np.round(fL.std(0),4)} ecg: {np.round(fE.std(0),4)}")
        # ----------------------------------------------------------------------

        if self.debug_print and not (self.normalize or self.normalize_ecg):
            def _report(tag, arr):
                flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
                m, s = flat.mean(0), flat.std(0)
                print(f"[{self.split}] {tag} assumed pre-standardized: mean={np.round(m,4)} std={np.round(s,4)}")
                if not (np.all(np.abs(m) < 0.2) and np.all(np.abs(s - 1.0) < 0.3)):
                    print(f"[{self.split}] WARNING: {tag} does not look z-scored but normalize=False (Option A).")
            _report("low", self.x_low_np)
            _report("ecg", self.x_ecg_np)

        # verify matching lengths
        N = len(self.x_low_np)
        assert len(self.x_ecg_np)  == N, "low/ecg length mismatch"
        assert len(self.m1_low_np) == N, "low/m1 length mismatch"
        if self.cond_np is not None:
            assert len(self.cond_np) == N, "labels length mismatch"

    def __len__(self) -> int:
        return len(self.x_low_np)

    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1) get windows
        x_low  = self.x_low_np[idx]    # (T_low, 2)
        x_ecg  = self.x_ecg_np[idx]    # (T_ecg, 1)
        m1_low = self.m1_low_np[idx]   # (T_low, K)

        # 2) optional sub-window on low-rate with aligned ECG crop
        T_low0 = x_low.shape[0]
        T_ecg0 = x_ecg.shape[0]

        # Use float ratio; do NOT round here
        ratio_f = T_ecg0 / float(max(T_low0, 1))

        if T_low0 > self.window_size_low:
            # choose start on LOW stream
            if self.split == "train":
                start_low = np.random.randint(0, T_low0 - self.window_size_low + 1)
            else:
                # deterministic center crop for val/test
                start_low = (T_low0 - self.window_size_low) // 2

            end_low = start_low + self.window_size_low

            # crop low + condition
            x_low  = x_low[start_low:end_low]
            m1_low = m1_low[start_low:end_low]

            # aligned ECG crop using SAME start scaled by ratio_f
            L_ecg_target = int(round(self.window_size_low * ratio_f))
            L_ecg_target = min(L_ecg_target, T_ecg0)   # cap to length

            start_ecg = int(round(start_low * ratio_f))
            start_ecg = max(0, min(start_ecg, T_ecg0 - L_ecg_target))
            end_ecg   = start_ecg + L_ecg_target

            x_ecg = x_ecg[start_ecg:end_ecg]

        elif T_low0 < self.window_size_low:
            # You can pad instead; current behavior is to error
            raise ValueError(f"Low-rate window ({T_low0}) shorter than required ({self.window_size_low}).")

        else:
            # T_low0 == window_size_low: low is already the target length.
            # Make ECG length consistent with rounded float ratio.
            L_ecg_target = int(round(T_low0 * ratio_f))
            L_ecg_target = min(L_ecg_target, T_ecg0)
            if T_ecg0 > L_ecg_target:
                s = (T_ecg0 - L_ecg_target) // 2
                x_ecg = x_ecg[s:s+L_ecg_target]

        # --- tiny guards ---
        if x_ecg.ndim == 1:
            x_ecg = x_ecg[:, None]

        # Assert invariants (optional but helpful)
        target_low_len = min(T_low0, self.window_size_low)
        target_ecg_len = min(T_ecg0, int(round(target_low_len * ratio_f)))
        assert x_low.shape[0] == target_low_len, (x_low.shape, target_low_len)
        assert x_ecg.shape[0] == target_ecg_len, (x_ecg.shape, target_ecg_len)
        # -------------------

        # 3) augmentation (low-rate only) — train/eval controlled by self.augment
        if self.augment:
            jitter = np.random.randn(*x_low.shape).astype(np.float32) * self.aug_jitter
            scale  = (1.0 + np.random.uniform(-self.aug_scale, self.aug_scale, size=(1, x_low.shape[1]))).astype(np.float32)
            x_low = x_low * scale + jitter

        # 4) to tensors
        sig_low  = torch.from_numpy(x_low).float()
        sig_ecg  = torch.from_numpy(x_ecg).float()
        cond_low = torch.from_numpy(m1_low).float()

        sample: Dict[str, Any] = {
            "signal_low":  sig_low,
            "signal_ecg":  sig_ecg,
            "condition":   cond_low,
        }
        if self.cond_np is not None and self.cond_scalar:
            lab = int(self.cond_np[idx])
            sample["label"] = torch.tensor(lab, dtype=torch.long)
            
        return sample



# ---------- loader ----------
def make_loader(
    root_dir: str | Path,
    fold: str,
    split: str,
    window_size_low: int,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    weighted_sampling: bool = False,
    condition_dim: int = 4,
    # new controls (optional)
    augment: bool = True,
    normalize: bool = False,
    normalize_ecg: bool = False,
    stats_low_path: Optional[str] = None,
    stats_ecg_path: Optional[str] = None,
    expected_ecg_len: Optional[int] = None,
    debug_print: bool = False,
    force_use_stats: bool = False,
    use_split_stats_if_needed: bool = False,
    persistent_workers: bool = True
) -> DataLoader:
    ds = WESADSequenceDataset(
        root_dir=root_dir,
        fold=fold,
        split=split,
        window_size_low=window_size_low,
        condition_dim=condition_dim,
        augment=augment if split == "train" else False,
        normalize=normalize,
        normalize_ecg=normalize_ecg,
        stats_low_path=stats_low_path,
        stats_ecg_path=stats_ecg_path,
        expected_ecg_len=expected_ecg_len,
        debug_print=debug_print,
        force_use_stats=force_use_stats,
        use_split_stats_if_needed=use_split_stats_if_needed,
    )

    pw = persistent_workers if num_workers > 0 else False
    drop_last = (split == "train")

    if weighted_sampling and ds.cond_np is not None and ds.cond_scalar:
        labels = ds.cond_np
        class_counts = np.bincount(labels)
        weights = 1.0 / np.maximum(class_counts[labels], 1)
        sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=pw, drop_last=drop_last)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, persistent_workers=pw, pin_memory=torch.cuda.is_available())