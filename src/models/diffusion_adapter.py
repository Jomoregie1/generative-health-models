from __future__ import annotations
import torch.nn as nn
import torch

class TwoStreamDiffusionAdapter(nn.Module):
    """
    
    Two-stream diffusion generator adapter.

    Unifies two diffusion heads (LOW: EDA/RESP, ECG) behind a single
    generator-like call so existing code can do:

        fake_low, fake_ecg = adapter(z, cond_low)

    Notes:
    - 'z' is unused (kept for API compatibility).
    - cond_low: (B, 120, K) one-hot sequence for the low-rate head.
    - ECG uses a pooled per-window condition: cond_low[:, 0, :] → (B, K).
    - Sampling knobs (steps, method, cfg_scale) are read from cfg.
    - train()/eval() and device handling are propagated to both submodules.

    Returns:
    - fake_low: (B, 120, 2)  # EDA, RESP @ 4 Hz
    - fake_ecg: (B, 5250, 1) # ECG @ 175 Hz
    
    """
    def __init__(self, diff_low, diff_ecg, cfg, device):
        super().__init__()
        self.low = diff_low
        self.ecg = diff_ecg
        self.cfg = cfg
        self.dev = device

    def train(self, mode: bool = True):
        super().train(mode)
        # propagate to underlying models (safe if they are nn.Module)
        if hasattr(self.low, "train"): self.low.train(mode)
        if hasattr(self.ecg, "train"): self.ecg.train(mode)
        return self

    def eval(self):
        super().eval()
        if hasattr(self.low, "eval"): self.low.eval()
        if hasattr(self.ecg, "eval"): self.ecg.eval()
        return self

    @torch.no_grad()
    def forward(self, z, cond_low):
        """
        cond_low: (B,120,K) → reduce to (B,K) for ECG; keep full seq for low stream.
        Returns:
          fake_low: (B, 120, 2)
          fake_ecg: (B, 5250, 1)
        """
        # Ensure inputs live on the right device
        cond_low = cond_low.to(self.dev)
        y = cond_low[:, 0, :]  # (B,K)
        steps = getattr(self.cfg, "sampling_steps", 50)
        method = getattr(self.cfg, "sampling_method", "ddim")
        cfg_scale = float(getattr(self.cfg, "cfg_scale", 0.0))
        fake_low = self.low.sample(y_or_seq=cond_low, num_steps=steps, method=method, cfg_scale=cfg_scale)
        fake_ecg = self.ecg.sample(y_or_seq=y,         num_steps=steps, method=method, cfg_scale=cfg_scale)
        return fake_low, fake_ecg