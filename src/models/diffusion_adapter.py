from __future__ import annotations
import torch.nn as nn
import torch

class TwoStreamDiffusionAdapter(nn.Module):
    """
    Makes diffusion look like the GAN G:
      G(z, cond_low) -> (fake_low, fake_ecg)
    'z' is ignored. 'cond_low' is (B,120,K).
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