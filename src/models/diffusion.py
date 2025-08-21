from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: timestep embedding
# -----------------------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    From OpenAI diffusion utilities: build sinusoidal embeddings.
    timesteps: (B,) int or float tensor of diffusion steps
    returns: (B, dim) float
    """
    half = dim // 2
    # Compute inverse frequencies
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


# --------------------------------
# Basic building blocks for UNet1D
# --------------------------------
def default_groups(num_channels: int) -> int:
    # GroupNorm groups default: min(32, channels), but at least 1
    return max(1, min(32, num_channels))


class FiLM(nn.Module):
    """
    Computes (scale, shift) given time and class embeddings.
    Each ResBlock has its own FiLM to map embeddings to 2*out_channels.
    """
    def __init__(self, emb_dim: int, out_channels: int):
        super().__init__()
        self.lin_time = nn.Linear(emb_dim, 2 * out_channels)
        self.lin_cond = nn.Linear(emb_dim, 2 * out_channels)

    def forward(self, t_emb: torch.Tensor, c_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # t_emb, c_emb: (B, emb_dim)
        h = self.lin_time(t_emb) + self.lin_cond(c_emb)  # (B, 2*out_channels)
        scale, shift = torch.chunk(h, 2, dim=1)          # (B, out_channels) each
        return scale, shift


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU and per-block FiLM (scale/shift) from time + condition embeddings.
    """
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, kernel_size: int = 3, groups: Optional[int] = None):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g1 = nn.GroupNorm(default_groups(in_channels) if groups is None else groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.film = FiLM(emb_dim, out_channels)
        self.g2 = nn.GroupNorm(default_groups(out_channels) if groups is None else groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), t_emb: (B, E), c_emb: (B, E)
        h = self.conv1(F.silu(self.g1(x)))  # (B, out, T)
        scale, shift = self.film(t_emb, c_emb)  # (B, out), (B, out)
        # Broadcast along time
        h = self.g2(h)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Downsample1D(nn.Module):
    """
    Exact downsample by integer factor s using stride=s, kernel_size=s (no padding).
    Input length must be divisible by s.
    """
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.s = stride
        self.conv = nn.Conv1d(channels, channels, kernel_size=stride, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert T % self.s == 0, f"Downsample expects length divisible by stride={self.s}, got T={T}."
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    Exact upsample by integer factor s using ConvTranspose1d with stride=s, kernel_size=s (no padding).
    """
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.s = stride
        self.deconv = nn.ConvTranspose1d(channels, channels, kernel_size=stride, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class UNet1D(nn.Module):
    """
    Generic 1D U-Net for time series, with FiLM-conditioned residual blocks.
    - in_channels: # signal channels
    - base_channels: base number of channels
    - channel_mults: per-level multipliers (length = len(downs) + 1)
    - num_res_blocks: residual blocks per level
    - downs: list of integer strides for each downsampling stage (e.g., [2,2,2])
    - cond_dim: K (class one-hot dim)
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        channel_mults: Optional[Sequence[int]] = None,
        num_res_blocks: int = 2,
        downs: Optional[Sequence[int]] = None,
        cond_dim: int = 4,
        time_emb_dim: Optional[int] = None,
        out_channels: Optional[int] = None,
        cond_drop_prob: float = 0.0,
        groups: Optional[int] = None,
    ):
        super().__init__()
        assert in_channels > 0 and cond_dim > 0
        if downs is None:
            downs = [2, 2, 2]  # default 3 levels of downsample
        levels = len(downs) + 1

        if channel_mults is None:
            # modest growth to keep memory sane (e.g., [1,2,4,8] if 3 downs)
            channel_mults = tuple(2 ** i for i in range(levels))
        assert len(channel_mults) == levels, "channel_mults length must equal len(downs)+1"

        time_emb_dim = time_emb_dim or (base_channels * 4)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4 // 3),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4 // 3, time_emb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.cond_drop_prob = float(cond_drop_prob)

        self.in_conv = nn.Conv1d(in_channels, base_channels * channel_mults[0], kernel_size=3, padding=1)

        # Build down path
        chs: List[int] = []  # track for skip connections
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels * channel_mults[0]
        for level in range(levels):
            out_ch = base_channels * channel_mults[level]
            res = nn.ModuleList([ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim=time_emb_dim, groups=groups)
                                 for i in range(num_res_blocks)])
            self.down_blocks.append(res)
            chs.append(out_ch)
            if level < len(downs):
                self.down_blocks.append(Downsample1D(out_ch, stride=downs[level]))
                in_ch = out_ch  # next level in_ch equals out_ch

        # Middle block
        mid_ch = base_channels * channel_mults[-1]
        self.mid1 = ResBlock(mid_ch, mid_ch, emb_dim=time_emb_dim, groups=groups)
        self.mid2 = ResBlock(mid_ch, mid_ch, emb_dim=time_emb_dim, groups=groups)

        # Build up path (mirror of down)
        self.up_blocks = nn.ModuleList()
        up_in_ch = mid_ch
        for level in reversed(range(levels)):
            skip_ch = chs[level]
            out_ch = base_channels * channel_mults[level]
            # first, upsample except at the topmost level
            if level < len(downs):
                self.up_blocks.append(Upsample1D(up_in_ch, stride=downs[level]))
            # then residual blocks after concatenating skip
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                in_channels_block = up_in_ch + skip_ch if i == 0 else out_ch
                blocks.append(ResBlock(in_channels_block, out_ch, emb_dim=time_emb_dim, groups=groups))
            self.up_blocks.append(blocks)
            up_in_ch = out_ch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(default_groups(up_in_ch) if groups is None else groups, up_in_ch),
            nn.SiLU(),
            nn.Conv1d(up_in_ch, out_channels if out_channels is not None else in_channels, kernel_size=3, padding=1),
        )

        # Remember down strides to shape the up path
        self.downs = list(downs)
        self.num_res_blocks = num_res_blocks
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim

    @staticmethod
    def _reduce_condition(y_or_seq: torch.Tensor) -> torch.Tensor:
        """
        Accept (B,K) or (B,T,K) and reduce to (B,K).
        We assume condition is constant across time; take the first step if sequence given.
        """
        if y_or_seq.dim() == 3:
            return y_or_seq[:, 0, :]  # (B,K)
        elif y_or_seq.dim() == 2:
            return y_or_seq
        else:
            raise ValueError(f"Condition must be (B,K) or (B,T,K). Got shape {tuple(y_or_seq.shape)}")

    def forward(self, x: torch.Tensor, t: torch.Tensor, y_or_seq: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        t: (B,) long/int timesteps
        y_or_seq: (B, K) or (B, T, K)
        returns noise prediction with same shape as x
        """
        assert x.dim() == 3, "UNet1D expects (B, C, T)"
        B, C, T = x.shape
        # Build embeddings
        t_emb = timestep_embedding(t, self.time_emb_dim)        # (B, E)
        t_emb = self.time_mlp(t_emb)                            # (B, E)
        y = self._reduce_condition(y_or_seq).to(x.dtype)        # (B, K)
        if self.training and self.cond_drop_prob > 0.0:
            drop = (torch.rand(B, device=x.device) < self.cond_drop_prob).unsqueeze(1)
            y = torch.where(drop, torch.zeros_like(y), y)
        c_emb = self.cond_mlp(y)



        c_emb = self.cond_mlp(y)                                # (B, E)

        # Down path
        h = self.in_conv(x)
        skip_stack: List[torch.Tensor] = []

        it = iter(self.down_blocks)
        for level in range(len(self.downs) + 1):
            # residual blocks at this level
            res_blocks = next(it)
            for rb in res_blocks:
                h = rb(h, t_emb, c_emb)
                skip_stack.append(h)
            # downsample except after last level
            if level < len(self.downs):
                ds = next(it)
                h = ds(h)

        # Middle
        h = self.mid1(h, t_emb, c_emb)
        h = self.mid2(h, t_emb, c_emb)

        # Up path (mirror)
        up_it = iter(self.up_blocks)
        for level in reversed(range(len(self.downs) + 1)):
            if level < len(self.downs):
                up = next(up_it)       # Upsample
                h = up(h)
            blocks = next(up_it)       # Residual blocks
            # concatenate with skip(s)
            for i, rb in enumerate(blocks):
                # pop matching skip
                skip = skip_stack.pop()
                # shape align (can differ by at most 1 due to stride arithmetic; pad if needed)
                if skip.shape[-1] != h.shape[-1]:
                    # center crop or pad to match
                    diff = skip.shape[-1] - h.shape[-1]
                    if diff > 0:
                        skip = skip[..., :h.shape[-1]]
                    else:
                        h = F.pad(h, (0, -diff))
                h = torch.cat([h, skip], dim=1) if i == 0 else h
                h = rb(h, t_emb, c_emb)

        out = self.out_conv(h)
        assert out.shape == x.shape, f"UNet1D output shape mismatch: got {out.shape}, expected {x.shape}"
        return out


# --------------------------------------
# Diffusion core: betas and DDPM/DDIM
# --------------------------------------
def make_beta_schedule(num_steps: int, schedule: str = "cosine", linear_start: float = 1e-4, linear_end: float = 2e-2) -> torch.Tensor:
    if schedule.lower() == "linear":
        return torch.linspace(linear_start, linear_end, num_steps, dtype=torch.float32)
    elif schedule.lower() == "cosine":
        # Nichol & Dhariwal cosine schedule
        s = 0.008
        steps = num_steps
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float32) / steps
        alphas_bar = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(min=1e-8, max=0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusion1D(nn.Module):
    """
    Standard DDPM with epsilon prediction objective, DDPM/DDIM sampling,
    and approximate variational bound (NLL) computation.
    """
    def __init__(
        self,
        model: UNet1D,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        device: Optional[torch.device] = None,
        x0_clip_q: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.T = timesteps
        betas = make_beta_schedule(timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Precompute frequently used terms
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Numerical stability: clip extremely tiny values
        posterior_variance = posterior_variance.clamp(min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.device_override = device 
        self.x0_clip_q = float(x0_clip_q)

    def _device(self) -> torch.device:
        if self.device_override is not None:
            return self.device_override
        return next(self.model.parameters()).device


    def _apply_x0_clip(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Clamp predicted x0 (optional) and record how often we'd have clipped.
        - If x0_clip_q > 0: per-batch quantile clamp (your original behavior)
        - Else: fixed threshold clamp using x0_clip_value (default 3.0)
        - If debug_clip_x0 is False: skip clamping entirely (but set frac=0)
        Records:
        self._last_frac_clipped (float in [0,1])
        """
        # Allow the schedule to turn clamping OFF
        if not getattr(self, "debug_clip_x0", True):
            self._last_frac_clipped = 0.0
            if not hasattr(self, "_printed_clip_off_once"):
                print("[x0_clamp] OFF")
                self._printed_clip_off_once = True
            return x0

        B = x0.shape[0]
        use_quantile = float(getattr(self, "x0_clip_q", 0.0)) > 0.0

        if use_quantile:
            # per-batch |x0| quantile, clamped to >=1.0 to avoid tiny thresholds early on
            flat_abs = x0.detach().abs().reshape(B, -1).float()
            if torch.isfinite(flat_abs).all():
                q = float(getattr(self, "x0_clip_q", 0.995))
                s = torch.quantile(flat_abs, q, dim=1)
            else:
                # fallback if NaNs/Infs sneak in
                s = torch.full((B,), float(getattr(self, "x0_clip_value", 3.0)), device=x0.device)
            s = s.clamp(min=1.0).view(B, 1, 1)
        else:
            # fixed clip (used when you set x0_clip_q <= 0)
            s_val = float(getattr(self, "x0_clip_value", 3.0))
            s = torch.full((B, 1, 1), s_val, device=x0.device, dtype=x0.dtype)

        # instrumentation: fraction that would be clipped
        frac = (x0.detach().abs() > s).float().mean().item()
        self._last_frac_clipped = float(frac)

        # (optional) one-time log so you see the typical clamp scale
        if not hasattr(self, "_printed_clip_once"):
            mode = "quantile" if use_quantile else "fixed"
            approx_clip = s.mean().item()
            print(f"[x0_clamp] mode={mode} clip≈{approx_clip:.3f} frac_clipped={frac:.4f}")
            self._printed_clip_once = True

        return x0.clamp(-s, s)
        
    def _cfg_eps(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        if cfg_scale <= 0.0:
            return self._predict_eps(x_t, t, y)
        y_uncond = torch.zeros_like(y)
        eps_c = self._predict_eps(x_t, t, y)
        eps_u = self._predict_eps(x_t, t, y_uncond)
        return (1.0 + cfg_scale) * eps_c - cfg_scale * eps_u

    # --------------- q(x_t | x_0) ---------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse x0 to timestep t using precomputed sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod.
        x0: (B, C, T)
        t:  (B,) int64 in [0, T-1]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise

    # --------------- p_theta(x_{t-1} | x_t) ---------------
    def _predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t, y)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t - self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * eps

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = self._predict_eps(x_t, t, y)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps)
        x0_pred = self._apply_x0_clip(x0_pred)

        # >>> DEBUG clamp to tame early-step explosions (remove later) <<<
        if getattr(self, "debug_clip_x0", True):
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
        
        # Mean of p_theta(x_{t-1}|x_t)
        mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x0_pred
            + self.posterior_mean_coef2[t].view(-1, 1, 1) * x_t
        )
        var = self.posterior_variance[t].view(-1, 1, 1)
        return mean, var

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Single DDPM step from x_t to x_{t-1}
        """
        mean, var = self.p_mean_variance(x_t, t, y)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def ddpm_sample(self, shape, y_or_seq, x_T=None, cfg_scale=0.0):
        self._last_return_std_bad = False
        self._last_return_std_value = None  # optional: for logging
        device = self._device()
        B, C, T = shape
        tag = ("low" if C == 2 else "ecg" if C == 1 else f"C{C}")

        # create starting state first
        if x_T is None:
            x_t = torch.randn(shape, device=device)
        else:
            assert x_T.shape == shape
            x_t = x_T.to(device)

        # NOW it's safe to print init
        _x = x_t.detach()
        print(f"[sampler:init:{tag}] shape={tuple(_x.shape)} "
            f"std={_x.float().std(unbiased=False).item():.6f} "
            f"min={_x.min().item():.3f} max={_x.max().item():.3f}")

        y = y_or_seq
        printed_step1 = False
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            if cfg_scale > 0.0:
    # (optional but symmetric) grab eps scale on the pre-update state
                if not printed_step1:
                    eps_dbg = self._cfg_eps(x_t, t, y, cfg_scale)
                    ab_t = float(self.alphas_cumprod[step].item()) if hasattr(self, "alphas_cumprod") else float("nan")
                    print(f"[ddpm:step1_dbg:{tag}] eps_std={eps_dbg.float().std(unbiased=False).item():.6f}  alpha_bar_t={ab_t:.6e}")

                # your existing CFG update
                eps = self._cfg_eps(x_t, t, y, cfg_scale)
                x0_pred = self.predict_x0_from_eps(x_t, t, eps)
                x0_pred = self._apply_x0_clip(x0_pred)
                if getattr(self, "debug_clip_x0", True):
                    x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
                mean = (self.posterior_mean_coef1[t].view(-1,1,1) * x0_pred
                        + self.posterior_mean_coef2[t].view(-1,1,1) * x_t)
                if (t == 0).all():
                    x_t = mean
                else:
                    noise = torch.randn_like(x_t)
                    var = self.posterior_variance[t].view(-1,1,1)
                    x_t = mean + torch.sqrt(var) * noise

                if not printed_step1:
                    _x = x_t.detach()
                    print(f"[sampler:step1:{tag}] std={_x.float().std(unbiased=False).item():.6f} "
                        f"min={_x.min().item():.3f} max={_x.max().item():.3f}")
                    printed_step1 = True

            else:
                # pre-update eps scale for debugging
                if not printed_step1:
                    eps_dbg = self._predict_eps(x_t, t, y)
                    ab_t = float(self.alphas_cumprod[step].item()) if hasattr(self, "alphas_cumprod") else float("nan")
                    print(f"[ddpm:step1_dbg:{tag}] eps_std={eps_dbg.float().std(unbiased=False).item():.6f}  alpha_bar_t={ab_t:.6e}")

                # perform the update
                x_t = self.p_sample(x_t, t, y)

                # post-update stats (this is the “step1” we care about)
                if not printed_step1:
                    _x = x_t.detach()
                    print(f"[sampler:step1:{tag}] std={_x.float().std(unbiased=False).item():.6f} "
                        f"min={_x.min().item():.3f} max={_x.max().item():.3f}")
                    printed_step1 = True

        _x0 = x_t.detach()
        ret_std = float(_x0.float().std(unbiased=False).item())
        self._last_return_std_value = ret_std  # optional, helpful for logs
        thr = float(getattr(self, "ret_std_max", 3.0))

        if not math.isfinite(ret_std):
            self._last_return_std_bad = True
        else:
            self._last_return_std_bad = (ret_std > thr)

        print(f"[sampler:return:{tag}] std={ret_std:.6f} thr={thr:.3f} "
      f"min={_x0.min().item():.3f} max={_x0.max().item():.3f}")
        return x_t
    

    @staticmethod
    def _schedule_head_skip_torch(ddim_alpha_bar_start: float, grid: torch.Tensor):
        ab_start = max(float(ddim_alpha_bar_start), 5e-4)
        i = int(torch.searchsorted(grid, torch.tensor(ab_start, device=grid.device), right=False).item())
        max_head_skip = max(0, grid.numel() - 1)
        head_skip = max(2, min(i, max_head_skip))
        return ab_start, head_skip

    # --------------- DDIM sampling (fast) ---------------
    def _make_ddim_timesteps(self, num_steps: int) -> torch.Tensor:
        # Uniform subsampling of original steps
        if num_steps >= self.T:
            return torch.arange(self.T - 1, -1, -1, dtype=torch.long, device=self._device())
        c = self.T // num_steps
        # Pick evenly spaced timesteps (inclusive) and reverse
        ts = torch.arange(self.T - 1, -1, -c, dtype=torch.long, device=self._device())
        if ts[-1] != 0:
            ts = torch.cat([ts, torch.zeros(1, dtype=torch.long, device=self._device())], dim=0)
        return ts


    @torch.no_grad()
    def ddim_sample(self, shape, y_or_seq, num_steps=50, eta=0.0, x_T=None, cfg_scale=0.0):
        self._last_return_std_bad = False
        self._last_return_std_value = None  # optional: for logging
        device = self._device()
        B, C, T = shape
        tag = ("low" if C == 2 else "ecg" if C == 1 else f"C{C}")

        # create starting state first
        if x_T is None:
            x_t = torch.randn(shape, device=device)
        else:
            assert x_T.shape == shape
            x_t = x_T.to(device)

        # NOW print init
        _x = x_t.detach()
        print(f"[sampler:init:{tag}] shape={tuple(_x.shape)} "
            f"std={_x.float().std(unbiased=False).item():.6f} "
            f"min={_x.min().item():.3f} max={_x.max().item():.3f}")

        y = y_or_seq
        
        ddim_ts = self._make_ddim_timesteps(num_steps)  # e.g., [999, 979, ..., 0]

        # Build ᾱ for the planned steps in the *same order* you’ll iterate.
        alphas_cumprod = self.alphas_cumprod.to(device)
        alpha_bar_grid = alphas_cumprod[ddim_ts]  # should be ascending along iteration order

        if __debug__:
            assert torch.all(alpha_bar_grid[1:] >= alpha_bar_grid[:-1]), "alpha_bar_grid must be ascending"


        # Enforce ᾱ floor + minimum head-skip via the helper
        desired_ab_start = float(getattr(self, "ddim_alpha_bar_start", 0.0))
        ab_start, head_skip = GaussianDiffusion1D._schedule_head_skip_torch(desired_ab_start, alpha_bar_grid)

        # Keep model in sync (useful for logs/externals)
        self.ddim_alpha_bar_start = float(ab_start)
        self.head_skip = int(head_skip)

        # Actually drop the steps
        if head_skip > 0:
            kept_t  = int(ddim_ts[head_skip].item())
            kept_ab = float(alpha_bar_grid[head_skip].item())
            print(f"[ddim] head-skip={head_skip} (ᾱ_start={ab_start:.3e} -> keep t={kept_t}, ᾱ={kept_ab:.3e})")
            ddim_ts = ddim_ts[head_skip:]
            alpha_bar_grid = alpha_bar_grid[head_skip:]

        for i, step in enumerate(ddim_ts):
            t = torch.full((B,), step.item(), device=device, dtype=torch.long)
            eps = self._cfg_eps(x_t, t, y, cfg_scale) if cfg_scale > 0.0 else self._predict_eps(x_t, t, y)
            alpha_bar_t = self.alphas_cumprod[step]
            x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = self._apply_x0_clip(x0_pred)
            if getattr(self, "debug_clip_x0", True):
                x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            if i == 0:
                # schedule scalars for the very first step
                ab_t = float(alpha_bar_t.item())
                print(
                    f"[ddim:step1_dbg:{tag}] "
                    f"eps_std={eps.float().std(unbiased=False).item():.6f}  "
                    f"alpha_bar_t={ab_t:.6e}  "
                    f"sqrt_ab_t={math.sqrt(ab_t):.6e}  "
                    f"sqrt_1m_ab_t={math.sqrt(max(1.0 - ab_t, 0.0)):.6e}"
                )

            if step == 0:
                x_t = x0_pred
                if i == 0:  # rare case num_steps==1
                    _x = x_t.detach()
                    print(f"[sampler:step1:{tag}] std={_x.float().std(unbiased=False).item():.6f} "
                        f"min={_x.min().item():.3f} max={_x.max().item():.3f}")
                continue

            step_prev = ddim_ts[i + 1]
            alpha_bar_prev = self.alphas_cumprod[step_prev]

            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            noise = torch.randn_like(x_t) if eta != 0.0 else torch.zeros_like(x_t)
            x_t = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps + sigma * noise

            if i == 0:
                _x = x_t.detach()
                print(f"[sampler:step1:{tag}] std={_x.float().std(unbiased=False).item():.6f} "
                    f"min={_x.min().item():.3f} max={_x.max().item():.3f}")

        # final print reflects what you're returning
        _x0 = x_t.detach()
        ret_std = float(_x0.float().std(unbiased=False).item())
        self._last_return_std_value = ret_std  # optional, helpful for logs
        thr = float(getattr(self, "ret_std_max", 3.0))

        if not math.isfinite(ret_std):
            self._last_return_std_bad = True
        else:
            self._last_return_std_bad = (ret_std > thr)

        print(f"[sampler:return:{tag}] std={ret_std:.6f} thr={thr:.3f} "
            f"min={_x0.min().item():.3f} max={_x0.max().item():.3f}")

        return x_t

    # --------------- Training loss ---------------
    def loss(self, x0: torch.Tensor, y_or_seq: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard epsilon-prediction MSE.
        x0: (B,C,T)
        y_or_seq: (B,K) or (B,T,K)
        t: optional (B,) steps to use; else uniform random in [0, T-1]
        """
        device = self._device()
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self._predict_eps(x_t, t, y_or_seq)
        loss = F.mse_loss(eps_pred, noise)
        return loss

    # --------------- VLB / NLL (approx) ---------------
    @torch.no_grad()
    def nll_bound(self, x0: torch.Tensor, y_or_seq: torch.Tensor, num_steps_eval: Optional[int] = None) -> Dict[str, float]:
        """
        Approximate variational lower bound (in nats) and bits/dim using:
        - KL[q(x_{t-1}|x_t,x0) || p_theta(x_{t-1}|x_t)] terms (t=1..T-1)
        - "decoder" term L_0 ~= -log p_theta(x0|x1) with Monte Carlo for x1 ~ q(x1|x0)
        - prior term L_T = KL[q(x_T|x0) || N(0,I)] closed-form
        We evaluate on a subset of timesteps if num_steps_eval is provided; then scale to T.
        """
        device = self._device()
        B, C, Tlen = x0.shape
        dims_per_sample = C * Tlen

        # Prior term L_T (closed form KL of two diagonals)
        alpha_bar_T = self.alphas_cumprod[-1]
        # KL(N(m,sigma^2) || N(0,1)) summed over dims
        # q(x_T|x0): mean = sqrt(alpha_bar_T) x0, var = (1 - alpha_bar_T)
        var_qT = (1.0 - alpha_bar_T)  # scalar tensor
        LT = 0.5 * (
            (alpha_bar_T * (x0 ** 2)).sum(dim=(1, 2))
            + dims_per_sample * (var_qT - torch.log(var_qT) - 1.0)
        )

        # Decoder term L_0 ~= -log p_theta(x0|x1) with fixed variance beta_1
        beta1 = self.betas[0]
        t1 = torch.full((B,), 1, device=device, dtype=torch.long)
        x1 = self.q_sample(x0, t1, torch.randn_like(x0))
        eps1 = self._predict_eps(x1, t1, y_or_seq)
        x0_pred_1 = self.predict_x0_from_eps(x1, t1, eps1)
        L0 = 0.5 * (((x0 - x0_pred_1) ** 2) / beta1 + math.log(2 * math.pi) + torch.log(beta1)).sum(dim=(1, 2))  # (B,)

        # KL sum over t=2..T (we index 1..T-1 in 0-based)
        # Optionally subsample timesteps to approximate
        total_steps = self.T - 1
        if num_steps_eval is None or num_steps_eval >= total_steps:
            t_indices = torch.arange(1, self.T, device=device, dtype=torch.long)  # 1..T-1
            scale = 1.0
        else:
            t_indices = torch.linspace(1, self.T - 1, steps=num_steps_eval, device=device).round().long().unique(sorted=True)
            scale = float(total_steps) / float(len(t_indices))

        kl_sum = torch.zeros(B, device=device)
        for t in t_indices:
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            # Sample x_t ~ q(x_t|x0)
            x_t = self.q_sample(x0, t_batch, torch.randn_like(x0))
            # Means of q and p_theta
            # q mean uses true x0
            mean_q = (
                self.posterior_mean_coef1[t_batch].view(-1, 1, 1) * x0
                + self.posterior_mean_coef2[t_batch].view(-1, 1, 1) * x_t
            )
            var_q = self.posterior_variance[t_batch].view(-1, 1, 1)  # = var_p if we use fixed variance
            # p_theta mean uses predicted x0
            eps_pred = self._predict_eps(x_t, t_batch, y_or_seq)
            x0_pred = self.predict_x0_from_eps(x_t, t_batch, eps_pred)
            mean_p = (
                self.posterior_mean_coef1[t_batch].view(-1, 1, 1) * x0_pred
                + self.posterior_mean_coef2[t_batch].view(-1, 1, 1) * x_t
            )
            # KL between two Gaussians with identical diagonal variance: (1/(2σ^2)) * ||μ_q - μ_p||^2
            kl = 0.5 * (((mean_q - mean_p) ** 2) / var_q).sum(dim=(1, 2))  # (B,)
            kl_sum += kl

        kl_sum = kl_sum * scale

        nll_nats_per_sample = LT + L0 + kl_sum  # (B,)
        nll_nats = nll_nats_per_sample.mean().item()
        bits_per_dim = nll_nats / (math.log(2.0) * dims_per_sample)
        return {"nll_nats": float(nll_nats), "bits_per_dim": float(bits_per_dim)}

    # --------------- Public sampling API ---------------
    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int], y_or_seq: torch.Tensor, method: str = "ddpm", num_steps: Optional[int] = None, x_T: Optional[torch.Tensor] = None,
               cfg_scale: float = 0.0) -> torch.Tensor:
        if method.lower() == "ddpm":
            return self.ddpm_sample(shape, y_or_seq, x_T=x_T, cfg_scale=cfg_scale)
        elif method.lower() == "ddim":
            steps = num_steps if num_steps is not None else min(self.T, 50)
            return self.ddim_sample(shape, y_or_seq, num_steps=steps, eta=0.0, x_T=x_T, cfg_scale=cfg_scale)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


# ----------------------------------------------------
# Convenience wrappers (keep external shapes as (B,T,C))
# ----------------------------------------------------
class _BaseTwoStreamWrapper(nn.Module):
    """
    Shared wrapper utilities: permute shapes between (B,T,C) <-> (B,C,T)
    """
    def __init__(self, diffusion: GaussianDiffusion1D, seq_length: int, in_channels: int, cond_dim: int):
        super().__init__()
        self.diff = diffusion
        self.Tlen = seq_length
        self.C = in_channels
        self.K = cond_dim

    @staticmethod
    def _to_bct(x_btc: torch.Tensor) -> torch.Tensor:
        assert x_btc.dim() == 3, "Expected (B,T,C)"
        return x_btc.permute(0, 2, 1).contiguous()

    @staticmethod
    def _to_btc(x_bct: torch.Tensor) -> torch.Tensor:
        assert x_bct.dim() == 3, "Expected (B,C,T)"
        return x_bct.permute(0, 2, 1).contiguous()

    def loss(self, x0: torch.Tensor, y_or_seq: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        x0_bct = self._to_bct(x0)
        return self.diff.loss(x0_bct, y_or_seq, t)

    @torch.no_grad()
    def sample(self, y_or_seq: torch.Tensor, num_steps: Optional[int] = None, method: str = "ddpm", x_T: Optional[torch.Tensor] = None,
               cfg_scale: float = 0.0) -> torch.Tensor:
        B = y_or_seq.shape[0]
        shape = (B, self.C, self.Tlen)
        x_bct = self.diff.sample(shape, y_or_seq, method=method, num_steps=num_steps,
                                 x_T=None if x_T is None else self._to_bct(x_T),
                                 cfg_scale=cfg_scale)
        return self._to_btc(x_bct)

    @torch.no_grad()
    def nll_bound(self, x0: torch.Tensor, y_or_seq: torch.Tensor, num_steps_eval: Optional[int] = None) -> Dict[str, float]:
        x0_bct = self._to_bct(x0)
        return self.diff.nll_bound(x0_bct, y_or_seq, num_steps_eval=num_steps_eval)


class DiffusionLow(_BaseTwoStreamWrapper):
    """
    Low-rate stream diffusion for (T=120, C=2).
    Defaults are lightweight but configurable.
    """
    def __init__(
        self,
        condition_dim: int,
        base_channels: int = 32,
        num_res_blocks: int = 2,
        downs: Sequence[int] = (2, 2, 2),            # 120 -> 60 -> 30 -> 15
        channel_mults: Optional[Sequence[int]] = None,
        diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
        device: Optional[torch.device] = None,
        cond_drop_prob: float = 0.0,
        x0_clip_q: float = 0.0,
    ):
        in_channels = 2
        seq_length = 120
        unet = UNet1D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            downs=downs,
            cond_dim=condition_dim,
            out_channels=in_channels,
            cond_drop_prob=cond_drop_prob,
        )
        diff = GaussianDiffusion1D(unet, timesteps=diffusion_steps, beta_schedule=beta_schedule,
                                   device=device, x0_clip_q=x0_clip_q)
        super().__init__(diff, seq_length=seq_length, in_channels=in_channels, cond_dim=condition_dim)


class DiffusionECG(_BaseTwoStreamWrapper):
    """
    High-rate ECG stream diffusion for (T=5250, C=1).
    We downsample by [5,5,3]: 5250 -> 1050 -> 210 -> 70 to keep memory/time reasonable.
    """
    def __init__(
        self,
        condition_dim: int,
        base_channels: int = 32,
        num_res_blocks: int = 2,
        downs: Sequence[int] = (5, 5, 3),            # 5250 -> 1050 -> 210 -> 70
        channel_mults: Optional[Sequence[int]] = None,
        diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
        device: Optional[torch.device] = None,
        cond_drop_prob: float = 0.0,
        x0_clip_q: float = 0.0,
    ):
        in_channels = 1
        seq_length = 5250
        # Ensure divisibility by product of downs
        prod = 1
        for d in downs:
            prod *= d
        assert seq_length % prod == 0, f"ECG length {seq_length} must be divisible by product(downs)={prod}"
        unet = UNet1D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            downs=downs,
            cond_dim=condition_dim,
            out_channels=in_channels,
            cond_drop_prob=cond_drop_prob,
        )
        diff = GaussianDiffusion1D(unet, timesteps=diffusion_steps, beta_schedule=beta_schedule,
                                   device=device, x0_clip_q=x0_clip_q)
        super().__init__(diff, seq_length=seq_length, in_channels=in_channels, cond_dim=condition_dim)