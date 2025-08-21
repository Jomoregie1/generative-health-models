import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def minibatch_std_1d(x, group_size=4, eps=1e-8):
    # x: (N, C, T)
    N, C, T = x.shape
    G = min(group_size, N)
    if N % G != 0:
        G = 1  # fallback when per-GPU batch is small
    y = x.view(G, -1, C, T)                      # (G, N//G, C, T)
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt((y ** 2).mean(dim=0) + eps)   # (N//G, C, T)
    y = y.mean(dim=(1, 2), keepdim=True)         # (N//G, 1, 1)
    y = y.repeat(G, 1, T)                        # (N, 1, T)
    return torch.cat([x, y], dim=1)              # (N, C+1, T)

class ScaledTanh(nn.Module):
    """tanh followed by a constant scale; keeps outputs away from ±1."""
    def __init__(self, scale: float = 0.90):
        super().__init__()
        self.scale = float(scale)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.scale * self.tanh(x)

def boundary_loss(x: torch.Tensor, margin) -> torch.Tensor:
    """
    Penalize amplitudes that exceed |margin|.
    `margin` can be a float or a tensor that broadcasts over x (e.g. [C] or [1,1,C]).
    Call this on the signals you feed to D (i.e., before any external clamp).
    """
    if not torch.is_tensor(margin):
        margin = torch.tensor(margin, dtype=x.dtype, device=x.device)
    # Broadcast margin to x's shape
    while margin.dim() < x.dim():
        margin = margin.unsqueeze(0)
    return F.relu(x.abs() - margin).mean()


class Embedder(nn.Module):
    """
    E: Multi-stream embedder for TimeGAN.

    Inputs:
      - x_low : (B, T_low, 2)    # low-rate EDA+RESP
      - x_ecg : (B, T_ecg, 1) or None

    Output:
      - h     : (B, H, T_latent) where T_latent = T_low // latent_downsample

    Design:
      • Low-rate path: two stride-2 convs (by default) to reach T_latent.
      • ECG path: light conv -> anti-aliased decimation toward T_latent
                  using fixed depthwise smoothing + avg-pool steps (7,5,3,2),
                  then a final linear interpolation to exactly T_latent.
      • Fusion: concat(low, ecg) → 1x1 conv → GELU → H channels.
    """
    def __init__(
        self,
        low_channels: int = 2,
        ecg_channels: int = 1,
        hidden_dim: int = 256,
        seq_length_low: int = 120,
        seq_length_ecg: int = 5280,
        latent_downsample: int = 4,
        use_ecg: bool = True,
    ):
        super().__init__()
        assert latent_downsample in (1, 2, 4, 8), "latent_downsample should be 1,2,4,8"
        self.low_channels = low_channels
        self.ecg_channels = ecg_channels
        self.hidden_dim = hidden_dim
        self.seq_length_low = seq_length_low
        self.seq_length_ecg = seq_length_ecg
        self.latent_downsample = latent_downsample
        self.use_ecg = use_ecg

        # --- Low-rate down path: stride-2 conv blocks to reach T_low/latent_downsample
        ch1 = max(64, hidden_dim // 4)
        ch2 = max(128, hidden_dim // 2)
        blocks = []
        in_ch = low_channels
        num_down = int(round(math.log2(latent_downsample))) if latent_downsample > 1 else 0
        for i in range(num_down):
            out_ch = ch1 if i == 0 else (ch2 if i == 1 else hidden_dim)
            blocks += [nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1), nn.GELU()]
            in_ch = out_ch
        if in_ch != hidden_dim:
            blocks += [nn.Conv1d(in_ch, hidden_dim, kernel_size=1), nn.GELU()]
        self.low_down = nn.Sequential(*blocks) if blocks else nn.Identity()

        # --- ECG path: light conv + fixed depthwise LPF used during decimation
        ch_ecg = max(64, hidden_dim // 4)
        self.ecg_in = nn.Conv1d(ecg_channels, ch_ecg, kernel_size=7, padding=3)
        self.ecg_act = nn.GELU()
        self.ecg_smooth = nn.Conv1d(
            ch_ecg, ch_ecg, kernel_size=7, padding=3, groups=ch_ecg, bias=False
        )
        with torch.no_grad():
            k = torch.tensor([1., 6., 15., 20., 15., 6., 1.], dtype=torch.float32)
            k = (k / k.sum()).view(1, 1, -1)
            self.ecg_smooth.weight.copy_(k.repeat(ch_ecg, 1, 1))
        for p in self.ecg_smooth.parameters():
            p.requires_grad_(False)

        # --- Fusion to hidden_dim channels
        fusion_in = hidden_dim + (ch_ecg if use_ecg else 0)
        self.fuse = nn.Sequential(
            nn.Conv1d(fusion_in, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

    @staticmethod
    def _greedy_decimate(x: torch.Tensor, smooth: nn.Module, target_len: int) -> torch.Tensor:
        """
        Anti-aliased greedy decimation toward target_len using pool factors (7,5,3,2).
        Any remaining mismatch is handled by final linear interpolation.
        x: (B, C, T)
        """
        T = x.size(-1)
        for p in (7, 5, 3, 2):                 # largest first to reduce passes
            while T // p >= target_len and T // p >= 2:
                x = smooth(x)
                x = F.avg_pool1d(x, kernel_size=p, stride=p, ceil_mode=False)
                T = x.size(-1)
        return x

    def forward(self, x_low: torch.Tensor, x_ecg: torch.Tensor = None) -> torch.Tensor:
        """
        Returns latent h: (B, H, T_latent)
        """
        B, T_low, C_l = x_low.shape
        assert C_l == self.low_channels, f"Expected low channels={self.low_channels}, got {C_l}"
        T_latent = T_low // max(1, self.latent_downsample)

        # Low path → (B,H,T_latent)
        low = self.low_down(x_low.transpose(1, 2))  # (B, H, T_low/latent_downsample)

        feats = [low]
        if self.use_ecg and x_ecg is not None:
            _, T_ecg, C_e = x_ecg.shape
            assert C_e == self.ecg_channels, f"Expected ecg channels={self.ecg_channels}, got {C_e}"
            ecg = self.ecg_act(self.ecg_in(x_ecg.transpose(1, 2)))  # (B, ch_ecg, T_ecg)
            ecg = self._greedy_decimate(ecg, self.ecg_smooth, target_len=T_latent)
            if ecg.size(-1) != T_latent:
                ecg = F.interpolate(ecg, size=T_latent, mode='linear', align_corners=False)
            feats.append(ecg)

        # Fuse streams and return latent
        h = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]  # (B, H+ch_ecg, T_latent)
        h = self.fuse(h)                                             # (B, H, T_latent)
        return h

    def encode_time_major(self, x_low: torch.Tensor, x_ecg: torch.Tensor = None) -> torch.Tensor:
        """Convenience for Supervisor: (B,T,H) instead of (B,H,T)."""
        return self.forward(x_low, x_ecg).transpose(1, 2)
    
class Recovery(nn.Module):
    """
    R: Multi-stream recovery.
      Input : h (B, H, T_latent)  where T_latent = T_low // latent_downsample
      Outputs:
        - low_out: (B, T_low, 2)
        - ecg_out: (B, T_ecg, 1)

    Low path: anti-aliased upsample x2 then x2 (for latent_downsample=4), refine, head->2ch.
    ECG path: series of anti-aliased upscales {2,3,5,7} towards T_ecg, final
              linear interpolate to exact T_ecg, small refine head->1ch.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        seq_length_low: int = 120,
        seq_length_ecg: int = 5280,
        latent_downsample: int = 4,
        use_ecg: bool = True,
    ):
        super().__init__()
        assert latent_downsample in (1, 2, 4, 8), "latent_downsample must be 1,2,4,8"
        self.hidden_dim        = hidden_dim
        self.seq_length_low    = seq_length_low
        self.seq_length_ecg    = seq_length_ecg
        self.latent_downsample = latent_downsample
        self.use_ecg           = use_ecg

        # ---- Low-rate up path (invert E's downsamples)
        ch1 = max(128, hidden_dim // 2)
        ch2 = max(64,  hidden_dim // 4)
        ups = []
        in_ch = hidden_dim
        if latent_downsample >= 2:
            ups += [AntiAliasUp1D(in_ch, scale=2, ksize=7), nn.Conv1d(in_ch, ch1, 1), nn.GELU()]
            in_ch = ch1
        if latent_downsample >= 4:
            ups += [AntiAliasUp1D(in_ch, scale=2, ksize=5), nn.Conv1d(in_ch, ch2, 1), nn.GELU()]
            in_ch = ch2
        self.low_up = nn.Sequential(*ups) if ups else nn.Identity()
        self.low_head = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, 2,  kernel_size=3, padding=1),
        )

        # ---- ECG up path
        self.ecg_in = nn.Conv1d(hidden_dim, max(64, hidden_dim // 4), kernel_size=1)
        self.ecg_up_ops = nn.ModuleList()
        self._built_for = None  # cache: (T_low, T_ecg)
        self.ecg_refine = nn.Sequential(
            nn.Conv1d(max(64, hidden_dim // 4), 64, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(64, 1,  kernel_size=3, padding=1),
        )

    @staticmethod
    def _factorize_scale(total: int):
        """
        Factorize an integer upscaling ratio into steps from {7,5,3,2}.
        Any leftover (>1 and not in that set) will be handled by final interpolate.
        """
        steps = []
        for p in (7, 5, 3, 2):         # large-to-small is usually smoother
            while total % p == 0 and total > 1:
                steps.append(p)
                total //= p
        if total > 1:
            steps.append(total)        # non-factorable remainder -> final interpolate
        return steps
    
    def _maybe_build_ecg_up(self, T_low: int, T_ecg: int):
        if self._built_for == (T_low, T_ecg):
            return
        self._built_for = (T_low, T_ecg)
        self.ecg_up_ops = nn.ModuleList()

        T_latent = T_low // max(1, self.latent_downsample)
        if not self.use_ecg or T_latent < 2:
            return

        total = max(1, T_ecg // T_latent)
        steps = self._factorize_scale(total)  # e.g., 175 -> [7, 5, 5]
        ch = max(64, self.hidden_dim // 4)

        for s in steps:
            if s in (2, 3, 5, 7):
                self.ecg_up_ops.append(AntiAliasUp1D(ch, scale=int(s), ksize=5))
            else:
                self.ecg_up_ops.append(nn.Identity())
        
        dev = self.ecg_in.weight.device
        for mod in self.ecg_up_ops:
            mod.to(dev)
        
        

    def forward(self, h: torch.Tensor, out_lengths=None):
        """
        h: (B, H, T_latent)
        out_lengths (optional): tuple (T_low, T_ecg) to override defaults
        Returns:
          low_out: (B, T_low, 2)
          ecg_out: (B, T_ecg, 1)  or None if use_ecg=False
        """
        B, H, T_latent = h.shape
        T_low = self.seq_length_low if out_lengths is None else out_lengths[0]
        T_ecg = self.seq_length_ecg if out_lengths is None else out_lengths[1]

        # ---- Low branch
        low = self.low_up(h)                                 # (B, C, ~T_low)
        if low.size(-1) != T_low:
            low = F.interpolate(low, size=T_low, mode='linear', align_corners=False)
        low = self.low_head(low)                             # (B, 2, T_low)
        low_out = low.transpose(1, 2)                        # (B, T_low, 2)

        # ---- ECG branch
        ecg_out = None
        if self.use_ecg:
            self._maybe_build_ecg_up(T_low, T_ecg)
            x = self.ecg_in(h)                    # (B, ch, T_latent)
            for mod in self.ecg_up_ops:
                x = mod(x)                        # registered modules → correct device
            if x.size(-1) != T_ecg:
                x = F.interpolate(x, size=T_ecg, mode='linear', align_corners=False)
            x = self.ecg_refine(x)
            ecg_out = x.transpose(1, 2)

        return low_out, ecg_out

class AutoencoderER(nn.Module):
    """
    Minimal AE wrapper that exposes:
        self.E : Embedder
        self.R : Recovery
    so checkpoints save keys 'E.*' and 'R.*'.

    Use for AE pretraining and warm-starting joint TimeGAN training.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        seq_length_low: int = 120,
        seq_length_ecg: int = 5280,
        latent_downsample: int = 4,
        use_ecg: bool = True,
        E: Embedder | None = None,
        R: Recovery | None = None,
    ):
        super().__init__()
        # Instantiate if not provided
        self.E = E if E is not None else Embedder(
            low_channels=2,
            ecg_channels=1,
            hidden_dim=hidden_dim,
            seq_length_low=seq_length_low,
            seq_length_ecg=seq_length_ecg,
            latent_downsample=latent_downsample,
            use_ecg=use_ecg,
        )
        self.R = R if R is not None else Recovery(
            hidden_dim=hidden_dim,
            seq_length_low=seq_length_low,
            seq_length_ecg=seq_length_ecg,
            latent_downsample=latent_downsample,
            use_ecg=use_ecg,
        )

        # Convenience metadata (handy to stash in checkpoints)
        self.hidden_dim = hidden_dim
        self.seq_length_low = seq_length_low
        self.seq_length_ecg = seq_length_ecg
        self.latent_downsample = latent_downsample
        self.use_ecg = use_ecg

    # ---- API ----
    def encode(self, x_low: torch.Tensor, x_ecg: torch.Tensor | None = None) -> torch.Tensor:
        """(B,T_low,2),(B,T_ecg,1)-> h: (B,H,T_latent). Keeps grads by design."""
        return self.E(x_low, x_ecg)

    def decode(self, h: torch.Tensor, out_lengths: tuple[int,int] | None = None):
        """h: (B,H,T_latent) -> (low_hat, ecg_hat)."""
        return self.R(h, out_lengths=out_lengths)

    def reconstruct(self, x_low: torch.Tensor, x_ecg: torch.Tensor | None = None):
        """Full AE pass: inputs -> h -> reconstructions."""
        h = self.encode(x_low, x_ecg)
        return self.decode(h)

    def forward(self, x_low: torch.Tensor, x_ecg: torch.Tensor | None = None):
        """Alias to reconstruct for nn.Module semantics."""
        return self.reconstruct(x_low, x_ecg)


class AntiAliasUp1D(nn.Module):
    def __init__(self, channels, scale: int, ksize: int = 5):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)
        self.smooth = nn.Conv1d(
            channels, channels, kernel_size=ksize,
            padding=ksize // 2, groups=channels, bias=False
        )
        with torch.no_grad():
            if ksize == 5:
                w = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
            elif ksize == 7:
                w = torch.tensor([1, 6, 15, 20, 15, 6, 1], dtype=torch.float32)
            else:
                w = torch.ones(ksize, dtype=torch.float32)
            w = (w / w.sum()).view(1, 1, -1)
            self.smooth.weight.copy_(w.repeat(channels, 1, 1))
        
        for p in self.smooth.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.smooth(self.up(x))
    



class Supervisor(nn.Module):
    """
    S: Predicts next-step latent features.
      Input : h_seq (B, T, H)      # latent from E, time-major
      Output: y     (B, T, H)      # aligned to input timesteps; use y[:, :-1] as h_{t+1} preds

    Train with teacher forcing:
        pred, tgt = S.predict_next(h_seq)
        L_sup = mse(pred, tgt)

    Notes:
      - Keep this active (non-zero weight) during GAN training.
      - Pair with a short rollout check to ensure stability (optional).
    """
    def __init__(self,
                 latent_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        h_seq: (B, T, H)
        returns y: (B, T, H) — per-timestep predictions (interpreted as h_{t+1})
        """
        y, _ = self.gru(h_seq)   # (B, T, hidden[*2])
        y = self.proj(y)         # (B, T, H)
        return self.ln(y)

    def predict_next(self, h_seq: torch.Tensor):
        """
        Convenience for teacher-forced loss:
          pred = y[:, :-1]  vs  target = h_seq[:, 1:]
        """
        y = self.forward(h_seq)
        return y[:, :-1, :], h_seq[:, 1:, :]

    @torch.no_grad()
    def rollout(self, h_start: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Optional: open-loop rollout for diagnostics.
        h_start: (B, 1, H) initial latent
        returns: (B, steps, H)
        """
        self.eval()
        outs = []
        h_t = h_start
        h_n = None
        for _ in range(steps):
            y, h_n = self.gru(h_t, h_n)     # (B, 1, hidden)
            h_pred = self.ln(self.proj(y))  # (B, 1, H)
            outs.append(h_pred)
            h_t = h_pred
        return torch.cat(outs, dim=1)
    
class SmoothUp1D(nn.Module):
    """Linear upsample + fixed depthwise smoothing (anti-alias)."""
    def __init__(self, channels, scale: int, ksize: int = 5):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)
        self.smooth = nn.Conv1d(
            channels, channels, kernel_size=ksize,
            padding=ksize // 2, groups=channels, bias=False
        )
        with torch.no_grad():
            if ksize == 5:
                w = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
            elif ksize == 7:
                w = torch.tensor([1, 6, 15, 20, 15, 6, 1], dtype=torch.float32)
            else:
                w = torch.ones(ksize, dtype=torch.float32)
            w = (w / w.sum()).view(1, 1, -1)
            self.smooth.weight.copy_(w.repeat(channels, 1, 1))
        for p in self.smooth.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.smooth(self.up(x))

class TwoStreamGenerator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        condition_dim: int,
        seq_length_low: int,
        seq_length_ecg: int,
        hidden_dim: int,
    ):
        super().__init__()
        # Shared projection (noise → hidden)
        self.noise_proj = nn.Linear(noise_dim, hidden_dim * (seq_length_low // 4))

         # ── Low‑rate branch (EDA+RESP) — anti‑alias upsampling ───
        self.low_up1 = AntiAliasUp1D(hidden_dim, scale=2, ksize=7)
        self.low_ch1 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1)
        self.tc_low1 = TimeConditionalBlock(hidden_dim // 2, condition_dim)

        self.low_up2 = AntiAliasUp1D(hidden_dim // 2, scale=2, ksize=5)
        self.low_ch2 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=1)
        self.tc_low2 = TimeConditionalBlock(hidden_dim // 4, condition_dim)

        self.low_head = nn.Sequential(
            nn.Conv1d(hidden_dim // 4, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 2, 3, padding=1),
        )

        # ── ECG branch ────────────────────────────────────────────
        self.ecg_proj = nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1)
        ch_ecg = hidden_dim // 4

        # Anti‑alias upsampling stack
        self.ecg_up = nn.ModuleList([
            SmoothUp1D(ch_ecg, scale=5, ksize=7),
            SmoothUp1D(ch_ecg, scale=5, ksize=5),
            SmoothUp1D(ch_ecg, scale=7, ksize=5),
        ])

        for blk in self.ecg_up:
            sm = getattr(blk, "smooth", None)
            if sm is not None:                # exists?
                for p in sm.parameters():     # weight and (maybe) bias
                    p.requires_grad_(False)

        # Refine head – NO activation here (A3 splits it out)
        self.ecg_refine = nn.Sequential(
            nn.Conv1d(ch_ecg, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
        )

        # Final fixed LPF then activation (A3)
        self.final_smooth = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            k = torch.tensor([1, 4, 6, 4, 1],
                             dtype=self.final_smooth.weight.dtype,
                             device=self.final_smooth.weight.device)
            k = (k / k.sum()).view(1, 1, -1)
            self.final_smooth.weight.copy_(k)
        for p in self.final_smooth.parameters():
            p.requires_grad_(False)  # keep the LPF fixed

        # FiLM conditioning for ECG path
        self.cond_ecg_film = nn.Sequential(
            nn.Linear(condition_dim, 128), nn.ReLU(),
            nn.Linear(128, ch_ecg * 2)
        )

        self.out_scale_low  = nn.Parameter(torch.tensor([1.4, 1.4], dtype=torch.float32))  # EDA, RESP
        self.out_scale_ecg  = nn.Parameter(torch.tensor(1.4, dtype=torch.float32)) 

        self.ecg_act = nn.Identity() 

        self.seq_length_low = seq_length_low
        self.seq_length_ecg = seq_length_ecg

    def forward(self, noise, condition_low):
        B = noise.size(0)
        # Shared backbone
        x = self.noise_proj(noise).view(B, -1, self.seq_length_low // 4)

        # ── Low‑rate branch ───────────────────────────────────────
        x_low = self.low_up1(x)
        x_low = F.gelu(self.low_ch1(x_low))
        x_low = x_low.transpose(1, 2)
        cond1 = F.interpolate(condition_low.transpose(1, 2), size=x_low.size(1)).transpose(1, 2)
        x_low = self.tc_low1(x_low, cond1).transpose(1, 2)

        x_low = self.low_up2(x_low)
        x_low = F.gelu(self.low_ch2(x_low))
        x_low = x_low.transpose(1, 2)
        cond2 = F.interpolate(condition_low.transpose(1, 2), size=x_low.size(1)).transpose(1, 2)
        x_low = self.tc_low2(x_low, cond2).transpose(1, 2)

        low_out = self.low_head(x_low).transpose(1, 2)                     # [B, 2, T]
        low_out = low_out * self.out_scale_low.view(1, 1, 2)               # [B, T, 2]

        # ── ECG branch ────────────────────────────────────────────
        x_ecg = self.ecg_proj(x)                                      # [B, C, T0]
        cond_vec = condition_low.mean(dim=1)                          # [B, K]
        gamma, beta = self.cond_ecg_film(cond_vec).chunk(2, dim=-1)   # [B, C], [B, C]
        x_ecg = x_ecg * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

        for up in self.ecg_up:
            x_ecg = up(x_ecg)
        if x_ecg.size(-1) != self.seq_length_ecg:
            x_ecg = F.interpolate(x_ecg, size=self.seq_length_ecg, mode='linear', align_corners=False)

        x_ecg = self.ecg_refine(x_ecg)    # linear pre‑activation
        x_ecg = self.final_smooth(x_ecg)  # gentle fixed LPF (A3)
        x_ecg = self.ecg_act(x_ecg)       # single activation point (A2)

        ecg_out = (x_ecg * self.out_scale_ecg).transpose(1, 2) 

        return low_out, ecg_out

class TimeConditionalBlock(nn.Module):
    """Time-conditional processing block with attention mechanism"""
    def __init__(self, signal_dim, condition_dim, hidden_dim=128):
        super().__init__()
        self.signal_dim = signal_dim
        self.condition_dim = condition_dim
        
        # Signal processing
        self.signal_proj = nn.Linear(signal_dim, hidden_dim)
        
        # Condition processing (emotional state labels)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Cross-attention between signal and condition
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, signal_dim)
        self.layer_norm = nn.LayerNorm(signal_dim)
        
    def forward(self, signal, condition):
        # signal: [batch, time, signal_dim]
        # condition: [batch, time, condition_dim] - your m1_seq
        
        # Project both inputs
        s_proj = self.signal_proj(signal)  # [batch, time, hidden]
        c_proj = self.condition_proj(condition)  # [batch, time, hidden]
        
        # Cross-attention: signal attends to condition
        attended, _ = self.attention(s_proj, c_proj, c_proj)
        
        # Residual connection + output
        output = self.output_proj(attended)
        return self.layer_norm(signal + output)

class TwoStreamDiscriminator(nn.Module):
    """
    Discriminator with projection-style conditioning:
        logit = f_uncond(h) + <h~, v(y)~>
    where h is the pooled feature, and y is the condition.
    """
    def __init__(self, condition_dim: int, hidden_dim: int = 256,
                 proj_type: str = "linear", l2_normalize: bool = True):
        super().__init__()
        from torch.nn.utils import spectral_norm as SN
        assert proj_type in ("linear", "embedding")
        self.proj_type = proj_type
        self.l2_normalize = l2_normalize
        self.condition_dim = condition_dim

        # ── Low-rate branch ──
        self.low_conv = nn.Sequential(
            SN(nn.Conv1d(2,   32, 4, 2, 1)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(32,  64, 4, 2, 1)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),
        )

        # ── ECG branch ──
        self.ecg_conv = nn.Sequential(
            SN(nn.Conv1d(1,   32, 4, 2, 1)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(32,  64, 4, 2, 1)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),
        )

        self.proj_scale = nn.Parameter(torch.tensor(1.0))

        # Shared trunk (with time-conditional block)
        feat_ch = 128 + 128  # after concat
        self.tc_block = TimeConditionalBlock(signal_dim=feat_ch, condition_dim=condition_dim)

        # Minibatch-std
        self.use_mbstd = True
        self.mb_group = 4

        # Head input width (add +1 channel if mbstd used)
        self.in_head = feat_ch + (1 if self.use_mbstd else 0)
        self.proj_width = feat_ch

        # Global pool
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Unconditional real/fake head (kept as your two-layer MLP)
        self.real_fake_head = nn.Sequential(
            SN(nn.Linear(self.in_head, hidden_dim)), nn.LeakyReLU(0.2),
            SN(nn.Linear(hidden_dim, 1)),
        )

        # ── Projection branch (replaces the auxiliary emotion head) ──
        if self.proj_type == "embedding":
            # For integer class IDs, or soft labels via F.linear with weight
            self.class_emb = nn.Embedding(condition_dim, self.proj_width)
        else:
            # For continuous / multi-hot conditions
            self.cond_proj = SN(nn.Linear(condition_dim, self.proj_width, bias=False))

    # --- feature extractor (unchanged) ---
    def extract_features(self, sig_low: torch.Tensor, sig_ecg: torch.Tensor, cond: torch.Tensor):
        """
        Returns pooled feature vector [B, C] from the shared trunk.
        """
        f_low = self.low_conv(sig_low.transpose(1, 2))  # [B, 128, TL']
        f_ecg = self.ecg_conv(sig_ecg.transpose(1, 2))  # [B, 128, TE']

        # Align in time
        T = min(f_low.size(2), f_ecg.size(2))
        f_low = F.interpolate(f_low, size=T, mode='linear', align_corners=False)
        f_ecg = F.interpolate(f_ecg, size=T, mode='linear', align_corners=False)

        # Concat and TC block
        feat = torch.cat([f_low, f_ecg], dim=1)        # [B, 256, T]
        feat_t = feat.transpose(1, 2)                   # [B, T, 256]
        cond_t = F.interpolate(cond.transpose(1, 2), size=T).transpose(1, 2)  # [B, T, K]
        feat_t = self.tc_block(feat_t, cond_t)          # [B, T, 256]
        feat = feat_t.transpose(1, 2)                   # [B, 256, T]

        if self.use_mbstd:
            feat = minibatch_std_1d(feat, group_size=self.mb_group)  # [B, 257, T] if enabled

        pooled = self.global_pool(feat).squeeze(-1)     # [B, in_head]
        pooled_proj = pooled[:, :self.proj_width]
        return pooled, pooled_proj

    def _pool_condition(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond can be [B, K] or [B, T, K]. We average over time if needed.
        Returns [B, K] (float) or [B] (long) for class IDs.
        """
        if cond.dim() == 3:          # [B, T, K]
            return cond.mean(dim=1)  # time-average -> [B, K]
        return cond                  # [B, K] (float) or [B] (long)

    def _embed_condition(self, cond_avg: torch.Tensor) -> torch.Tensor:
        """
        Map condition to [B, in_head] for the projection term.
        """
        if self.proj_type == "embedding":
            if cond_avg.dtype == torch.long and cond_avg.dim() == 1:
                # integer class IDs
                e = self.class_emb(cond_avg)                    # [B, in_head]
            else:
                # soft labels / multi-label: use embedding weights as a linear map
                e = F.linear(cond_avg, self.class_emb.weight)   # [B, in_head]
        else:
            e = self.cond_proj(cond_avg)                        # [B, in_head]
        return e

    def forward(self, sig_low: torch.Tensor, sig_ecg: torch.Tensor, cond: torch.Tensor):
        # 1) pooled features
        pooled, pooled_proj = self.extract_features(sig_low, sig_ecg, cond)      # [B, in_head]

        # 2) unconditional logit
        logit_uncond = self.real_fake_head(pooled)                  # [B, 1]

        # 3) projection term
        cond_avg = self._pool_condition(cond)                       # [B, K] or [B]
        cond_vec = self._embed_condition(cond_avg)                  # [B, in_head]

        if self.l2_normalize:
            h = F.normalize(pooled_proj, dim=1) 
            v = F.normalize(cond_vec, dim=1)
        else:
            h, v = pooled_proj, cond_vec

        proj = (F.normalize(pooled_proj, 1) * F.normalize(cond_vec, 1)).sum(1, keepdim=True)                 # [B, 1]

        # 4) final logit = unconditional + projection
        logit = logit_uncond + self.proj_scale * proj                                 # [B, 1]

        # Keep tuple shape for backward compatibility: (logit, aux, pooled)
        # 'aux' is None because the auxiliary head is removed.
        return logit, None, pooled

# Model initialization function
def create_tc_multigan(config):
    """Create two-stream TC-MultiGAN from either a dict or a dataclass-like config."""
    def cfg_get(cfg, key, *alts):
        if isinstance(cfg, dict):
            if key in cfg: 
                return cfg[key]
            for a in alts:
                if a in cfg:
                    return cfg[a]
        else:
            if hasattr(cfg, key):
                return getattr(cfg, key)
            for a in alts:
                if hasattr(cfg, a):
                    return getattr(cfg, a)
        raise KeyError(f"Missing '{key}' in config")

    nz   = cfg_get(config, "z_dim", "noise_dim")
    K    = cfg_get(config, "condition_dim")
    Tlow = cfg_get(config, "seq_length_low")
    Tecg = cfg_get(config, "seq_length_ecg")
    hdim = cfg_get(config, "hidden_dim")

    G = TwoStreamGenerator(
        noise_dim      = nz,
        condition_dim  = K,
        seq_length_low = Tlow,
        seq_length_ecg = Tecg,
        hidden_dim     = hdim,
    )
    D = TwoStreamDiscriminator(
        condition_dim = K,
        hidden_dim    = hdim,
    )
    return G, D

