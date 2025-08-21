import argparse
import json
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in ('y', 'yes', 'true', 't', '1'):
        return True
    if s in ('n', 'no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean, got "{v}"')

def add_bool_arg(parser, name, default=False, help_text=''):
    """
    Adds a robust boolean flag:
      --flag              -> True
      --flag true/false   -> True/False
      --no-flag           -> False
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f'--{name}', nargs='?', const=True, default=default, type=str2bool,
                       help=f'{help_text} (default: {default})')
    group.add_argument(f'--no-{name}', dest=name, action='store_false',
                       help=argparse.SUPPRESS)

def _build_parser(defaults=None):
    p = argparse.ArgumentParser(
        description="Training configuration for tc-multigan",
        allow_abbrev=False
    )

    # ---------------- Core / paths ----------------
    p.add_argument('--config_json', type=str, default=None,
                   help='Path to a JSON config whose values become defaults (CLI still overrides).')
    p.add_argument('--ckpt_dir', type=str, default='results/checkpoints',
                   help='Directory to save checkpoints')
    p.add_argument('--sample_dir', type=str, default='results/samples',
                   help='Directory to save generated samples')
    p.add_argument('--log_dir', type=str, default='results/logs',
                   help='Directory to save logs')
    p.add_argument('--data_root', type=str, default=str(Path('data/processed').resolve()),
                   help='Root folder of processed data')
    p.add_argument('--fold', type=str, default='tc_multigan_fold_S10',
                   help='Fold / split identifier')
    p.add_argument('--resume', type=str, default='',
                   help='Path to checkpoint to resume from')

    # ---------------- Device / misc ----------------
    # Default device chosen at runtime in train script; we accept either here
    p.add_argument('--device', type=str, default='cuda',
                   choices=['cpu', 'cuda'], help='Device to use')

    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # ---------------- Model / data dims ------------
    p.add_argument('--z_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--n_channels', type=int, default=3)
    p.add_argument('--condition_dim', type=int, default=4)

    p.add_argument('--fs_low', type=int, default=4)
    p.add_argument('--fs_ecg', type=int, default=175)

    p.add_argument('--seq_length', type=int, default=240, help='(If used) full-length setting')
    p.add_argument('--seq_length_low', type=int, default=120)
    p.add_argument('--seq_length_ecg', type=int, default=5250)

    # ---------------- Data / splits ----------------
    add_bool_arg(p, 'weighted_sampling', default=False, help_text='Use weighted sampling of classes')
    p.add_argument('--train_split', type=str, default='train', choices=['train', 'val', 'test'])
    p.add_argument('--val_split', type=str, default='test', choices=['train', 'val', 'test'])

    # ---------------- Optim / schedule -------------
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr_g', type=float, default=1e-4)
    p.add_argument('--lr_d', type=float, default=2e-4)
    p.add_argument('--epochs_ae', type=int, default=20)
    p.add_argument('--epochs_gan', type=int, default=50)
    p.add_argument('--ckpt_interval', type=int, default=5)
    p.add_argument('--sample_interval', type=int, default=1)
    p.add_argument('--g_steps', type=int, default=2)
    p.add_argument('--d_steps', type=int, default=1)

    # ---------------- Loss weights -----------------
    p.add_argument('--lambda_adv', type=float, default=1.0)
    p.add_argument('--lambda_rec', type=float, default=100.0)
    p.add_argument('--lambda_tc', type=float, default=2.0)
    p.add_argument('--lambda_mismatch', type=float, default=0.5)
    p.add_argument('--lambda_fm', type=float, default=10.0)
    p.add_argument('--lambda_tv_low', type=float, default=0.0)
    p.add_argument('--lambda_tv_ecg', type=float, default=0.0)

    # --- Spectral loss knobs ---
    p.add_argument("--lambda_spec_low", type=float, default=0.5,
                        help="Weight for spectral L1 on low-rate channels (EDA/RESP)")
    p.add_argument("--lambda_spec_ecg", type=float, default=1.0,
                        help="Weight for spectral L1 on ECG")
    p.add_argument("--spec_nfft_low", type=int, default=256,
                        help="Target FFT length for low-rate spectral loss (>= T will zero-pad)")
    p.add_argument("--spec_nfft_ecg", type=int, default=4096,
                        help="Target FFT length for ECG spectral loss (>= T will zero-pad)")
    

    
    # optional band limits (Hz); leave None to use full band
    p.add_argument("--spec_ecg_fmin", type=float, default=5.0)
    p.add_argument("--spec_ecg_fmax", type=float, default=40.0)
    p.add_argument("--spec_low_fmin", type=float, default=None)  # typically None
    p.add_argument("--spec_low_fmax", type=float, default=None)
    p.add_argument('--spec_warmup_epochs', type=int, default=12)
    

    # Feature-matching warmup (you referenced this in train_one_epoch)
    p.add_argument('--fm_warmup_epochs', type=int, default=15,
                   help='Ramp lambda_fm from 0 to its value over these epochs (0 = off)')


    # ---------------- Labels (if used) --------------
    p.add_argument('--real_label', type=float, default=0.8)
    p.add_argument('--fake_label', type=float, default=0.2)
    p.add_argument('--gen_label', type=float, default=0.8)

    # ---------------- Aug / noise -------------------
    p.add_argument('--aug_jitter', type=float, default=0.01)
    p.add_argument('--aug_scale', type=float, default=0.1)
    p.add_argument('--inst_noise_std', type=float, default=0.01)
    p.add_argument('--inst_noise_warm_epochs', type=int, default=20)

    # ---------------- Regularizers / options --------
    add_bool_arg(p, 'use_r1', default=False, help_text='Enable R1 gradient penalty in D')
    p.add_argument('--r1_gamma', type=float, default=1.0)

    add_bool_arg(p, 'use_ema', default=False, help_text='Enable EMA for G weights')
    p.add_argument('--ema_decay', type=float, default=0.999)

    # ---------------- Sampling ----------------------
    p.add_argument('--sample_n', type=int, default=8)

    p.add_argument('--lambda_spike', type=float, default=0.4)
    p.add_argument('--spike_tau', type=float, default=2.0)

    # --------------- Boundary penalty for low ---------------
    p.add_argument('--lambda_boundary_low', type=float, default=3e-3)
    p.add_argument('--boundary_margin_low', type=float, default=0.92)

    # --------------- Channel-specific spectral bands for low -----------
    p.add_argument("--spec_eda_fmin", type=float, default=0.03)
    p.add_argument("--spec_eda_fmax", type=float, default=0.25)
    p.add_argument("--spec_resp_fmin", type=float, default=0.10)
    p.add_argument("--spec_resp_fmax", type=float, default=0.50)

    p.add_argument('--lambda_tv_eda',  type=float, default=None,
               help='If set, TV weight for EDA channel overrides lambda_tv_low')
    p.add_argument('--lambda_tv_resp', type=float, default=None,
               help='If set, TV weight for RESP channel overrides lambda_tv_low')
    
    p.add_argument('--lambda_boundary_low_eda',  type=float, default=None,
               help='Override boundary penalty for EDA; default to lambda_boundary_low')
    p.add_argument('--lambda_boundary_low_resp', type=float, default=None,
               help='Override boundary penalty for RESP; default to lambda_boundary_low')
    
    p.add_argument('--spec_resp_shape_only', type=str2bool, default=True,
               help='Normalize RESP spectra before comparison (shape-only spectral loss)')
    
    p.add_argument('--lambda_mm', type=float, default=0.0,
               help='Weight for batch moment matching (mean/std) on low + ECG')
    
    p.add_argument('--ecg_boundary_margin', type=float, default=0.95)
    p.add_argument('--lambda_boundary_ecg', type=float, default=0.08)
    p.add_argument('--ecg_margin', type=float, default=0.98)  


    # Allow config_json defaults to be injected
    if defaults:
        # Only apply keys that exist as arguments
        valid = {a.dest for a in p._actions}
        to_apply = {k: v for k, v in defaults.items() if k in valid}
        p.set_defaults(**to_apply)

    return p

def _load_json_defaults(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'config_json not found: {path}')
    with path.open('r') as f:
        data = json.load(f)
    return data

def _pretty_print(args):
    print("Configuration:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

def _validate(args):
    # basic sanity checks
    assert args.batch_size > 0
    assert args.lr_g > 0 and args.lr_d > 0
    assert args.epochs_gan >= 1
    assert args.seq_length_low > 0 and args.seq_length_ecg > 0
    assert 0.0 < args.ema_decay < 1.0 or (args.use_ema is False)
    assert args.ckpt_interval >= 1 and args.sample_interval >= 1

def parse_args():
    # Stage 1: peek at --config_json, so we can use its values as defaults
    peek = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    peek.add_argument('--config_json', type=str, default=None)
    known, _ = peek.parse_known_args()

    defaults = None
    if known.config_json:
        defaults = _load_json_defaults(known.config_json)

    # Stage 2: full parse with JSON defaults (if given), then CLI overrides
    parser = _build_parser(defaults)
    args = parser.parse_args()

    # Make paths absolute early (optional)
    args.ckpt_dir = str(Path(args.ckpt_dir))
    args.sample_dir = str(Path(args.sample_dir))
    args.log_dir = str(Path(args.log_dir))
    args.data_root = str(Path(args.data_root))
    args.resume = str(Path(args.resume)) if args.resume else ''

    _validate(args)
    _pretty_print(args)
    return args