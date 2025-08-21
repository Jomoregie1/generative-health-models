from pathlib import Path
import numpy as np, json
from src.datasets.preprocess_wesad import preprocess_single_subject

# --- config paths ---
FOLD_DIR = Path("data/processed/tc_multigan_fold_S10")
RAW_DIR  = Path("data/raw/wesad/WESAD")  # folder that contains S2/, S3/, ...

manifest = json.load(open(FOLD_DIR / "dataset_config.json", "r"))
train_ids = manifest["train_subject_ids"]

# --- helpers ---
def pick_subject_pkl(root: Path, sid: str) -> Path | None:
    """Return the most likely chest .pkl for a subject (prefer chest over wrist)."""
    sub = root / sid
    # common names first
    candidates = [
        sub / f"{sid}.pkl",
        sub / "chest.pkl",
        sub / f"{sid}_chest.pkl",
    ]
    for q in candidates:
        if q.exists():
            return q
    # fallback: search recursively; prefer chest, avoid wrist
    pkls = list(sub.glob("**/*.pkl"))
    pkls = [p for p in pkls if "wrist" not in p.name.lower()]
    # if both chest & others exist, prefer chest-like names
    pkls.sort(key=lambda p: (("chest" not in p.name.lower()), p.name))
    return pkls[0] if pkls else None

# --- running sums for μ/σ ---
sum_low   = np.zeros(2, dtype=np.float64)
sumsq_low = np.zeros(2, dtype=np.float64)
count_low = 0

sum_ecg   = np.zeros(1, dtype=np.float64)
sumsq_ecg = np.zeros(1, dtype=np.float64)
count_ecg = 0

for sid in train_ids:
    p = pick_subject_pkl(RAW_DIR, sid)
    if p is None:
        print(f"❌ No .pkl found for {sid} under {RAW_DIR / sid}")
        continue

    try:
        res = preprocess_single_subject(
            subject_file=p,
            target_rate=manifest["fs_low"],  # 4
            original_rate=700,
            channels=("ECG","EDA","RESP"),
            transition_pad_s=5.0,
            min_valid_run_s=30.0,
            eda_hp_cutoff=0.03, eda_hp_order=2,
            eda_robust=False,   # ← ensure NO z-scaling inside
            verbose=False
        )
        if res is None:
            print(f"❌ Preprocess returned None for {sid}")
            continue
    except Exception as e:
        print(f"❌ Error processing {sid}: {e}")
        continue

    x_low = res["m2_low"].astype(np.float64)   # (N_low, 2)  [EDA, RESP]
    x_ecg = res["m2_ecg"].astype(np.float64)   # (N_ecg, 1)  [ECG]

    sum_low   += x_low.sum(axis=0)
    sumsq_low += (x_low**2).sum(axis=0)
    count_low += x_low.shape[0]

    sum_ecg   += x_ecg.sum(axis=0).ravel()
    sumsq_ecg += (x_ecg**2).sum(axis=0).ravel()
    count_ecg += x_ecg.shape[0]

# --- finalize μ/σ ---
if count_low == 0 or count_ecg == 0:
    raise RuntimeError("No samples accumulated; check RAW_DIR and subject files.")

mu_low  = sum_low / count_low
var_low = sumsq_low / count_low - mu_low**2
std_low = np.sqrt(np.maximum(var_low, 1e-12))

mu_ecg  = sum_ecg / count_ecg
var_ecg = sumsq_ecg / count_ecg - mu_ecg**2
std_ecg = np.sqrt(np.maximum(var_ecg, 1e-12))

# --- save ---
np.savez(FOLD_DIR / "norm_low.npz", mean=mu_low.astype(np.float32), std=std_low.astype(np.float32))
np.savez(FOLD_DIR / "norm_ecg.npz", mean=mu_ecg.astype(np.float32), std=std_ecg.astype(np.float32))

print("low  μ,σ:", mu_low, std_low)
print("ecg  μ,σ:", mu_ecg, std_ecg)