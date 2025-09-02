from pathlib import Path
import argparse, json
from preprocess_wesad_core import (
    build_manifest, save_json,
    process_multiple_subjects, create_training_sequences_two_stream,
    zscore_train_and_apply, save_two_stream
)

def parse_args():
    ap = argparse.ArgumentParser("Two-stream WESAD preprocessing (LOSO fold)")
    ap.add_argument("--wesad_root",   required=True, help="path to data/raw/wesad")
    ap.add_argument("--manifest",     default="data/processed/wesad_manifest.json")
    ap.add_argument("--loso_splits",  default="data/processed/wesad_loso_splits.json")
    ap.add_argument("--fold_idx",     type=int, default=0)
    ap.add_argument("--out_root",     default="data/processed/two_stream")
    ap.add_argument("--window_s",     type=int, default=30)
    ap.add_argument("--step_s",       type=int, default=15)
    return ap.parse_args()


def ensure_loso_splits(manifest, splits_path, val_subjects=2, seed=42):
    import numpy as np
    splits_path = Path(splits_path)
    if splits_path.exists():
        return json.loads(splits_path.read_text())

    ids = sorted({m["subject"] for m in manifest}, key=lambda s: int(s[1:]))  # S2..S17
    rng = np.random.default_rng(seed)
    folds = []
    for test in ids:
        train = [i for i in ids if i != test]
        k = min(val_subjects, len(train))
        val_idx = rng.choice(len(train), size=k, replace=False)
        val = [train[i] for i in sorted(val_idx)]
        train_final = [train[i] for i in range(len(train)) if i not in set(val_idx)]
        folds.append({"train": train_final, "val": val, "test": [test]})

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(folds, indent=2))
    print(f"📝 LOSO splits created: {splits_path}")
    return folds

def main():
    import sys, time, traceback
    import numpy as np

    t0 = time.time()
    args = parse_args()

    # --- header ---
    print("\n" + "="*72, flush=True)
    print(f"▶️  Starting fold {args.fold_idx}", flush=True)
    print(f"cwd               : {Path.cwd().resolve()}", flush=True)
    print(f"--wesad_root      : {Path(args.wesad_root).resolve()}", flush=True)
    print(f"--manifest        : {Path(args.manifest).resolve()}", flush=True)
    print(f"--loso_splits     : {Path(args.loso_splits).resolve()}", flush=True)
    print(f"--out_root        : {Path(args.out_root).resolve()}", flush=True)
    print(f"--window_s / step : {args.window_s} / {args.step_s}", flush=True)
    print("="*72, flush=True)

    try:
        wesad_root = Path(args.wesad_root)

        # --- manifest ---
        t = time.time()
        man_path = Path(args.manifest)
        if not man_path.exists():
            print("🧭 Manifest missing → building…", flush=True)
            manifest = build_manifest(wesad_root)
            save_json(manifest, man_path)
            print(f"✅ Manifest written: {man_path} (subjects={len(manifest)})  [{time.time()-t:.2f}s]", flush=True)
        else:
            manifest = json.loads(man_path.read_text())
            print(f"🧭 Manifest loaded : {man_path} (subjects={len(manifest)})  [{time.time()-t:.2f}s]", flush=True)

        # --- splits ---
        t = time.time()
        splits_path = Path(args.loso_splits)
        if not splits_path.exists():
            raise FileNotFoundError(f"LOSO splits not found: {splits_path}")
        splits = ensure_loso_splits(manifest, splits_path)
        fold = splits[args.fold_idx]
        train_ids = fold["train"] + fold.get("val", [])
        test_ids  = fold["test"]
        print(f"🧩 Fold {args.fold_idx}: train={train_ids} | test={test_ids}  [{time.time()-t:.2f}s]", flush=True)

        # --- map IDs to paths ---
        id2path = {m["subject"]: m["path"] for m in manifest}
        missing = [sid for sid in (train_ids + test_ids) if sid not in id2path]
        if missing:
            raise RuntimeError(f"Subjects missing from manifest: {missing}")

        train_paths = [id2path[s] for s in train_ids]
        test_paths  = [id2path[s] for s in test_ids]
        print(f"📁 Train paths={len(train_paths)} | Test paths={len(test_paths)}", flush=True)

        # --- combine subjects (filter/decimate/mask) ---
        t = time.time()
        print("⚙️  process_multiple_subjects(...) …", flush=True)
        ds_fold = process_multiple_subjects(train_paths + test_paths, verbose=True)
        print(f"✅ combined: X_low={ds_fold['X_low'].shape}, X_ecg={ds_fold['X_ecg'].shape} "
              f"| fs_low={ds_fold['fs_low']} fs_ecg={ds_fold['fs_ecg']} "
              f"| segments={len(ds_fold['segments'])}  [{time.time()-t:.2f}s]", flush=True)

        # --- helper to slice & window a subject set ---
        def slice_ids(ids):
            seg_keep = [s for s in ds_fold["segments"] if s["subject_id"] in ids]
            if not seg_keep:
                raise RuntimeError(f"No segments matched IDs={ids}")

            # Concatenate data for these subjects
            X_low_chunks = [ds_fold["X_low"][s["low_start"]:s["low_end"]] for s in seg_keep]
            y_low_chunks = [ds_fold["y_low"][s["low_start"]:s["low_end"]] for s in seg_keep]
            X_ecg_chunks = [ds_fold["X_ecg"][s["ecg_start"]:s["ecg_end"]] for s in seg_keep]
            y_ecg_chunks = [ds_fold["y_ecg"][s["ecg_start"]:s["ecg_end"]] for s in seg_keep]

            X_low = np.concatenate(X_low_chunks)
            y_low = np.concatenate(y_low_chunks)
            X_ecg = np.concatenate(X_ecg_chunks)
            y_ecg = np.concatenate(y_ecg_chunks)

            # Rebuild segments with NEW indices relative to the concatenated arrays
            segments = []
            l_cur = e_cur = 0
            for s, xl, xe in zip(seg_keep, X_low_chunks, X_ecg_chunks):
                l_len = len(xl); e_len = len(xe)
                segments.append({
                    "subject_id": s["subject_id"],
                    "low_start": l_cur, "low_end": l_cur + l_len,
                    "ecg_start": e_cur, "ecg_end": e_cur + e_len,
                })
                l_cur += l_len; e_cur += e_len

            ds = dict(
                X_low=X_low, y_low=y_low, X_ecg=X_ecg, y_ecg=y_ecg,
                fs_low=ds_fold["fs_low"], fs_ecg=ds_fold["fs_ecg"],
                channels_low=ds_fold["channels_low"], channels_ecg=ds_fold["channels_ecg"],
                segments=segments,  # <-- rebuilt
            )
            return create_training_sequences_two_stream(ds, window_s=args.window_s, step_s=args.step_s)

        # --- window train/test ---
        t = time.time()
        print("🪟 Slicing & windowing TRAIN…", flush=True)
        tr_seq = slice_ids(train_ids)
        print(f"   train windows: low={tr_seq['X_low'].shape}, ecg={tr_seq['X_ecg'].shape}  [{time.time()-t:.2f}s]", flush=True)

        t = time.time()
        print("🪟 Slicing & windowing TEST…", flush=True)
        te_seq = slice_ids(test_ids)
        print(f"   test  windows: low={te_seq['X_low'].shape}, ecg={te_seq['X_ecg'].shape}  [{time.time()-t:.2f}s]", flush=True)
        
        def labels_to_m1_seq(y, T, classes=None):
            y = np.asarray(y).astype(int)
            uniq = np.unique(y) if classes is None else np.asarray(sorted(classes))
            lab2idx = {lab: i for i, lab in enumerate(uniq.tolist())}
            idx = np.vectorize(lab2idx.get)(y).astype(int)
            K = len(uniq)
            one_hot = np.eye(K, dtype=np.float32)[idx]            # (N,K)
            m1 = np.repeat(one_hot[:, None, :], T, axis=1)        # (N,T,K)
            return m1.astype(np.float32), K

    
        classes = sorted(set(tr_seq["cond"].tolist()) | set(te_seq["cond"].tolist()))
        m1_tr, K = labels_to_m1_seq(tr_seq["cond"], tr_seq["T_low"])
        m1_te, _ = labels_to_m1_seq(te_seq["cond"], te_seq["T_low"], classes=range(1, K+1))


        # --- z-score (train only stats) ---
        t = time.time()
        Xl_tr, Xl_te, mu_l, sd_l = zscore_train_and_apply(tr_seq["X_low"], te_seq["X_low"])
        Xe_tr, Xe_te, mu_e, sd_e = zscore_train_and_apply(tr_seq["X_ecg"], te_seq["X_ecg"])
        print(f"📏 Z-scored: low(train/test)={Xl_tr.shape}/{Xl_te.shape} | ecg(train/test)={Xe_tr.shape}/{Xe_te.shape} "
              f"[{time.time()-t:.2f}s]", flush=True)

        # --- save ---
        out_dir = Path(args.out_root) / f"fold_{test_ids[0]}"
        t = time.time()
        save_two_stream(
            out_dir,
            train=dict(X_low=Xl_tr, X_ecg=Xe_tr, cond=tr_seq["cond"], mean_low=mu_l, std_low=sd_l,
                       m1_seq=m1_tr,mean_ecg=mu_e, std_ecg=sd_e),
            test=dict(X_low=Xl_te, X_ecg=Xe_te, cond=te_seq["cond"], m1_seq=m1_te),
            meta=dict(
                fs_low=tr_seq["fs_low"], fs_ecg=tr_seq["fs_ecg"],
                T_low=tr_seq["T_low"], T_ecg=tr_seq["T_ecg"],
                channels_low=tr_seq["channels_low"], channels_ecg=tr_seq["channels_ecg"],
                train_subject_ids=train_ids, test_subject_ids=test_ids, 
                K=K, classes=classes  # (optional but handy for loaders)
            )
        )
        print(f"✅ Saved: {out_dir.resolve()}  [{time.time()-t:.2f}s]", flush=True)

        print(f"⏱️  Fold {args.fold_idx} done in {time.time()-t0:.2f}s", flush=True)

    except Exception as e:
        print("\n❌ ERROR during preprocessing", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()