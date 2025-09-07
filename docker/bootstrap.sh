#!/usr/bin/env bash
set -euo pipefail

ART_DIR="/app/results/checkpoints/diffusion"
FOLD_DIR="/app/data/processed/two_stream/fold_S10"

mkdir -p "$ART_DIR/milestones" "$FOLD_DIR"

if [ -n "${WEIGHTS_ZIP_URL:-}" ]; then
  echo "[bootstrap] Downloading: $WEIGHTS_ZIP_URL"
  TMP_ZIP="/tmp/inference_bundle.zip"
  curl -L "$WEIGHTS_ZIP_URL" -o "$TMP_ZIP"

  # Optional integrity check
  if [ -n "${BUNDLE_SHA256:-}" ]; then
    echo "${BUNDLE_SHA256}  ${TMP_ZIP}" | sha256sum -c -
  fi

  echo "[bootstrap] Unzipping..."
  # Robust unzip that normalizes Windows backslashes to POSIX slashes
  python3 - <<'PY' "/tmp/inference_bundle.zip" "/app"
import sys, zipfile, pathlib
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as zf:
    for m in zf.infolist():
        name = m.filename.replace('\\','/')  # normalize separators
        if name.endswith('/'):
            (pathlib.Path(dst) / name).mkdir(parents=True, exist_ok=True)
        else:
            p = pathlib.Path(dst) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as s, open(p, 'wb') as o:
                o.write(s.read())
PY
  rm -f "$TMP_ZIP"

  # Move up if a top-level "inference_bundle/" folder exists
  if [ -d /app/inference_bundle ]; then
    echo "[bootstrap] Moving unpacked bundle into place..."
    shopt -s dotglob
    mv /app/inference_bundle/* /app/
    rmdir /app/inference_bundle || true
    shopt -u dotglob
  fi
fi  # <-- close the outer if

echo "[bootstrap] Listing expected paths:"
ls -la "$ART_DIR" || true
ls -la "$ART_DIR/milestones" || true
ls -la "$FOLD_DIR" || true

echo "[bootstrap] Starting Streamlit..."
exec streamlit run src/app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0