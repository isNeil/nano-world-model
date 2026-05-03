#!/usr/bin/env bash
# setup.sh — one-shot setup for nano-world-model on a single RTX 4090
# Run from the repo root: bash setup.sh
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${HOME}/data"
RESULTS_DIR="${HOME}/data/nanowm_results"

echo "=== Step 1: Create conda environment ==="
conda env create -f "${REPO_DIR}/environment.yml" || echo "[INFO] env already exists, skipping"

echo ""
echo "=== Step 2: Download i3d model for FID/FVD metrics ==="
mkdir -p "${REPO_DIR}/pretrained_models/i3d"
I3D_PATH="${REPO_DIR}/pretrained_models/i3d/i3d_torchscript.pt"
if [ ! -f "${I3D_PATH}" ]; then
    curl -L "https://www.dropbox.com/scl/fi/c5nfs6c422nlpj880jbmh/i3d_torchscript.pt?rlkey=x5xcjsrz0818i4qxyoglp5bb8&dl=1" \
        -o "${I3D_PATH}"
    echo "[OK] i3d model downloaded to ${I3D_PATH}"
else
    echo "[OK] i3d model already present"
fi

echo ""
echo "=== Step 3: Create results directory ==="
mkdir -p "${RESULTS_DIR}"

echo ""
echo "=== Step 4: Dataset ==="
echo ""
echo "  Download DINO-WM PushT data from OSF:"
echo "  https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28"
echo ""
echo "  Download the 'pusht_noise' folder and unzip so the layout is:"
echo "  ${DATA_DIR}/dino_wm/"
echo "  └── pusht_noise/"
echo "      ├── train/"
echo "      └── val/"
echo ""
echo "  Then come back and run: bash train_baseline.sh"
echo ""
echo "=== Setup complete (dataset download is manual — see above) ==="
