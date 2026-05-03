#!/usr/bin/env bash
# train_baseline.sh — smoke test then full 30k-step diffusion baseline on PushT
# Run from repo root after completing setup.sh and downloading the dataset.
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${REPO_DIR}/.venv/bin/python"
cd "${REPO_DIR}/src"

COMMON_ARGS=(
    experiment=dino_wm_pusht
    dataset=dino_wm/pusht
    model=nanowm_s2
    dataset_dir=/home/neil/data/dino_wm
    results_dir=/home/neil/data/nanowm_results
    wandb.enabled=false
    logger.name=tensorboard
)

echo "=== Smoke test (100 steps) ==="
"${VENV}" main.py \
    "${COMMON_ARGS[@]}" \
    experiment.training.max_steps=100 \
    experiment.infra.compile=false

echo ""
echo "=== Full baseline: NanoWM-S/2, v-prediction diffusion, 30k steps ==="
"${VENV}" main.py \
    "${COMMON_ARGS[@]}" \
    experiment.training.max_steps=30000

echo ""
echo "=== Baseline training complete. Check ~/data/nanowm_results/ for checkpoints. ==="
