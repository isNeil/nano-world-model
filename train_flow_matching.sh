#!/usr/bin/env bash
# train_flow_matching.sh — 30k-step flow matching run on PushT (identical config to baseline)
# Run from repo root after the baseline training is complete.
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${REPO_DIR}/.venv/bin/python"
cd "${REPO_DIR}/src"

COMMON_ARGS=(
    experiment=dino_wm_pusht_flow
    dataset=dino_wm/pusht
    model=nanowm_s2
    dataset_dir=/home/neil/data/dino_wm
    results_dir=/home/neil/data/nanowm_results
    wandb.enabled=false
    logger.name=tensorboard
)

echo "=== Smoke test: flow matching (100 steps) ==="
"${VENV}" main.py \
    "${COMMON_ARGS[@]}" \
    experiment.training.max_steps=100 \
    experiment.infra.compile=false

echo ""
echo "=== Full flow matching run: NanoWM-S/2, 30k steps ==="
"${VENV}" main.py \
    "${COMMON_ARGS[@]}" \
    experiment.training.max_steps=30000

echo ""
echo "=== Flow matching training complete. Check ~/data/nanowm_results/ for checkpoints. ==="
