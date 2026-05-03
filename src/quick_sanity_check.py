"""
Quick sanity check: load baseline checkpoint, run VAE encode-decode and diffusion
sampling on a single real batch, print color stats and save side-by-side frames.

Usage (from src/):
  python quick_sanity_check.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
import torchvision

# ── Config ────────────────────────────────────────────────────────────────────
CKPT = "/home/neil/data/nanowm_results/20260503_063651-NanoWM-S-2-F4S5-pusht/checkpoints/across_timesteps/epoch=0-step=30000.ckpt"
DATA_VAL = "/home/neil/data/dino_wm/pusht_noise/val"
OUT_DIR = "/tmp/sanity_check"
DEVICE = "cuda:0"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load a real batch from the dataset ────────────────────────────────────────
print("[1] Loading dataset...")
from wm_datasets import create_train_val_datasets
_, val_ds = create_train_val_datasets(
    dataset_name="pusht",
    num_frames=4,
    frame_interval=5,
    image_size=(256, 256),
    data_path_train="/home/neil/data/dino_wm/pusht_noise/train",
    data_path_val=DATA_VAL,
    normalize_action=True,
    normalize_pixel=True,
    validation_size=32,
    use_relative_actions=True,
    action_scale=100.0,
    with_velocity=True,
)
# Grab one batch of 2 samples
batch_videos = torch.stack([val_ds[i]["video"] for i in range(2)], dim=0)  # [2, F, C, H, W]
print(f"  batch shape: {batch_videos.shape}, range: [{batch_videos.min():.2f}, {batch_videos.max():.2f}]")

# ── Load VAE ──────────────────────────────────────────────────────────────────
print("[2] Loading VAE...")
from diffusers import AutoencoderKL
from utils.vae_ops import encode_first_stage, decode_first_stage
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
vae.eval()

# ── Test 1: VAE encode-decode (reconstruction) ────────────────────────────────
print("[3] VAE reconstruction test...")
x = batch_videos.to(DEVICE)           # [2, 4, 3, 256, 256], [-1, 1]
B, F, C, H, W = x.shape
x_flat = rearrange(x, "b f c h w -> (b f) c h w")
with torch.no_grad():
    z = encode_first_stage(vae, x_flat)           # [-1,1] → latent
    x_reconst_flat = decode_first_stage(vae, z)   # latent → [-1,1]
x_reconst = rearrange(x_reconst_flat, "(b f) c h w -> b f c h w", b=B)

print(f"  Input range:  [{x.min():.3f}, {x.max():.3f}]  mean={x.mean():.3f}")
print(f"  Reconst range:[{x_reconst.min():.3f}, {x_reconst.max():.3f}]  mean={x_reconst.mean():.3f}")
x_vis = ((x[0] + 1) / 2).clamp(0, 1)          # [F, 3, 256, 256]
r_vis = ((x_reconst[0] + 1) / 2).clamp(0, 1)  # [F, 3, 256, 256]
print(f"  Input [0,1] per-channel mean: R={x_vis[:,0].mean():.3f} G={x_vis[:,1].mean():.3f} B={x_vis[:,2].mean():.3f}")
print(f"  Reconst[0,1] per-channel mean: R={r_vis[:,0].mean():.3f} G={r_vis[:,1].mean():.3f} B={r_vis[:,2].mean():.3f}")

# Save frames as PNG strips
def save_strip(tensor, path):
    """Save [F, C, H, W] float [0,1] as horizontal PNG strip."""
    strip = torch.cat([tensor[f] for f in range(tensor.shape[0])], dim=2)  # [C, H, F*W]
    arr = (strip.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(arr).save(path)

save_strip(x_vis, f"{OUT_DIR}/input_frames.png")
save_strip(r_vis, f"{OUT_DIR}/reconst_frames.png")
print(f"  Saved: {OUT_DIR}/input_frames.png  (input)")
print(f"  Saved: {OUT_DIR}/reconst_frames.png (VAE reconst)")

# ── Test 2: Load model and run diffusion sampling ─────────────────────────────
print("[4] Loading model checkpoint...")
from experiments.train_experiment import NanoWMTrainingModule
from hydra import initialize_config_dir, compose
import hydra

# Load config from checkpoint dir
cfg_path = "/home/neil/data/nanowm_results/20260503_063651-NanoWM-S-2-F4S5-pusht/config.yaml"
cfg = OmegaConf.load(cfg_path)

pl_module = NanoWMTrainingModule(cfg)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"]
pl_module.load_state_dict(state_dict, strict=True)
pl_module = pl_module.to(DEVICE)
pl_module.eval()
print("  Model loaded successfully.")

# ── Test 3: Run log_images (same path as training eval) ───────────────────────
print("[5] Running sampling (same path as training callbacks)...")
batch_samples = [val_ds[i] for i in range(2)]
batch = {
    "video": torch.stack([s["video"] for s in batch_samples], dim=0),
    "video_name": [s["video_name"] for s in batch_samples],
    "action": torch.stack([s["action"] for s in batch_samples], dim=0),
}
print(f"  Action shape: {batch['action'].shape}, range: [{batch['action'].min():.2f}, {batch['action'].max():.2f}]")
with torch.no_grad():
    logs = pl_module.log_images(batch, split="val")

x_pred = logs["samples"]   # [2, 3, 4, 256, 256] in [-1,1]
x_gt   = logs["gt"]        # [2, 3, 4, 256, 256]

print(f"  Pred range: [{x_pred.min():.3f}, {x_pred.max():.3f}]  mean={x_pred.mean():.3f}")
print(f"  GT range:   [{x_gt.min():.3f}, {x_gt.max():.3f}]  mean={x_gt.mean():.3f}")

pred_vis = ((x_pred[0] + 1) / 2).clamp(0, 1)  # [C, F, H, W]
gt_vis   = ((x_gt[0] + 1) / 2).clamp(0, 1)

# Rearrange to [F, C, H, W] for saving
pred_vis = pred_vis.permute(1, 0, 2, 3)
gt_vis   = gt_vis.permute(1, 0, 2, 3)

print(f"  Pred per-channel mean: R={pred_vis[:,0].mean():.3f} G={pred_vis[:,1].mean():.3f} B={pred_vis[:,2].mean():.3f}")
print(f"  GT   per-channel mean: R={gt_vis[:,0].mean():.3f} G={gt_vis[:,1].mean():.3f} B={gt_vis[:,2].mean():.3f}")

save_strip(pred_vis, f"{OUT_DIR}/pred_frames.png")
save_strip(gt_vis,   f"{OUT_DIR}/gt_frames.png")
print(f"  Saved: {OUT_DIR}/pred_frames.png")
print(f"  Saved: {OUT_DIR}/gt_frames.png")

print()
print("=== SUMMARY ===")
print(f"  Input → Reconst mean difference: {(r_vis - x_vis).abs().mean():.4f}")
print(f"  GT    → Pred    mean difference: {(pred_vis - gt_vis).abs().mean():.4f}")
print(f"  View: eog {OUT_DIR}/*.png  or  xdg-open {OUT_DIR}/")
