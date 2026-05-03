"""
Overfit test for flow matching: train on 2 fixed examples for 500 steps,
then sample and check if the model can reproduce those exact examples.

If colors are correct and the T-block appears, the algorithm is working.

Usage (from src/):
  python overfit_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

OUT_DIR = "/tmp/overfit_test"
DEVICE = "cuda:0"
STEPS = 500
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load 2 real training samples ──────────────────────────────────────────────
print("[1] Loading 2 training samples...")
from wm_datasets import create_train_val_datasets
train_ds, _ = create_train_val_datasets(
    dataset_name="pusht",
    num_frames=4, frame_interval=5, image_size=(256, 256),
    data_path_train="/home/neil/data/dino_wm/pusht_noise/train",
    data_path_val="/home/neil/data/dino_wm/pusht_noise/val",
    normalize_action=True, normalize_pixel=True,
    validation_size=32,
    use_relative_actions=True, action_scale=100.0, with_velocity=True,
)
samples = [train_ds[0], train_ds[1]]
batch = {
    "video":  torch.stack([s["video"]  for s in samples]),   # [2, 4, 3, 256, 256]
    "action": torch.stack([s["action"] for s in samples]),   # [2, 4, 10]
}
print(f"  video shape: {batch['video'].shape}, range [{batch['video'].min():.2f}, {batch['video'].max():.2f}]")

# ── Build a fresh flow matching model (same config as training run) ────────────
print("[2] Building flow matching model...")
cfg = OmegaConf.load("/home/neil/data/nanowm_results/20260503_142952-NanoWM-S-2-F4S5-pusht/config.yaml")
# Force fp32 and no compile for simplicity
cfg.experiment.infra.compile = False
cfg.experiment.infra.mixed_precision = False

from experiments.train_experiment import NanoWMTrainingModule
pl = NanoWMTrainingModule(cfg)
pl = pl.to(DEVICE)
pl.train()

opt = torch.optim.Adam(pl.model.parameters(), lr=2e-4)

# ── Overfit loop ───────────────────────────────────────────────────────────────
print(f"[3] Overfitting for {STEPS} steps on 2 examples...")
video = batch["video"].to(DEVICE)
action = batch["action"].to(DEVICE)
B, F, C, H, W = video.shape

with torch.no_grad():
    x_flat = rearrange(video, "b f c h w -> (b f) c h w")
    z = pl._vae_encode(x_flat)
    z = rearrange(z, "(b f) c h w -> b f c h w", b=B)

model_kwargs = {"y": None, "action": action}

losses = []
for step in range(STEPS):
    t = torch.randint(0, pl.diffusion.num_timesteps, (B, F), device=DEVICE)
    loss_dict = pl.diffusion.training_losses(pl.model, z, t, model_kwargs)
    loss = loss_dict["loss"].mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if (step + 1) % 100 == 0:
        print(f"  step {step+1}/{STEPS}  loss={loss.item():.4f}")

# ── Sample from the overfitted model ──────────────────────────────────────────
print("[4] Sampling from overfitted model...")
pl.eval()
from diffusion.df_sample import dfot_sample

with torch.no_grad():
    z_sample = dfot_sample(
        diffusion=pl.diffusion,
        model=pl.model,
        shape=z.shape,
        context=z[:, :1],    # 1 context frame
        n_context_frames=1,
        scheduling_mode="full_sequence",
        num_sampling_steps=20,
        model_kwargs=model_kwargs,
        device=DEVICE,
        progress=True,
    )
    z_flat = rearrange(z_sample, "b f c h w -> (b f) c h w")
    x_pred_flat = pl._vae_decode(z_flat)
    x_pred = rearrange(x_pred_flat, "(b f) c h w -> b f c h w", b=B)

# ── Save comparison strips ─────────────────────────────────────────────────────
def save_strip(tensor, path, label=""):
    """tensor: [F, 3, H, W] float [0,1]"""
    frames = [(tensor[f].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8) for f in range(tensor.shape[0])]
    strip = np.concatenate(frames, axis=1)
    Image.fromarray(strip).save(path)
    print(f"  Saved {label}: {path}")

gt_vis   = ((video[0] + 1) / 2).clamp(0, 1)    # [F, 3, H, W]
pred_vis = ((x_pred[0] + 1) / 2).clamp(0, 1)

print()
print(f"  GT   per-channel mean: R={gt_vis[:,0].mean():.3f} G={gt_vis[:,1].mean():.3f} B={gt_vis[:,2].mean():.3f}")
print(f"  Pred per-channel mean: R={pred_vis[:,0].mean():.3f} G={pred_vis[:,1].mean():.3f} B={pred_vis[:,2].mean():.3f}")

save_strip(gt_vis,   f"{OUT_DIR}/overfit_gt.png",   "GT")
save_strip(pred_vis, f"{OUT_DIR}/overfit_pred.png", "Pred")

print()
print("=== RESULT ===")
print(f"  Final loss: {losses[-1]:.4f}  (started: {losses[0]:.4f})")
diff = (pred_vis - gt_vis).abs().mean()
print(f"  Mean abs pixel error: {diff:.4f}  (0=perfect, >0.1=bad)")
if diff < 0.05:
    print("  PASS: model can memorize examples — flow matching algorithm is correct")
elif diff < 0.15:
    print("  MARGINAL: partial memorization — may need more steps or LR tuning")
else:
    print("  FAIL: model cannot memorize even 2 examples — algorithm has a bug")
