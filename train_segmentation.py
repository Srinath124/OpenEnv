import os
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import N_CLASSES
from dataset import MaskDataset
from model import OffRoadSegNet
from losses import SegLoss
from metrics import compute_iou

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def train():
    parser = argparse.ArgumentParser(description="OffRoadSegNet Training")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--image_width", type=int, default=448)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = "cuda" if device.type == "cuda" else "cpu"

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Offroad_Segmentation_Training_Dataset", "train")
    val_dir = os.path.join(script_dir, "Offroad_Segmentation_Training_Dataset", "val")
    models_dir = os.path.join(script_dir, "models")
    stats_dir = os.path.join(script_dir, "train_stats")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    best_path = os.path.join(models_dir, "offroadnet_best.pth")
    H, W = args.image_height, args.image_width

    print(f"\n{'='*60}")
    print(f"  OffRoadSegNet  |  Device: {device}  |  AMP: {device.type == 'cuda'}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.learning_rate}")
    print(f"  Resolution: {H}x{W}  |  Loss: 0.5*Dice + 0.3*BCE + 0.2*IoU")
    print(f"  Aux weight: 0.3  |  Scheduler: CosineAnnealingLR")
    print(f"{'='*60}\n")

    # -------------------------------------------------
    # Data (joint augmentation in dataset.py)
    # -------------------------------------------------
    train_dataset = MaskDataset(data_dir, img_size=(H, W), augment=True)
    val_dataset = MaskDataset(val_dir, img_size=(H, W), augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Train: {len(train_dataset)} imgs ({len(train_loader)} batches)")
    print(f"Val  : {len(val_dataset)} imgs ({len(val_loader)} batches)\n")

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = OffRoadSegNet(N_CLASSES).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_par:,} ({n_par/1e6:.2f}M)\n")

    # -------------------------------------------------
    # Optimizer: AdamW + CosineAnnealingLR
    # -------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    criterion = SegLoss()
    scaler = torch.amp.GradScaler(amp_ctx, enabled=(device.type == "cuda"))

    best_iou = 0.0
    history = {"train_loss": [], "val_iou": [], "lr": []}

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs}",
                    unit="batch", dynamic_ncols=True)

        for img, mask in pbar:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_ctx, enabled=(device.type == "cuda")):
                main_pred, aux_pred = model(img)
                loss_main = criterion(main_pred, mask)
                loss_aux = criterion(aux_pred, mask)
                loss = loss_main + 0.3 * loss_aux   # aux reduced to 0.3

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        # Epoch-level scheduler step (CosineAnnealingLR)
        scheduler.step()

        # Validation
        model.eval()
        val_ious = []

        with torch.inference_mode():
            for img, mask in val_loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.amp.autocast(amp_ctx, enabled=(device.type == "cuda")):
                    pred = model(img)
                iou = compute_iou(pred, mask, N_CLASSES)
                val_ious.append(iou.item() if isinstance(iou, torch.Tensor) else iou)

        mean_iou = float(np.mean(val_ious))
        mean_loss = epoch_loss / len(train_loader)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(mean_loss)
        history["val_iou"].append(mean_iou)
        history["lr"].append(cur_lr)

        star = " *BEST*" if mean_iou > best_iou else ""
        print(f"  E{epoch+1:02d} | Loss: {mean_loss:.4f} | Val IoU: {mean_iou:.4f} | LR: {cur_lr:.2e}{star}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), best_path)

    # -------------------------------------------------
    # Save curves
    # -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], color="red", linewidth=1.5)
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_iou"], color="blue", linewidth=1.5)
    axes[1].set_title("Val IoU"); axes[1].set_xlabel("Epoch"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["lr"], color="green", linewidth=1.5)
    axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "training_curves.png"), dpi=150)
    plt.close()

    # Save metrics txt
    with open(os.path.join(stats_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Best Val IoU: {best_iou:.4f}\n")
        f.write(f"Total Epochs: {args.epochs}\n\n")
        f.write(f"{'Ep':>4} {'Loss':>10} {'ValIoU':>10} {'LR':>12}\n")
        f.write("-" * 40 + "\n")
        for i in range(len(history["train_loss"])):
            f.write(f"{i+1:4d} {history['train_loss'][i]:10.4f} "
                    f"{history['val_iou'][i]:10.4f} {history['lr'][i]:12.2e}\n")

    print(f"\n{'='*60}")
    print(f"  DONE  |  Best Val IoU: {best_iou:.4f}")
    print(f"  Model: {best_path}")
    print(f"  Plots: {stats_dir}/training_curves.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
