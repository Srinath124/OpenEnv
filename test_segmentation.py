import argparse
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import N_CLASSES, CLASS_NAMES, COLOR_PALETTE
from dataset import MaskDataset
from model import OffRoadSegNet
from metrics import compute_per_class_iou

def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        color[mask == c] = COLOR_PALETTE[c]
    return color

def test():
    parser = argparse.ArgumentParser(description="Offroad Segmentation Inference")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--model_path", type=str, default=os.path.join(script_dir, "models", "offroadnet_best.pth"), help="Path to weights")
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "Offroad_Segmentation_testImages"), help="Test images directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "predictions"), help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_width", type=int, default=448)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    H, W = args.image_height, args.image_width

    test_dataset = MaskDataset(args.data_dir, img_size=(H, W), augment=False, return_name=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = OffRoadSegNet(N_CLASSES).to(device)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
        
    model.eval()

    masks_dir = os.path.join(args.output_dir, "masks")
    masks_color_dir = os.path.join(args.output_dir, "masks_color")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)

    print(f"Running inference on {len(test_dataset)} images...")
    
    iou_scores = []
    all_cls_iou = []

    with torch.inference_mode():
        for imgs, masks, names in tqdm(test_loader, unit="batch"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            out = model(imgs)
            preds = torch.argmax(out, dim=1)

            if test_dataset.has_masks:
                mean_iou, cls_iou = compute_per_class_iou(out, masks, N_CLASSES)
                iou_scores.append(mean_iou)
                all_cls_iou.append(cls_iou)

            for i in range(imgs.shape[0]):
                name = names[i]
                stem = os.path.splitext(name)[0]
                pred_arr = preds[i].cpu().numpy().astype(np.uint8)

                # Save raw mask
                cv2.imwrite(os.path.join(masks_dir, f"{stem}_pred.png"), pred_arr)

                # Save color mask
                color_arr = mask_to_color(pred_arr)
                cv2.imwrite(os.path.join(masks_color_dir, f"{stem}_pred_color.png"), cv2.cvtColor(color_arr, cv2.COLOR_RGB2BGR))

    if test_dataset.has_masks:
        final_iou = float(np.nanmean(iou_scores))
        avg_cls_iou = np.nanmean(all_cls_iou, axis=0).tolist()

        print(f"\nMean test IoU: {final_iou:.4f}")
        for n, iou in zip(CLASS_NAMES, avg_cls_iou):
            print(f"{n}: {iou:.4f}")

if __name__ == "__main__":
    test()
