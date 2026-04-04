import torch
import numpy as np

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_i = pred == cls
        target_i = target == cls

        intersection = (pred_i & target_i).sum().float()
        union = (pred_i | target_i).sum().float()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return torch.tensor(0.0)
    return torch.mean(torch.stack(ious))

def compute_per_class_iou(pred, target, num_classes=10):
    """Computes mean IoU and a list of per-class IoU for detailed evaluation."""
    pred = torch.argmax(pred, dim=1)
    ious = []
    cls_iou = []

    for cls in range(num_classes):
        pred_i = pred == cls
        target_i = target == cls

        intersection = (pred_i & target_i).sum().float()
        union = (pred_i | target_i).sum().float()

        if union > 0:
            val = intersection / union
            ious.append(val)
            cls_iou.append(val.item())
        else:
            cls_iou.append(float("nan"))
            
    mean_iou = torch.mean(torch.stack(ious)).item() if len(ious) > 0 else 0.0
    return mean_iou, cls_iou
