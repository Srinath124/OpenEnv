import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, probs.shape[1]).permute(0, 3, 1, 2).float()
        smooth = 1e-5
        inter = (probs * targets_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice = (2.0 * inter + smooth) / (union + smooth)
        return 1.0 - dice.mean()


class IoULoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, probs.shape[1]).permute(0, 3, 1, 2).float()
        smooth = 1e-5
        inter = (probs * targets_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3)) - inter
        iou = (inter + smooth) / (union + smooth)
        return 1.0 - iou.mean()


class SegLoss(nn.Module):
    """
    Total Loss = 0.5 * Dice + 0.3 * BCE + 0.2 * IoU
    
    - Dice  -> improves overlap
    - BCE   -> stabilizes training (per-class independent gradients)
    - IoU   -> directly optimizes the evaluation metric
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, logits, targets):
        targets_oh = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()

        bce_loss = self.bce(logits, targets_oh)
        dice_loss = self.dice(logits, targets)
        iou_loss = self.iou(logits, targets)

        return 0.5 * dice_loss + 0.3 * bce_loss + 0.2 * iou_loss
