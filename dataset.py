import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from config import VALUE_MAP


def convert_mask(mask):
    arr = np.array(mask, dtype=np.int32)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for k, v in VALUE_MAP.items():
        new_arr[arr == k] = v
    return Image.fromarray(new_arr)


class MaskDataset(Dataset):
    """
    Joint image+mask dataset with boundary-preserving augmentations.
    
    Train augmentations (applied jointly to img AND mask):
        - Random Scale (0.75x - 1.25x)
        - Random Rotation (+/- 10 degrees)
        - Random Horizontal Flip
        - Color Jitter (image only)
        - Random Crop to target size
    
    Val: resize only, no augmentation.
    """

    def __init__(self, root, img_size=(256, 448), augment=False, return_name=False):
        self.img_dir = os.path.join(root, "Color_Images")
        self.mask_dir = os.path.join(root, "Segmentation")
        self.return_name = return_name
        self.has_masks = os.path.isdir(self.mask_dir)
        self.augment = augment
        self.h, self.w = img_size

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(
                f"Color_Images not found in {root}. Check for nested folders."
            )

        self.ids = sorted(os.listdir(self.img_dir))

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")

        if self.has_masks and os.path.exists(os.path.join(self.mask_dir, name)):
            mask = Image.open(os.path.join(self.mask_dir, name))
            mask = convert_mask(mask)
        else:
            mask = Image.new("L", img.size, 0)

        if self.augment:
            img, mask = self._augment(img, mask)
        else:
            img = TF.resize(img, (self.h, self.w), Image.BILINEAR)
            mask = TF.resize(mask, (self.h, self.w), Image.NEAREST)

        # To tensor + normalize
        img_t = TF.normalize(TF.to_tensor(img), self.mean, self.std)
        mask_t = torch.from_numpy(np.array(mask)).long()

        if self.return_name:
            return img_t, mask_t, name
        return img_t, mask_t

    def _augment(self, img, mask):
        # 1. Random scale (0.75 - 1.25)
        scale = random.uniform(0.75, 1.25)
        new_h = int(self.h * scale)
        new_w = int(self.w * scale)
        img = TF.resize(img, (new_h, new_w), Image.BILINEAR)
        mask = TF.resize(mask, (new_h, new_w), Image.NEAREST)

        # 2. Random rotation (+/- 10 degrees)
        angle = random.uniform(-10, 10)
        img = TF.rotate(img, angle, interpolation=Image.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST, fill=0)

        # 3. Random crop back to target size (or pad if scaled down)
        if new_h < self.h or new_w < self.w:
            # Pad if too small
            pad_h = max(self.h - new_h, 0)
            pad_w = max(self.w - new_w, 0)
            img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
            new_h = max(new_h, self.h)
            new_w = max(new_w, self.w)

        i = random.randint(0, new_h - self.h)
        j = random.randint(0, new_w - self.w)
        img = TF.crop(img, i, j, self.h, self.w)
        mask = TF.crop(mask, i, j, self.h, self.w)

        # 4. Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 5. Color jitter (image only - doesn't affect mask alignment)
        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))

        return img, mask
