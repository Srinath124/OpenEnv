import numpy as np

# Reproducibility constraints are in train_segmentation.py
# Mask mapping
VALUE_MAP = {
    0:0, 100:1, 200:2, 300:3, 500:4,
    550:5, 700:6, 800:7, 7100:8, 10000:9
}

N_CLASSES = len(VALUE_MAP)

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky",
]

COLOR_PALETTE = np.array([
    [0,   0,   0  ],   # Background    - black
    [34,  139, 34 ],   # Trees         - forest green
    [0,   255, 0  ],   # Lush Bushes   - lime
    [210, 180, 140],   # Dry Grass     - tan
    [139, 90,  43 ],   # Dry Bushes    - brown
    [128, 128, 0  ],   # Ground Clutter- olive
    [139, 69,  19 ],   # Logs          - saddle brown
    [128, 128, 128],   # Rocks         - gray
    [160, 82,  45 ],   # Landscape     - sienna
    [135, 206, 235],   # Sky           - sky blue
], dtype=np.uint8)
