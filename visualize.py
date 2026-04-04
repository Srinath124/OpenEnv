import argparse
import cv2
import numpy as np
import os
from pathlib import Path

# Predefined color palette for off-road segmentation (10 classes)
PREDEFINED_PALETTE = np.array([
    [0, 0, 0],        # 0: Background - black
    [34, 139, 34],    # 1: Trees - forest green
    [0, 255, 0],      # 2: Lush Bushes - lime
    [210, 180, 140],  # 3: Dry Grass - tan
    [139, 90, 43],    # 4: Dry Bushes - brown
    [128, 128, 0],    # 5: Ground Clutter - olive
    [139, 69, 19],    # 6: Logs - saddle brown
    [128, 128, 128],  # 7: Rocks - gray
    [160, 82, 45],    # 8: Landscape - sienna
    [135, 206, 235],  # 9: Sky - sky blue
], dtype=np.uint8)

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

def colorize_with_palette(mask, palette=None):
    """Colorize a mask using a predefined palette."""
    if palette is None:
        palette = PREDEFINED_PALETTE
    
    h, w = mask.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_vals = np.unique(mask)
    for val in unique_vals:
        idx = min(int(val), len(palette) - 1)
        colored[mask == val] = palette[idx]
    
    return colored

def main():
    parser = argparse.ArgumentParser(
        description='Colorize grayscale segmentation masks using a class-aware palette',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py ./predictions/masks --output_folder ./predictions/masks_color
  python visualize.py ./train_stats --use_random_colors
        """
    )
    parser.add_argument('input_folder', type=str, help='Folder containing grayscale mask images')
    parser.add_argument('--output_folder', type=str, default=None, 
                        help='Output folder (default: input_folder/colorized)')
    parser.add_argument('--use_random_colors', action='store_true',
                        help='Use random colors instead of predefined palette')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder or os.path.join(input_folder, 'colorized')

    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f'Input folder not found: {input_folder}')

    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {input_folder}")
        return

    print(f"Found {len(image_files)} image files to process")
    if args.use_random_colors:
        print("Using random colors for each unique value")
    else:
        print(f"Using predefined palette with {len(PREDEFINED_PALETTE)} class colors")

    color_map = {}

    for image_file in sorted(image_files):
        print(f"Processing: {image_file.name}", end=" ... ")

        im = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
        if im is None:
            print("Skipped (could not read)")
            continue

        if args.use_random_colors:
            unique_values = np.unique(im)
            im2 = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)

            for value in unique_values:
                if value not in color_map:
                    color_map[value] = np.random.randint(0, 255, (3,), dtype=np.uint8)
                im2[im == value] = color_map[value]
        else:
            im2 = colorize_with_palette(im, PREDEFINED_PALETTE)

        output_path = os.path.join(output_folder, f"{image_file.stem}_color.png")
        # Convert RGB to BGR for OpenCV
        im2_bgr = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, im2_bgr)
        print("Saved")

    print(f"\nProcessing complete!")
    print(f"Colorized images saved to: {output_folder}")
    if args.use_random_colors:
        print(f"Total unique values found: {len(color_map)}")


if __name__ == '__main__':
    main()