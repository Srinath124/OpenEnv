# Offroad Segmentation Scripts

Fast segmentation training and inference for the off-road dataset.

## Folder Layout

- `train_segmentation.py` - trains a Custom Efficient UNet model (`OffRoadSegNet`)
- `test_segmentation.py` - loads the trained model and writes predictions
- `app.py` - Streamlit interactive dashboard for testing multiple images and viewing real-time metrics
- `visualize.py` - colorizes grayscale segmentation masks
- `ENV_SETUP/` - Windows batch scripts for environment setup
- `config.py` - Configuration parameters
- `dataset.py` - Dataset loading and augmentations
- `losses.py` - Loss functions
- `metrics.py` - IoU Evaluation metrics
- `model.py` - Model definition (Custom Efficient UNet `OffRoadSegNet`)
- `Offroad_Segmentation_Training_Dataset/` - train and validation data
- `Offroad_Segmentation_testImages/` - test images and masks

## Custom Model Architecture: OffRoadSegNet

This project implements a custom, lightweight semantic segmentation architecture named **OffRoadSegNet**. It is carefully designed to balance real-time inference efficiency with high-fidelity segmentation accuracy for robust off-road navigation.

### Model Properties
- **Lightweight Encoder:** Utilizes Depthwise Separable Convolutions (`DSConv`) replacing standard convolutions. This drastically reduces the parameter count and computational overhead while maintaining representational capacity across downsampled stages (1/4, 1/8, 1/16).
- **Multi-Scale Context Module:** Incorporates an Atrous Spatial Pyramid Pooling (ASPP) bottleneck that captures diverse, multi-scale contextual features from varying receptive fields without degrading spatial resolution.
- **FPN-Style Decoder:** Employs a Feature Pyramid Network (FPN) progression. It smoothly upsamples and fuses high-level semantic features from the ASPP with high-resolution structural features mapped directly from the early encoder stages via skip connections.
- **Deep Supervision:** Employs an auxiliary segmentation head (`head_aux`) appended to an intermediate decoder layer. This enforces mid-level feature quality and delivers an extra gradient signal directly to deeper layers, acting as a strong regularizer and mitigating vanishing gradients.

### Training Strategy
- **Composite Objective Loss:** Trained comprehensively under a composed loss mapping of `0.5 * Dice Loss + 0.3 * BCE Loss + 0.2 * IoU Loss`. This specifically counters the impact of high class-imbalance typically found in off-road imagery and prioritizes accurate mask boundary reconstruction.
- **Auxiliary Training:** The accumulated training loss aggregates the primary and auxiliary head targets (`loss_main + 0.3 * loss_aux`) specifically during the training phase to accelerate early geometric alignment.
- **Optimized Scheduling:** Driven by the robust `AdamW` optimizer with a `CosineAnnealingLR` scheduler smoothly decaying the learning rate for precise local minimum convergence over the final epochs.
- **GPU Accelerated Mixed Precision:** Inherently utilizes PyTorch Automatic Mixed Precision (AMP) paired with runtime Gradient Scaling for vastly enhanced hardware utilization, memory footprint reduction, and reduced step bottlenecks on local GPUs.

## Setup

Run the Windows setup script (which will orchestrate environment creation and package installation):

```bash
ENV_SETUP\setup_env.bat
```

This will run `create_env.bat` first to create the Conda environment, then `install_packages.bat` to install dependencies.

## Train

```bash
python train_segmentation.py
```

Useful options:

```bash
python train_segmentation.py --epochs 10 --batch_size 4 --image_width 384 --image_height 216
```

The trained checkpoints are saved as `offroadnet_best.pth` and `offroadnet_final.pth` by default.

## Dashboard

You can explore your best saved models interactively via the built-in Streamlit dashboard. It features live multi-file image upload processing alongside visual logging from the recent offline training stats.

```bash
streamlit run app.py
```

## Test

```bash
python test_segmentation.py --model_path models\offroadnet_best.pth
```

## Visualize Masks

Colorize grayscale segmentation masks with the class-aware palette:

```bash
python visualize.py path\to\masks
```

Options:

```bash
python visualize.py path\to\masks --output_folder path\to\output --use_random_colors
```

This writes colorized masks to `colorized/` subfolder by default, or to your specified output folder.

## Notes

- Training and testing both use the same raw-mask class mapping.
- The current default training resolution is smaller than the original data to keep training faster.

## Outputs and Results

The most recent training run achieved the following results on the validation set:

- **Best Validation IoU**: 0.5249
- **Total Epochs**: 60

**Training Curve Summary (Last Epochs)**:
```text
  Ep       Loss     ValIoU           LR
----------------------------------------
...
  58     0.3853     0.5234     3.74e-06
  59     0.3850     0.5249     1.68e-06
  60     0.3858     0.5246     1.00e-06
```

Detailed metrics and generated training plots are collected in the `train_stats/` folder:
- `evaluation_metrics.txt`: Full epoch-by-epoch loss and mIoU tracking.
- `training_curves.png`: Loss and metrics progression plotted vs. epochs.
