# GD-CSRNet

## Overview

**GD-CSRNet** is a PyTorch-based multi-contrast MRI super-resolution training framework. The core model is the **Gated Delta Cross-modal Structure Refinement Network (GD-CSRNet)**, designed for super-resolution reconstruction and quality enhancement of multi-modal MRI images.

---

## Repository Structure

| File / Directory | Description |
|---|---|
| `train.py` | Main entry point: loads configs, prepares data, builds model, runs train/val/test |
| `engine.py` | Training loop, validation & test logic; supports distributed training, model selection, and checkpoint management |
| `utils.py` | Utility functions: logging, optimizer, scheduler, and random seed setup |
| `configs/config_setting.py` | Default training config, model hyperparameters, data paths, and preprocessing transforms |
| `datasets/dataset.py` | `BraTs_datasets` class supporting BraTs2020 multi-contrast MRI loading and clinical degradation augmentation |
| `model/gated_delta_sr.py` | Main GD-CSRNet model definition |
| `model/DGCFBlock.py` | Key network module implementation |
| `model/SDDFBlock.py` | Key network module implementation |

---

## Requirements

Python 3.8+ is recommended. Install the main dependencies via:

```bash
pip install torch torchvision tensorboardX monai scipy SimpleITK medpy pillow opencv-python
```

> For distributed training, ensure your PyTorch installation supports `torch.distributed`.

---

## Dataset Format

The default configuration uses the `BraTs2020_t1_t2` dataset. The expected directory structure is:
```
datasets/BraTs2020_t1_t2/
├── train/
│   ├── oriT2/          # High-resolution target images
│   ├── oriT1/          # Reference images
│   ├── orLRbicT1/x4/   # Low-resolution reference images
│   └── orLRbicT2/x4/   # Low-resolution target images
└── val/
├── oriT2/
├── oriT1/
├── orLRbicT1/x4/
└── orLRbicT2/x4/
```
The data loader in `datasets/dataset.py` expects four input types:

- **`oriT2`** — High-resolution target image
- **`oriT1`** — Reference image
- **`orLRbicT1/x4`** — Low-resolution version of the reference image
- **`orLRbicT2/x4`** — Low-resolution version of the target image

---

## Getting Started

Run training with default settings:

```bash
python train.py
```

All configuration is read from `configs/config_setting.py`. To change the network architecture, dataset, or training hyperparameters, edit that file directly.

---

## Default Configuration

| Parameter | Default Value |
|---|---|
| `network` | `'gdcsr'` |
| `datasets` | `'BraTs2020_t1_t2'` |
| `batch_size` | `4` |
| `epochs` | `150` |
| `num_gpus` | `2` |
| `enable_logging` | `True` |
| `work_dir` | `results/{network}_{datasets}_x{upscale}/` |

---

## Model Architecture

The framework supports multiple network architectures selectable via `config.network` in `train.py`. The default is `gdcsr`.

Model files included in this repository:

- `model/gated_delta_sr.py` — Full GD-CSRNet definition
- `model/DGCFBlock.py` — Gated Delta Cross-modal Fusion Block
- `model/SDDFBlock.py` — Structure-guided Dual-domain Fusion Block

---

## Training Outputs

All outputs are saved under `work_dir/`:
```
work_dir/
├── log/           # Training log files
├── checkpoints/   # Saved model weights
│   ├── latest.pth # Most recent checkpoint
│   └── best.pth   # Best checkpoint (by validation PSNR)
├── summary/       # TensorBoard logs
└── outputs/       # Optional output results
```
---

## License

This project is released under the [MIT License](LICENSE).
