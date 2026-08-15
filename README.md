# Wafer-Minds — Joint Denoising & 2x Super-Resolution for Semiconductor Wafer Inspection

**i4C x KLA SemiCon AI Hackathon — Problem Statement 01: AI-Based Restoration of Degraded Images**

## Team

| Name |  Contact |

| REYNA |   kreynareddy@gmail.com |
| SACI | sacimense@gmail.com |

College: PESU(RR)

---

## Problem

Semiconductor inspection sensors produce images degraded by a compound pipeline of **speckle noise**, **additive Gaussian noise**, and **downsampling**, applied in unknown order. This repository restores those degraded low-resolution scans back into clean, high-resolution images that KLA's downstream defect-detection systems can rely on — recovering fine wafer line patterns that noise and resolution loss would otherwise hide.

- **Input**: `.npy`, `128×128`, `float32`, unclipped range `[-0.08, 1.71]` (values outside `[0,1]` are genuine sensor signal, not corrupted data)
- **Target**: `.npy`, `256×256`, `float32`, clean, bounded `[0.0, 1.0]`
- **Task**: Joint denoising + 2x super-resolution (`128×128` → `256×256`)

---

## Approach

We use a **Joint Denoise-SR ResUNet** (~8.32M parameters) that performs denoising and 2x upsampling in a single forward pass rather than as two separate stages, so restoration and resolution enhancement are optimized jointly instead of compounding each other's errors.

**Why this design:**

- **Raw normalization**: Input values outside `[0,1]` encode real sensor information, so we never clip on input. The model consumes the raw float range and only enforces `[0,1]` via `torch.clamp` on the final output, matching the ground-truth distribution.
- **Multi-scale ResUNet encoder-decoder**: 3-stage downsampling gives the network enough receptive field to reason about noise globally, while skip connections preserve fine spatial detail for the decoder.
- **Residual Channel Attention Blocks (RCAB)**: adaptively reweight feature channels to suppress speckle/Gaussian noise while preserving fine wafer line edges.
- **Sub-pixel convolution (PixelShuffle 2x)** for upsampling, avoiding the checkerboard artifacts common with transposed convolutions.
- **Global bicubic shortcut**: the network predicts a residual on top of a bicubic-upsampled base rather than the full image from scratch, which speeds up convergence significantly.

**Loss function:**

```
L_total = 1.0 · L_Charbonnier + 0.2 · L_SSIM + 0.1 · L_Sobel-Edge
```

- **Charbonnier loss** — a robust smooth-L1 variant, less sensitive to speckle outliers than plain MSE/L1.
- **SSIM loss** — directly optimizes structural similarity, matching the hackathon's own evaluation metric.
- **Sobel edge loss** — an L1 gradient loss in X/Y that keeps wafer line edges sharp instead of smoothed over.

---

## Results

| Metric | Bicubic Baseline | Ours |
|---|---|---|
| PSNR | 22.4 db | 28.81 db |
| SSIM | 0.53| 0.792 |
| LPIPS | low| 0.2555 |

_Add before/after sample comparisons (degraded input → restored output → ground truth) here — e.g. link or embed images from `weights/visualizations/`._

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── model.py              # Joint Denoise-SR ResUNet definition
├── dataset.py             # Paired dataset loader & augmentations
├── losses.py               # Charbonnier + SSIM + Sobel-edge composite loss
├── utils.py                 # PSNR / SSIM / LPIPS metrics + visualization helpers
├── train.py                  # Training script (reproducible from scratch)
├── eval.py                    # Standalone evaluation / inference script
├── weights/
│   ├── model.pt                # Trained model checkpoint
│   └── training_log.csv         # Per-epoch training metrics
└── outputs/
    └── restored_test/              # Model's restored output on the test set (.npy)
```

> **Note on large files**: if `weights/model.pt` exceeds GitHub's upload limits, it is hosted externally — see [Model Weights](#model-weights) below.

---

## Setup

Requires **Python 3.10+** and a CUDA-capable GPU (CPU inference also works, just slower).

```bash
git clone https://github.com/yourspixie/Wafer-Minds-KLA-PS01.git
cd Wafer-Minds-KLA-PS01
pip install -r requirements.txt
```

---

## Running Inference (Evaluation Script)

`eval.py` loads the trained model, runs inference on every `.npy` file in an input directory, and writes restored `.npy` outputs to a specified output directory. It runs as-is with no manual edits — only the two required arguments below.

```bash
python eval.py --input_dir /path/to/test/npy/files --output_dir /path/to/save/restored/outputs
```

**Arguments:**

| Flag | Required | Description |
|---|---|---|
| `--input_dir` | Yes | Directory of degraded test `.npy` files (`128×128`, `float32`) |
| `--output_dir` | Yes | Directory to write restored `.npy` files (`256×256`, `float32`, clamped to `[0,1]`) — created if it doesn't exist |
| `--weights` | No | Path to a model checkpoint. Defaults to `weights/model.pt` |

Each output file is written with the same filename as its corresponding input file.

---

## Training (Reproducing From Scratch)

```bash
python train.py --data_dir /path/to/train --epochs 15 --batch_size 16 --lr 2e-4
```

Training expects paired degraded/ground-truth `.npy` files under `--data_dir` (see `dataset.py` for the expected folder layout). Per-epoch loss, PSNR, and SSIM are logged to `weights/training_log.csv`, and visual grids (Input | Prediction | Ground Truth | Error Map) are saved to `weights/visualizations/` for inspection.

---

## Model Weights

- File: `weights/model.pt`
- Format: PyTorch `state_dict` (`.pt`), loadable directly by `eval.py`
- Size: _TBD — add actual size_
---

## Technology & Feasibility

- **Framework**: PyTorch
- **Hardware used for training**: _GPU type, cloud platform_
- **Training time**: _TBD_
- **Model size**: ~8.32M parameters
- **Inference time**: ~2–5 ms/image on H100/A100-class GPUs

---

## Requirements

Full pinned dependency list (from `pip freeze` in the training environment) is in [`requirements.txt`](./requirements.txt).

---


