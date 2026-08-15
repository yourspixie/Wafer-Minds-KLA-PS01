# Wafer-Minds — Joint Denoising & 2× Super-Resolution for Semiconductor Wafer Inspection

**i4C x KLA SemiCon AI Hackathon — Problem Statement 01: AI-Based Restoration of Degraded Images**

## Team

| Name | College |
|---|---|
| Saci Sandip | PES University |
| Reyna Reddy K | PES University |

---

## Problem

Wafer inspection sensors capture images under tight throughput constraints — shorter exposure and smaller apertures introduce speckle and Gaussian sensor noise, while low-resolution capture keeps inspection fast. Undetected or blurred defects at this stage propagate into yield loss further down the fabrication line: a missed micro-defect here can fail an entire die later.

Compound degradation (noise + downsampling together) is harder than either problem alone — denoising blurs fine lines, and upsampling alone amplifies noise. A model that jointly restores clarity and resolution directly improves defect-detection accuracy without slowing down the inspection pipeline.

**Target degradation**: Speckle noise + Additive Gaussian noise + 2× downsampling → `128×128` noisy input → `256×256` clean output

- **Input**: `.npy`, `128×128`, `float32`, unclipped range (values outside `[0,1]` are genuine sensor signal, not corrupted data)
- **Target**: `.npy`, `256×256`, `float32`, clean, bounded `[0.0, 1.0]`

---

## Approach

A **Hybrid Attention ResUNet**, purpose-built for compound degradation: a supervised CNN combining three proven restoration mechanisms into one **8.32M-parameter** network — chosen for the best accuracy-per-parameter tradeoff on a compound multi-degradation task, without the training cost of a diffusion or transformer-based restorer.

| Mechanism | Addresses | How |
|---|---|---|
| **Residual Channel Attention Blocks (RCAB)** | Speckle noise | Adaptively reweight feature channels, suppressing multiplicative speckle patterns while preserving line edges |
| **Multi-scale ResUNet encoder-decoder** (skip connections) | Gaussian noise | Expands the receptive field to separate additive sensor noise from true structure |
| **Sub-pixel PixelShuffle upsampling** | 2× downsampling | Reconstructs fine detail without the checkerboard artifacts transposed convolutions introduce |

**Pipeline**: `NoisyLR (128×128)` → Conv Stem → Encoder (RCAB ×2) → Bottleneck (RCAB ×4) → Decoder + Skip Fusion → PixelShuffle SR Head → `Restored (256×256)`, with a **global bicubic residual shortcut** — the network learns only the correction on top of a bicubic upsample, accelerating convergence.

### Training Strategy

- AdamW optimizer, cosine-annealed LR (`2e-4 → 1e-6`)
- 80 epochs, batch size 16, mixed-precision (AMP)
- Gradient clipping (max-norm 1.0) for stability
- Paired NoisyLR → GT loader, verified by filename
- 90/10 train/validation split

### Composite Loss Function

```
L_total = 1.0 · L_Charbonnier + 0.2 · L_SSIM + 0.1 · L_Sobel-Edge
```

- **1.0 × Charbonnier** — robust pixel fidelity, less sensitive to speckle outliers than plain L1/MSE
- **0.2 × SSIM** — structural similarity, matching the hackathon's own evaluation metric
- **0.1 × Sobel edge** — L1 gradient loss in X/Y that keeps wafer line edges sharp

Weights are tuned so fidelity dominates while structure and edges refine.

### Data Augmentation

- Random horizontal / vertical flips
- Random 90° rotations
- Raw values fed unclipped — signal beyond `[0,1]` preserved

---

## Innovation & Uniqueness

- **Global bicubic residual learning** — the network predicts only the residual correction on top of a bicubic upsample rather than the full image from scratch, a dramatically easier learning target that speeds convergence and stabilizes early training.
- **Physically-grounded normalization** — raw sensor values outside `[0,1]` are preserved as real signal, not clipped away; the model sees the true dynamic range the sensor captured, only enforcing `[0,1]` at the final output.
- **Baseline-relative validation discipline** — every checkpoint was benchmarked against a bicubic-only baseline on real held-out pairs before being trusted. This caught an early undertrained checkpoint that looked reasonable in isolation but was statistically indistinguishable from doing nothing, before it could reach submission.
- **Numerically-stabilized mixed precision** — SSIM's variance/division terms are unstable under fp16 autocast and can silently produce NaN mid-training. The loss computation is isolated in fp32 with gradient-norm clipping, eliminating a failure mode that otherwise corrupts an entire run.

---

## Results

Measured on held-out validation split:

| Metric | Bicubic Baseline | Ours |
|---|---|---|
| PSNR | 22.4 dB | **28.81 dB** |
| SSIM | 0.53 | **0.792** |
| LPIPS (lower = better) | — | **0.255** |

Visual comparisons of degraded input → restored output → ground truth across held-out samples are included in the submission deck and in `weights/visualizations/`.

---

## Technology & Feasibility

| | |
|---|---|
| **Framework** | PyTorch 2.0+ (AMP mixed precision, AdamW, cosine LR schedule) |
| **Hardware used** | NVIDIA T4 GPU (Google Colab) |
| **Training time** | ≈ 90 minutes for 80 epochs (≈ 74s/epoch, 3,200 paired samples) |
| **Model size** | 8.32M parameters (≈ 33 MB, fp32 checkpoint) |
| **Inference latency** | ≈ 889 ms/image measured on CPU; sub-10 ms expected on GPU given model size |
| **Deployability** | Single forward pass, no post-processing — drop-in for an existing inspection pipeline |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── model.py              # Hybrid Attention ResUNet definition
├── dataset.py             # Paired NoisyLR/GT dataset loader & augmentations
├── losses.py               # Charbonnier + SSIM + Sobel-edge composite loss
├── utils.py                 # PSNR / SSIM / LPIPS metrics + visualization helpers
├── train.py                  # Training script (reproducible from scratch)
├── eval.py                    # Standalone evaluation / inference script
├── weights/
│   ├── model.pt                # Trained model checkpoint (~33 MB)
│   ├── training_log.csv         # Per-epoch training metrics
│   └── visualizations/           # Epoch visual grids (Input | Pred | GT | Error Map)
└── outputs/
    └── restored_test/              # Model's restored output on the test set (.npy)
```

---

## Setup

Requires **Python 3.10+**. A CUDA-capable GPU is recommended (CPU inference works, ≈889 ms/image).

```bash
git clone https://github.com/yourspixie/Wafer-Minds-KLA-PS01.git
cd Wafer-Minds-KLA-PS01
pip install -r requirements.txt
```

---

## Running Inference (Evaluation Script)

`eval.py` loads the trained model, runs inference on every `.npy` file in an input directory, and writes restored `.npy` outputs to a specified output directory. It runs as-is with no manual edits.

```bash
python eval.py --input_dir /path/to/test/npy/files --output_dir /path/to/save/restored/outputs
```

| Flag | Required | Description |
|---|---|---|
| `--input_dir` | Yes | Directory of degraded test `.npy` files (`128×128`, `float32`) |
| `--output_dir` | Yes | Directory to write restored `.npy` files (`256×256`, `float32`, clamped to `[0,1]`) — created if it doesn't exist |
| `--weights` | No | Path to a model checkpoint. Defaults to `weights/model.pt` |

Each output file is written with the same filename as its corresponding input file.

---

## Training (Reproducing From Scratch)

```bash
python train.py --data_dir /path/to/train --epochs 80 --batch_size 16 --lr 2e-4
```

Training expects paired degraded/ground-truth `.npy` files under `--data_dir` (see `dataset.py` for the expected folder layout). Per-epoch loss, PSNR, and SSIM are logged to `weights/training_log.csv`, and visual grids (Input | Prediction | Ground Truth | Error Map) are saved to `weights/visualizations/`.

---

## References

**Research Papers**
- Zhang et al., "Image Super-Resolution Using Very Deep Residual Channel Attention Networks (RCAN)", ECCV 2018
- Shi et al., "Real-Time Single Image Super-Resolution Using an Efficient Sub-Pixel CNN (ESPCN)", CVPR 2016
- Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)", IEEE TIP 2004
- Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)", CVPR 2018

**Dataset & Tools**
- KLA Hackathon 2026 — official paired NoisyLR / GT wafer-inspection dataset
- PyTorch, torchvision — model implementation & training
- scikit-image, OpenCV — metric computation & preprocessing
- `lpips` (PyPI) — perceptual similarity metric
- Google Colab (T4 GPU) — training environment
