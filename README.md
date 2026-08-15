# Wafer-Minds - Joint Denoising & 2x Super-Resolution for Semiconductor Wafer Inspection

i4C x KLA SemiCon AI Hackathon - Problem Statement 01: AI-Based Restoration of Degraded Images

Team

REYNA ( kreynareddy@gmail.com )

SACI ( sacimense@gmail.com )

College: PESU(RR)

Problem

Semiconductor inspection sensors generate images degraded by complex pipeline of speckle noise, additive Gaussian noise, and downsampling (in unknown order). This repo restores those degraded low-res scans to KLA’s high-res expectations, recovering fine wafer line patterns hidden by noise/downsampling.

Inputs: .npy 128×128 float32 (unclipped) [-0.08, 1.71] Values outside 0.0–1.0 are real sensor information, not “bad” data Target: .npy 256×256 float32 (clipped) [0.0, 1.0] Task: Joint denoising + 2x super-resolution (128×128 → 256×256)

Approach

We propose a Joint Denoise-SR ResUNet (~8.32M parameters) that denoises and performs 2x super-resolution (SR) simultaneously in a single forward pass instead of sequentially, thereby optimizing the joint task rather than cascading two separate tasks.

Key innovations:

Raw: Input images contain floats outside of 0.0–1.0, which must be preserved as they represent real information. We avoid clipping on input and only clamp to 0.0–1.0 on the model output to match target.

Multi-scale ResUNet: Encoder-decoder structure with 3 downsampling stages gives context for effective global noise modeling while preserving fine details through skip connections.

Residual Channel Attention Blocks: Adaptive channel-wise feature recalibration allows suppression of noisy frequencies while retaining fine wafer patterns.

Pixel Shuffle upsampling: Avoids transposed convolutions’ checkerboard artifacts. Global bicubic shortcut: Similar to UNet++, the model learns residual over bicubic upsampling for faster convergence.

Loss: 1.0 × Charbonnier + 0.2 × SSIM + 0.1 × Sobel-edge

Charbonnier loss: Robust to speckle noise outliers vs MSE/L1.

SSIM loss: Directly optimizes structural similarity, which drives the hackathon’s metrics.

Sobel-edge loss: Additional edge-aware smoothing in X/Y directions preserves fine wafer line edges from being overly smoothed.

Results

Metric Bicubic Baseline Ours

PSNR 22.4 db 28.81 db

SSIM 0.53 0.792

LPIPS low 0.2555

_Add before/after comparisons (degraded input → our output → ground truth) here_ - e.g. insert images from weights/visualizations/

Repo Structure

.

├── README.md

├── requirements.txt

├── model.py # Joint Denoise-SR ResUNet definition

├── dataset.py # Paired dataset loader & augmentations

├── losses.py # Charbonnier + SSIM + Sobel-edge composite loss

├── utils.py # PSNR / SSIM / LPIPS metrics + visualization helpers

├── train.py # Training script (reproducible from scratch)

├── eval.py # Standalone evaluation / inference script

├── weights/

│ ├── model.pt # Checkpoint of trained model

│ └── training_log.csv # Training metrics per epoch

└── outputs/

└── restored_test/ # Model’s output for the test set (.npy)

_Note: If weights/model.pt is too big for Github, it will need to be hosted elsewhere._

Setup

Python 3.10+ with CUDA-capable GPU recommended (CPU-only also works, but much slower). Clone the repo, then install dependencies via pip:

git clone https://github.com/yourspixie/Wafer-Minds-KLA-PS01.git

cd Wafer-Minds-KLA-PS01

pip install -r requirements.txt

Usage (Inference / Evaluation)

eval.py loads the trained model checkpoint and applies it to every .npy file in input_dir, saving the restored results to output_dir . It runs out-of-the-box with no modifications other than the two required args below:

python eval.py --input_dir /path/to/test/npy/files --output_dir /path/to/save/restored/outputs

Arguments:

Flag Required Description

--input_dir Yes Directory of degraded test .npy files (128×128 float32)

--output_dir Yes Directory of restored .npy files (256×256 float32, [0.0, 1.0]) (created if not exists)

--weights No Path to a model checkpoint. Default: weights/model.pt

Each file in output_dir will have the same name as the corresponding input file.

Training (Reproducing From Scratch)

python train.py --data_dir /path/to/train --epochs 15 --batch_size 16 --lr 2e-4

training expects the paired degraded/target .npy files under --data_dir (see dataset.py for details). The loss, epoch PSNR, and SSIM will be logged to weights/training_log.csv and visualized in weights/visualizations as the training proceeds.

Model Weights

File: weights/model.pt

Format: PyTorch state_dict (.pt)

Size: 33.1 MB

Technology / Feasibility

Framework PyTorch

Training Hardware: _GPU type, cloud provider_

Training Time: _Time taken_

Model size: ~8.32M parameters

Inference speed: ~2ms–5ms/image on H100/A100 GPUs

Requirements

The full dependency listing (from pip freeze) for the environment in which the model was trained is in requirements.txt
