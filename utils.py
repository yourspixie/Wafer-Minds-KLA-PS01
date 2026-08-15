import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def calculate_psnr(pred, gt):
    """
    Computes PSNR between pred and gt numpy arrays or PyTorch tensors in range [0, 1].
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


_GAUSS_WINDOW = None

def get_gaussian_window(window_size=11, sigma=1.5, device='cpu'):
    global _GAUSS_WINDOW
    if _GAUSS_WINDOW is None or _GAUSS_WINDOW.device != device:
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / (2 * sigma ** 2) for x in range(window_size)]))
        gauss = gauss / gauss.sum()
        _1D = gauss.unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        _GAUSS_WINDOW = _2D.to(device)
    return _GAUSS_WINDOW


def calculate_ssim(pred, gt):
    """
    Fast PyTorch Tensor vectorized SSIM computation.
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(np.ascontiguousarray(pred)).float()
    if not isinstance(gt, torch.Tensor):
        gt = torch.from_numpy(np.ascontiguousarray(gt)).float()

    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)

    if gt.ndim == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
    elif gt.ndim == 3:
        gt = gt.unsqueeze(0)

    device = pred.device
    window = get_gaussian_window(11, 1.5, device=device)

    mu1 = F.conv2d(pred, window, padding=5)
    mu2 = F.conv2d(gt, window, padding=5)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=5) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=5) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=5) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def save_comparison_triptych(input_np, pred_np, gt_np, save_path, metrics_str=""):
    """
    Saves a clean 4-panel comparison grid:
    [ NoisyLR (Bicubic 256x256) | Model Output | Ground Truth | Absolute Error Map ]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_2d = np.squeeze(input_np)
    pred_2d = np.squeeze(pred_np)
    gt_2d = np.squeeze(gt_np)

    # Bicubic resize input for visual side-by-side matching resolution
    if input_2d.shape != gt_2d.shape:
        import cv2
        input_vis = cv2.resize(input_2d, (gt_2d.shape[1], gt_2d.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        input_vis = input_2d

    error_map = np.abs(pred_2d - gt_2d)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Panel 1: Input NoisyLR
    im0 = axes[0].imshow(input_vis, cmap='gray')
    axes[0].set_title(f"NoisyLR Input\n(min={input_2d.min():.2f}, max={input_2d.max():.2f})", fontsize=10)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Restored Output
    im1 = axes[1].imshow(pred_2d, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Restored Output\n{metrics_str}", fontsize=10, fontweight='bold', color='navy')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Ground Truth
    im2 = axes[2].imshow(gt_2d, cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].set_title("Ground Truth (GT)", fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Error Map
    im3 = axes[3].imshow(error_map, cmap='inferno', vmin=0.0, vmax=0.2)
    axes[3].set_title("Abs Error |Pred - GT|", fontsize=10)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    gt = np.random.rand(256, 256).astype(np.float32)
    pred = np.clip(gt + np.random.normal(0, 0.02, (256, 256)).astype(np.float32), 0, 1)

    psnr_val = calculate_psnr(pred, gt)
    ssim_val = calculate_ssim(pred, gt)
    print(f"Vectorized PSNR Test: {psnr_val:.2f} dB, Vectorized SSIM Test: {ssim_val:.4f}")
