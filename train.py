import os
import time
import json
import csv
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Maximize CPU multi-threading performance if on CPU
if not torch.cuda.is_available():
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)

from dataset import get_train_val_dataloaders
from model import JointDenoiseSRResUNet
from losses import CompositeRestorationLoss
from utils import calculate_psnr, calculate_ssim, save_comparison_triptych

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================", flush=True)
    print(f" Training Joint Denoise-SR Model on Device: {device} (Threads: {torch.get_num_threads()})", flush=True)
    print(f"==================================================", flush=True)

    # 1. Create Directories
    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 2. Setup DataLoaders
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # 3. Instantiate Model
    model = JointDenoiseSRResUNet(in_channels=1, out_channels=1, base_channels=64, scale_factor=2).to(device)

    # 4. Setup Loss, Optimizer, Scheduler
    criterion = CompositeRestorationLoss(w_charbonnier=1.0, w_ssim=0.2, w_edge=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision scaler
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # LPIPS model setup
    lpips_net = None
    if LPIPS_AVAILABLE and torch.cuda.is_available():
        try:
            lpips_net = lpips.LPIPS(net='alex', verbose=False).to(device)
            lpips_net.eval()
            print("LPIPS metric module initialized (GPU accelerated).", flush=True)
        except Exception as e:
            print(f"LPIPS init warning: {e}", flush=True)
            lpips_net = None
    else:
        print("LPIPS calculation disabled on CPU for high training throughput (PSNR & SSIM active).", flush=True)

    # 5. Tracking metrics
    best_val_ssim = -1.0
    history = []
    log_csv_path = os.path.join(args.save_dir, "training_log.csv")

    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'TrainLoss', 'ValLoss', 'ValPSNR', 'ValSSIM', 'ValLPIPS', 'LR', 'TimeSec'])

    print(f"\nStarting training for {args.epochs} epochs...\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()

        train_loss_accum = 0.0
        train_batches = 0

        for idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(x_batch)
                    loss, _ = criterion(pred, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x_batch)
                loss, _ = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

            if args.fast_test and idx >= 5:
                break

        avg_train_loss = train_loss_accum / train_batches
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        val_psnr_list = []
        val_ssim_list = []
        val_lpips_list = []

        sample_vis = None

        with torch.no_grad():
            for idx, (x_val, y_val) in enumerate(val_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                pred_val = model(x_val)
                v_loss, _ = criterion(pred_val, y_val)

                val_loss_accum += v_loss.item()
                val_batches += 1

                # Calculate PSNR and SSIM in batch tensor mode
                batch_ssim = calculate_ssim(pred_val, y_val)
                val_ssim_list.append(batch_ssim)

                pred_np_batch = pred_val.cpu().numpy()
                gt_np_batch = y_val.cpu().numpy()

                for b in range(pred_np_batch.shape[0]):
                    val_psnr_list.append(calculate_psnr(pred_np_batch[b, 0], gt_np_batch[b, 0]))

                # LPIPS calculation if GPU enabled
                if lpips_net is not None:
                    p_3ch = (pred_val.repeat(1, 3, 1, 1) * 2.0) - 1.0
                    g_3ch = (y_val.repeat(1, 3, 1, 1) * 2.0) - 1.0
                    dist = lpips_net(p_3ch, g_3ch).mean().item()
                    val_lpips_list.append(dist)

                if sample_vis is None:
                    sample_vis = (x_val[0, 0].cpu().numpy(), pred_val[0, 0].cpu().numpy(), y_val[0, 0].cpu().numpy())

                if args.fast_test and idx >= 2:
                    break

        avg_val_loss = val_loss_accum / val_batches
        mean_psnr = float(np.mean(val_psnr_list))
        mean_ssim = float(np.mean(val_ssim_list))
        mean_lpips = float(np.mean(val_lpips_list)) if val_lpips_list else 0.0
        elapsed_sec = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val PSNR: {mean_psnr:.2f} dB | Val SSIM: {mean_ssim:.4f} | Val LPIPS: {mean_lpips:.4f} | Time: {elapsed_sec:.1f}s", flush=True)

        # Save to CSV
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_val_loss, mean_psnr, mean_ssim, mean_lpips, current_lr, elapsed_sec])

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_psnr': mean_psnr,
            'val_ssim': mean_ssim,
            'val_lpips': mean_lpips,
            'lr': current_lr
        })

        # Save Checkpoint if best SSIM
        if mean_ssim > best_val_ssim:
            best_val_ssim = mean_ssim
            best_model_path = os.path.join(args.save_dir, "model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ssim': mean_ssim,
                'val_psnr': mean_psnr,
                'val_lpips': mean_lpips,
                'config': vars(args)
            }, best_model_path)
            print(f"  --> Saved BEST Model Checkpoint to {best_model_path} (Val SSIM: {mean_ssim:.4f})", flush=True)

        # Save visualization every 5 epochs or on best
        if sample_vis is not None and (epoch % 5 == 0 or epoch == 1 or epoch == args.epochs):
            vis_path = os.path.join(vis_dir, f"vis_epoch_{epoch:03d}.png")
            m_str = f"PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}"
            save_comparison_triptych(sample_vis[0], sample_vis[1], sample_vis[2], vis_path, metrics_str=m_str)

    # Save complete history JSON
    with open(os.path.join(args.save_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete! Best Val SSIM: {best_val_ssim:.4f}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Joint Denoise-SR Model")
    parser.add_argument("--data_dir", type=str, default=r"d:\Saci\iesa\train", help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default=r"d:\Saci\iesa\weights", help="Directory to save model checkpoints and logs")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set split ratio")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fast_test", action="store_true", help="Run fast test mode for pipeline verification")

    args = parser.parse_args()
    train_model(args)
