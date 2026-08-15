import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PairedNpyDataset(Dataset):
    """
    PyTorch Dataset for paired numpy (.npy) arrays:
    NoisyLR (128x128 float32, unnormalized) -> GT (256x256 float32, [0,1])
    """
    def __init__(self, gt_paths, noisy_paths, is_train=True):
        assert len(gt_paths) == len(noisy_paths), f"Mismatch: {len(gt_paths)} GT vs {len(noisy_paths)} Noisy"
        self.gt_paths = gt_paths
        self.noisy_paths = noisy_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        noisy_img = np.load(self.noisy_paths[idx]).astype(np.float32) # (128, 128)
        gt_img = np.load(self.gt_paths[idx]).astype(np.float32)       # (256, 256)

        # Add channel dimension -> (1, H, W)
        if noisy_img.ndim == 2:
            noisy_img = noisy_img[np.newaxis, ...]
        if gt_img.ndim == 2:
            gt_img = gt_img[np.newaxis, ...]

        # Data augmentation for training
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                noisy_img = np.flip(noisy_img, axis=2).copy()
                gt_img = np.flip(gt_img, axis=2).copy()

            # Random vertical flip
            if random.random() > 0.5:
                noisy_img = np.flip(noisy_img, axis=1).copy()
                gt_img = np.flip(gt_img, axis=1).copy()

            # Random 90-degree rotations
            k = random.randint(0, 3)
            if k > 0:
                noisy_img = np.rot90(noisy_img, k=k, axes=(1, 2)).copy()
                gt_img = np.rot90(gt_img, k=k, axes=(1, 2)).copy()

        noisy_tensor = torch.from_numpy(noisy_img)
        gt_tensor = torch.from_numpy(gt_img)

        return noisy_tensor, gt_tensor


def get_train_val_dataloaders(data_dir=r"d:\Saci\iesa\train", batch_size=16, val_ratio=0.1, num_workers=0, seed=42):
    gt_dir = os.path.join(data_dir, "GT")
    noisy_dir = os.path.join(data_dir, "NoisyLR")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))

    # Verify matching filenames
    pairs = []
    noisy_dict = {os.path.basename(f): f for f in noisy_files}
    for g_path in gt_files:
        base = os.path.basename(g_path)
        if base in noisy_dict:
            pairs.append((g_path, noisy_dict[base]))

    print(f"Total verified paired samples: {len(pairs)}")

    # Reproducible random split by index
    random.seed(seed)
    random.shuffle(pairs)

    val_size = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    print(f"Train split: {len(train_pairs)} samples | Val split: {len(val_pairs)} samples")

    train_gt, train_noisy = zip(*train_pairs)
    val_gt, val_noisy = zip(*val_pairs)

    train_dataset = PairedNpyDataset(train_gt, train_noisy, is_train=True)
    val_dataset = PairedNpyDataset(val_gt, val_noisy, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

if __name__ == "__main__":
    t_loader, v_loader = get_train_val_dataloaders(batch_size=4)
    for x, y in t_loader:
        print(f"Batch X shape: {x.shape}, dtype: {x.dtype}, min: {x.min():.4f}, max: {x.max():.4f}")
        print(f"Batch Y shape: {y.shape}, dtype: {y.dtype}, min: {y.min():.4f}, max: {y.max():.4f}")
        break
