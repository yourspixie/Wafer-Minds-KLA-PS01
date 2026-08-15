"""
view_npy_grid.py

Quickly view a folder of .npy image arrays (e.g. restored_test/) as a single
PNG grid, since Windows Explorer can't preview .npy files directly.

Usage:
    python view_npy_grid.py --input_dir "D:\Saci\iesa\outputs\restored_test" --out grid.png
    python view_npy_grid.py --input_dir path/to/folder --n 25 --cols 5 --start 0
    python view_npy_grid.py --input_dir path/to/folder --files 000000.npy 000025.npy 000050.npy

Requires: numpy, matplotlib
    pip install numpy matplotlib
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_npy_as_display(path):
    arr = np.load(path)
    arr = np.squeeze(arr)  # drop channel dim if (1,H,W) or (H,W,1)
    return arr


def main():
    parser = argparse.ArgumentParser(description="View a folder of .npy images as a grid PNG.")
    parser.add_argument("--input_dir", required=True, help="Folder containing .npy files")
    parser.add_argument("--out", default="npy_grid.png", help="Output PNG path")
    parser.add_argument("--n", type=int, default=25, help="Number of images to show (ignored if --files given)")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns")
    parser.add_argument("--start", type=int, default=0, help="Start index into sorted file list")
    parser.add_argument("--files", nargs="*", default=None, help="Specific filenames to view instead of --n/--start")
    parser.add_argument("--clip", nargs=2, type=float, default=None,
                         help="Optional min max to clip display range, e.g. --clip 0 1")
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap (default: gray)")
    args = parser.parse_args()

    if args.files:
        filenames = args.files
    else:
        all_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".npy"))
        filenames = all_files[args.start: args.start + args.n]

    if not filenames:
        print("No .npy files found / selected.")
        return

    n = len(filenames)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, fname in enumerate(filenames):
        path = os.path.join(args.input_dir, fname)
        try:
            arr = load_npy_as_display(path)
            vmin, vmax = (args.clip if args.clip else (None, None))
            axes[i].imshow(arr, cmap=args.cmap, vmin=vmin, vmax=vmax)
            axes[i].set_title(fname, fontsize=8)
        except Exception as e:
            axes[i].set_title(f"{fname}\nERROR: {e}", fontsize=7, color="red")
        axes[i].axis("off")

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved grid of {n} images to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()