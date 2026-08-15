import os
import glob
import time
import argparse
import numpy as np
import torch
from model import JointDenoiseSRResUNet

def run_evaluation(input_dir, output_dir, weights_path=None):
    # Auto-detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================", flush=True)
    print(f" Running Evaluation / Inference on Device: {device}", flush=True)
    print(f"==================================================", flush=True)

    # Resolve default weights path if not specified
    if weights_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, "weights", "model.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at: {weights_path}")

    print(f"Loading model checkpoint from: {weights_path}", flush=True)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Initialize model architecture
    model = JointDenoiseSRResUNet(in_channels=1, out_channels=1, base_channels=64, scale_factor=2).to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model initialized and weights loaded successfully.", flush=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all input .npy files
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"Found {len(npy_files)} files in input directory: {input_dir}", flush=True)

    if len(npy_files) == 0:
        print("Warning: No .npy files found in input directory!", flush=True)
        return

    inference_times = []
    success_count = 0
    fail_count = 0

    print("Processing images...", flush=True)
    with torch.no_grad():
        for i, filepath in enumerate(npy_files):
            filename = os.path.basename(filepath)
            out_filepath = os.path.join(output_dir, filename)

            try:
                # Load input array
                input_arr = np.load(filepath).astype(np.float32)

                # Ensure (1, 1, H, W) shape
                if input_arr.ndim == 2:
                    input_tensor = torch.from_numpy(input_arr[np.newaxis, np.newaxis, ...])
                elif input_arr.ndim == 3:
                    input_tensor = torch.from_numpy(input_arr[np.newaxis, ...])
                else:
                    input_tensor = torch.from_numpy(input_arr)

                input_tensor = input_tensor.to(device)

                # Measure inference time
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                output_tensor = model(input_tensor)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                proc_time_ms = (t1 - t0) * 1000.0
                inference_times.append(proc_time_ms)

                # Extract (H, W) array bounded in [0, 1]
                output_arr = output_tensor.squeeze().cpu().numpy().astype(np.float32)
                output_arr = np.clip(output_arr, 0.0, 1.0)

                # Save output .npy
                np.save(out_filepath, output_arr)
                success_count += 1

                if (i + 1) % 50 == 0 or (i + 1) == len(npy_files):
                    print(f"  Processed {i+1}/{len(npy_files)} images | Avg latency: {np.mean(inference_times):.2f} ms/img", flush=True)

            except Exception as e:
                print(f"Error processing file '{filename}': {e}", flush=True)
                fail_count += 1

    mean_latency = np.mean(inference_times) if inference_times else 0.0
    total_time = np.sum(inference_times) / 1000.0 if inference_times else 0.0

    print(f"\n==================================================", flush=True)
    print(f" Evaluation Complete!", flush=True)
    print(f" Total Processed: {len(npy_files)} files", flush=True)
    print(f" Successful: {success_count} | Failed: {fail_count}", flush=True)
    print(f" Total Inference Time: {total_time:.2f} seconds", flush=True)
    print(f" Average Latency per Image: {mean_latency:.2f} ms", flush=True)
    print(f" Outputs saved to: {output_dir}", flush=True)
    print(f"==================================================", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate / Restore NoisyLR images for KLA Hackathon")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input NoisyLR .npy directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to write restored .npy directory")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights checkpoint (.pt)")

    args = parser.parse_args()
    run_evaluation(args.input_dir, args.output_dir, args.weights)
