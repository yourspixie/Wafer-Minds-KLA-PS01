import torch
import torch.nn as nn
import torch.nn.functional as F


class JointDenoiseSRResUNet(nn.Module):
    """
    Lightweight Joint Denoise + 2x Super-Resolution model.

    Input:
        (B, 1, 128, 128)

    Output:
        (B, 1, 256, 256)

    Uses bicubic interpolation as the base image and
    learns a residual correction for denoising/detail recovery.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        scale_factor=2
    ):
        super().__init__()

        self.scale_factor = scale_factor

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual refinement
        self.residual = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                base_channels,
                out_channels,
                3,
                padding=1
            )
        )

    def forward(self, x):

        # -----------------------------------------
        # 1. Bicubic baseline
        # -----------------------------------------
        bicubic = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False
        )

        # Keep baseline within valid image range
        bicubic = torch.clamp(bicubic, 0.0, 1.0)

        # -----------------------------------------
        # 2. Extract features from low-resolution input
        # -----------------------------------------
        features = self.features(x)

        # -----------------------------------------
        # 3. Learn residual at low resolution
        # -----------------------------------------
        residual_lr = self.residual(features)

        # -----------------------------------------
        # 4. Upsample learned residual
        # -----------------------------------------
        residual = F.interpolate(
            residual_lr,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False
        )

        # -----------------------------------------
        # 5. Bicubic + learned correction
        # -----------------------------------------
        output = bicubic + residual

        # -----------------------------------------
        # 6. Valid image range
        # -----------------------------------------
        output = torch.clamp(output, 0.0, 1.0)

        return output


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Joint Denoise-SR Model")
    print("=" * 60)

    # Use CPU explicitly for debugging
    device = torch.device("cpu")

    print("Device:", device)

    # Create model
    model = JointDenoiseSRResUNet().to(device)

    params = count_parameters(model)

    print(f"Trainable parameters: {params:,}")
    print(f"Trainable parameters: {params / 1e6:.3f} M")

    # Create dummy input
    x = torch.rand(
        1,
        1,
        128,
        128,
        device=device
    )

    print("Input shape:", x.shape)
    print(
        f"Input range: "
        f"{x.min().item():.4f} → "
        f"{x.max().item():.4f}"
    )

    # Forward pass
    print("Running forward pass...")

    with torch.no_grad():
        y = model(x)

    print("Forward pass completed!")

    print("Output shape:", y.shape)

    print(
        f"Output range: "
        f"{y.min().item():.4f} → "
        f"{y.max().item():.4f}"
    )

    # Verify expected dimensions
    assert y.shape == (
        1,
        1,
        256,
        256
    ), f"Wrong output shape: {y.shape}"

    # Verify output range
    assert y.min() >= 0.0
    assert y.max() <= 1.0

    print("=" * 60)
    print("MODEL TEST PASSED")
    print("=" * 60)