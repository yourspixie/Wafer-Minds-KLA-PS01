import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 variant)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, gt):
        diff = pred - gt
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


def gaussian_window(window_size, sigma):
    gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / (2 * sigma ** 2) for x in range(window_size)]))
    return gauss / gauss.sum()


def create_window(window_size, channel=1, sigma=1.5):
    _1D_window = gaussian_window(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window




class SSIMLoss(nn.Module):
    """Optimized Differentiable PyTorch SSIM Loss"""
    def __init__(self, window_size=11, sigma=1.5, size_average=True, channels=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = channels

        
        gauss = torch.exp(-torch.arange(window_size).sub(window_size // 2).pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        _1D = gauss.unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        
        
        window = _2D.expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer('window', window)
        self.padding = window_size // 2

    def forward(self, img1, img2):
        channel = img1.size(1)

       
        if channel != self.channels:
            window = self.window.repeat(channel, 1, 1, 1)
        else:
            window = self.window

        
        mu1 = F.conv2d(img1, window, padding=self.padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.padding, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.padding, groups=channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding=self.padding, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        sobel_kernel = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(1)
        
        self.register_buffer('sobel', sobel_kernel)

    def forward(self, pred, gt):

        pred_grad = F.conv2d(pred, self.sobel, padding=1)
        gt_grad   = F.conv2d(gt, self.sobel, padding=1)
        return F.l1_loss(pred_grad, gt_grad)

class CompositeRestorationLoss(nn.Module):
    """
    Combined Loss for Image Restoration:
    L = w_charbonnier * L_charbonnier + w_ssim * L_ssim + w_edge * L_edge
    """
    def __init__(self, w_charbonnier=1.0, w_ssim=0.2, w_edge=0.1):
        super(CompositeRestorationLoss, self).__init__()
        self.w_charbonnier = w_charbonnier
        self.w_ssim = w_ssim
        self.w_edge = w_edge

        self.charbonnier = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = SobelEdgeLoss()

    def forward(self, pred, gt):
        l_char = self.charbonnier(pred, gt)
        l_ssim = self.ssim_loss(pred, gt)
        l_edge = self.edge_loss(pred, gt)

        total_loss = self.w_charbonnier * l_char + self.w_ssim * l_ssim + self.w_edge * l_edge
        return total_loss, {'l_char': l_char.item(), 'l_ssim': l_ssim.item(), 'l_edge': l_edge.item()}


if __name__ == "__main__":
    pred = torch.rand(2, 1, 256, 256, requires_grad=True)
    gt = torch.rand(2, 1, 256, 256)

    criterion = CompositeRestorationLoss()
    loss, components = criterion(pred, gt)
    loss.backward()

    print(f"Total Loss: {loss.item():.4f}")
    print(f"Loss Components: {components}")
    print("Loss backward test PASSED cleanly!")
