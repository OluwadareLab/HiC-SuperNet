"""
HiC-SuperNet loss (PyTorch port of losses.improved_loss in the original repo):

    loss = 0.4 * MSE + 0.2 * MAE + 0.3 * (1 - Pearson) + 0.1 * (1 - SSIM)

SSIM here follows tf.image.ssim (11x11 Gaussian window, sigma = 1.5,
'valid' filtering, K1 = 0.01, K2 = 0.03, max_val = 1) so the loss matches the
TensorFlow version. Evaluation metrics elsewhere use Utils/SSIM.py, the same
SSIM that DiCARN reports, so scores stay comparable with DiCARN.
"""

import torch
import torch.nn.functional as F


def _gaussian_window(size=11, sigma=1.5, device=None, dtype=None):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def tf_ssim(x, y, max_val=1.0, size=11, sigma=1.5, k1=0.01, k2=0.03):
    """Per-image SSIM, shape (N,), equivalent to tf.image.ssim for 1-channel input."""
    c = x.shape[1]
    w = _gaussian_window(size, sigma, x.device, x.dtype).expand(c, 1, size, size)
    c1, c2 = (k1 * max_val) ** 2, (k2 * max_val) ** 2
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return (num / den).mean(dim=(1, 2, 3))


def pearson_per_sample(pred, target, eps=1e-8):
    p = pred.flatten(1)
    t = target.flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    return (p * t).sum(1) / (torch.sqrt((p ** 2).sum(1) * (t ** 2).sum(1)) + eps)


def improved_loss(pred, target):
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    pearson = 1.0 - pearson_per_sample(pred, target).mean()
    ssim = 1.0 - tf_ssim(target, pred).mean()
    return 0.4 * mse + 0.2 * mae + 0.3 * pearson + 0.1 * ssim


def pcc(pred, target, eps=1e-8):
    """Pearson correlation over all pixels of the batch (as in the original calculate_pcc)."""
    p = pred.flatten() - pred.mean()
    t = target.flatten() - target.mean()
    return (p * t).sum() / (torch.sqrt((p ** 2).sum() * (t ** 2).sum()) + eps)
