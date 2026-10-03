"""
HiC-SuperNet (PyTorch)
======================
Multi-scale attention-based CNN for Hi-C contact map enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keras BatchNormalization defaults: momentum=0.99, epsilon=1e-3
# PyTorch equivalent: momentum = 1 - 0.99 = 0.01, eps = 1e-3
BN_KW = dict(eps=1e-3, momentum=0.01)


class MultiScaleDilatedResBlock(nn.Module):
    """Three parallel 3x3 dilated convs (d = 1, 2, 4) -> concat -> BN -> ReLU
    -> 3x3 conv -> BN, plus a 1x1 projected shortcut, then ReLU."""

    def __init__(self, filters):
        super().__init__()
        branch = filters // 3
        self.conv_d1 = nn.Conv2d(filters, branch, 3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(filters, branch, 3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv2d(filters, branch, 3, padding=4, dilation=4)
        self.bn1 = nn.BatchNorm2d(branch * 3, **BN_KW)
        self.conv2 = nn.Conv2d(branch * 3, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters, **BN_KW)
        self.shortcut_conv = nn.Conv2d(filters, filters, 1)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        out = torch.cat([self.conv_d1(x), self.conv_d2(x), self.conv_d4(x)], dim=1)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        return F.relu(out + shortcut)


class DualAttention(nn.Module):
    """CBAM-style channel attention (shared MLP on global avg/max pool)
    followed by spatial attention (7x7 conv on channel-wise avg/max)."""

    def __init__(self, filters, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(filters, filters // reduction)
        self.fc2 = nn.Linear(filters // reduction, filters)
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3)

    def _mlp(self, v):
        return self.fc2(F.relu(self.fc1(v)))

    def forward(self, x):
        n, c = x.shape[:2]
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        channel_att = torch.sigmoid(self._mlp(avg_pool) + self._mlp(max_pool))
        x = x * channel_att.view(n, c, 1, 1)

        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class Generator(nn.Module):
    """
    HiC-SuperNet generator.

    - 7x7 feature-extraction conv -> BN -> ReLU
    - num_blocks x MultiScaleDilatedResBlock, DualAttention after every 2nd block
    - global residual connection from the initial features
    - progressive refinement: 3x3 convs with 2x, 1x, 0.5x filters (ReLU)
    - linear 3x3 output conv

    Named `Generator` so it is a drop-in replacement in DiCARN-style scripts.
    """

    def __init__(self, base_filters=64, num_blocks=8, in_channels=1):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, base_filters, 7, padding=3)
        self.initial_bn = nn.BatchNorm2d(base_filters, **BN_KW)

        self.msd_blocks = nn.ModuleList([MultiScaleDilatedResBlock(base_filters) for _ in range(num_blocks)])
        self.dual_attentions = nn.ModuleList([
            DualAttention(base_filters) if (i + 1) % 2 == 0 else nn.Identity()
            for i in range(num_blocks)
        ])

        self.recon_conv1 = nn.Conv2d(base_filters, base_filters * 2, 3, padding=1)
        self.recon_conv2 = nn.Conv2d(base_filters * 2, base_filters, 3, padding=1)
        self.recon_conv3 = nn.Conv2d(base_filters, base_filters // 2, 3, padding=1)
        self.output_conv = nn.Conv2d(base_filters // 2, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        # Match Keras: he_normal where the original used it, glorot_uniform elsewhere, zero biases
        he = {self.initial_conv, self.recon_conv1, self.recon_conv2, self.recon_conv3, self.output_conv}
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if m in he:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        initial_features = x
        for block, att in zip(self.msd_blocks, self.dual_attentions):
            x = att(block(x))
        x = x + initial_features
        x = F.relu(self.recon_conv1(x))
        x = F.relu(self.recon_conv2(x))
        x = F.relu(self.recon_conv3(x))
        return self.output_conv(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = Generator()
    y = net(torch.rand(2, 1, 40, 40))
    print('Output shape:', tuple(y.shape))
    print(f'Trainable params: {count_params(net):,}')
