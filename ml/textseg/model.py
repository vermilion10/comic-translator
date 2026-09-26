"""The text-map detector: an ImageNet backbone, a small FPN, one probability map.

A per-pixel map, not boxes: the training labels mix finders that draw text
regions differently (a line, a block, a whole balloon), and those paint the
same pixels even when their boxes disagree. Regions are grouped after the
model, in detector.py.

Input  `images` [1, 3, H, W], RGB 0..1, H and W multiples of 32 (1280x1280,
       letterboxed with grey 114 padding).
Output `prob`   [1, 1, H/4, W/4], probability that a pixel is inside the
       shrunk core of a text region.
Normalisation happens inside the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BACKBONE = "mobilenetv3_large_100"
FPN_CHANNELS = 96
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OUTPUT_STRIDE = 4


def conv_bn_relu(cin: int, cout: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TextMapNet(nn.Module):
    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        # Strides 4, 8, 16, 32; out_indices are timm feature-info indices, checked below.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )
        reductions = self.backbone.feature_info.reduction()
        assert reductions == [4, 8, 16, 32], f"{backbone} gives strides {reductions}"
        channels = self.backbone.feature_info.channels()

        self.lateral = nn.ModuleList(nn.Conv2d(c, FPN_CHANNELS, 1) for c in channels)
        self.smooth = nn.ModuleList(conv_bn_relu(FPN_CHANNELS, FPN_CHANNELS // 4) for _ in channels)
        self.head = nn.Sequential(
            conv_bn_relu(FPN_CHANNELS, FPN_CHANNELS // 2),
            nn.Conv2d(FPN_CHANNELS // 2, 1, 1),
        )
        # Start the map near "no text" (sigmoid(-4) ~ 0.02), since most pixels are background.
        nn.init.constant_(self.head[-1].bias, -4.0)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Logits at stride 4. Use probabilities() or the export wrapper for 0..1."""
        features = self.backbone((images - self.mean) / self.std)
        laterals = [conv(f) for conv, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        size = laterals[0].shape[-2:]
        merged = torch.cat(
            [
                F.interpolate(smooth(level), size=size, mode="nearest") if i else smooth(level)
                for i, (smooth, level) in enumerate(zip(self.smooth, laterals))
            ],
            dim=1,
        )
        return self.head(merged)

    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(images))


class ExportWrapper(nn.Module):
    """What goes into the ONNX file: 0..1 image in, 0..1 map out."""

    def __init__(self, net: TextMapNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net.probabilities(images)
