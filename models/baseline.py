"""Simple dual-ResNet late-fusion baseline for paired BUS/CEUS videos."""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class FrameResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, video):
        batch, channels, frames, height, width = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.features(x).flatten(1)
        return x.view(batch, frames, -1).mean(dim=1)


class DualResNet18Baseline(nn.Module):
    """Independent frame encoders followed by temporal mean and late fusion."""

    def __init__(self, num_classes=2, pretrained=True, dropout=0.3, **_):
        super().__init__()
        self.bus_encoder = FrameResNet18(pretrained)
        self.ceus_encoder = FrameResNet18(pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 * 2, num_classes),
        )

    def forward(self, bus, ceus):
        bus_features = self.bus_encoder(bus)
        ceus_features = self.ceus_encoder(ceus)
        return self.classifier(torch.cat((bus_features, ceus_features), dim=1))
