"""Dual-stream temporal Transformer baseline without cross-modal attention."""
import torch
import torch.nn as nn

from models.asycmst import FrameFeatureEncoder


class DualTransformerBaseline(nn.Module):
    """Independent temporal self-attention streams with late feature fusion.

    Each ResNet frame map is spatially pooled before the Transformer.  This
    isolates the value of temporal modelling and avoids turning the baseline
    into another large spatio-temporal architecture.
    """

    def __init__(self, embed_dim=256, num_classes=2, num_layers=2, num_heads=8,
                 sample_duration=16, pretrained=True,
                 dropout=0.1, **_):
        super().__init__()
        self.bus_encoder = FrameFeatureEncoder(embed_dim, pretrained)
        self.ceus_encoder = FrameFeatureEncoder(embed_dim, pretrained)
        self.bus_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ceus_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.bus_position = nn.Parameter(
            torch.randn(1, sample_duration + 1, embed_dim) * 0.02)
        self.ceus_position = nn.Parameter(
            torch.randn(1, sample_duration + 1, embed_dim) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.bus_transformer = nn.TransformerEncoder(
            layer(), num_layers, norm=nn.LayerNorm(embed_dim))
        self.ceus_transformer = nn.TransformerEncoder(
            layer(), num_layers, norm=nn.LayerNorm(embed_dim))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(embed_dim * 2, num_classes))

        nn.init.normal_(self.bus_cls, std=0.02)
        nn.init.normal_(self.ceus_cls, std=0.02)

    @staticmethod
    def tokenize(features):
        return features.mean(dim=(-1, -2)).transpose(1, 2)

    @staticmethod
    def add_cls(tokens, cls_token, position):
        cls = cls_token.expand(tokens.shape[0], -1, -1)
        return torch.cat((cls, tokens), dim=1) + position[:, :tokens.shape[1] + 1]

    def forward(self, bus, ceus):
        bus = self.add_cls(
            self.tokenize(self.bus_encoder(bus)), self.bus_cls, self.bus_position)
        ceus = self.add_cls(
            self.tokenize(self.ceus_encoder(ceus)), self.ceus_cls, self.ceus_position)
        bus = self.bus_transformer(bus)[:, 0]
        ceus = self.ceus_transformer(ceus)[:, 0]
        return self.classifier(torch.cat((bus, ceus), dim=1))
