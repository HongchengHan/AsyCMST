"""AsyCMST model from Medical Image Analysis 112 (2026) 104127."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class FrameFeatureEncoder(nn.Module):
    def __init__(self, embed_dim=384, pretrained=False):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Conv2d(512, embed_dim, 1)

    def forward(self, video):
        b, c, t, h, w = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.projection(self.encoder(x))
        oh, ow = x.shape[-2:]
        return x.view(b, t, -1, oh, ow).permute(0, 2, 1, 3, 4)


class FrameSelfSorting(nn.Module):
    def __init__(self, embed_dim, clamp=0.9):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.clamp = clamp

    def forward(self, features):
        pooled = 0.5 * (features.mean((-1, -2)) + features.amax((-1, -2)))
        pooled = pooled.transpose(1, 2)  # B, T, C
        return self.scorer(pooled).squeeze(-1)

    def loss(self, scores):
        t = scores.shape[1]
        if t < 2:
            return scores.sum() * 0
        # score[j] should be greater than score[i] for every j > i.
        differences = scores[:, None, :] - scores[:, :, None]
        upper = torch.triu(torch.ones(t, t, device=scores.device, dtype=torch.bool), diagonal=1)
        probabilities = torch.sigmoid(differences[:, upper]).clamp(max=self.clamp)
        return -torch.log(probabilities.clamp_min(1e-7)).mean()


class MaskDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        channels = [embed_dim, 192, 96, 48, 24, 1]
        layers = []
        for i in range(5):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], 4, 2, 1))
            if i < 4:
                layers.extend((nn.BatchNorm2d(channels[i + 1]), nn.ReLU(inplace=True)))
        self.decoder = nn.Sequential(*layers)

    def forward(self, features, output_size):
        b, c, t, h, w = features.shape
        x = features.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = F.interpolate(self.decoder(x), size=output_size, mode="bilinear", align_corners=False)
        return x.view(b, t, *output_size)


class AsymmetricCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, direction):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim, self.direction = embed_dim // num_heads, direction
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(embed_dim, embed_dim) for _ in range(3))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_value, frames, locations):
        b = query.shape[0]
        q = query.view(b, frames, locations, self.embed_dim)
        k = key_value.view(b, frames, locations, self.embed_dim)
        v = key_value.view(b, frames, locations, self.embed_dim)
        if self.direction == "temporal":
            q, k, v = (x.permute(0, 2, 1, 3).reshape(b * locations, frames, self.embed_dim) for x in (q, k, v))
            group, length = b * locations, frames
        else:
            q, k, v = (x.reshape(b * frames, locations, self.embed_dim) for x in (q, k, v))
            group, length = b * frames, locations
        q = self.q_proj(q).view(group, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(group, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(group, length, self.num_heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1) / self.head_dim**0.5, -1)
        out = self.out_proj((weights @ v).transpose(1, 2).reshape(group, length, self.embed_dim))
        if self.direction == "temporal":
            out = out.view(b, locations, frames, self.embed_dim).permute(0, 2, 1, 3)
        else:
            out = out.view(b, frames, locations, self.embed_dim)
        return out.reshape(b, frames * locations, self.embed_dim)


class ACMSTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, direction, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = AsymmetricCrossAttention(embed_dim, num_heads, direction)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x, other, frames, locations):
        x = self.norms[0](x + self.self_attention(x, x, x, need_weights=False)[0])
        x = self.norms[1](x + self.cross_attention(x, other, frames, locations))
        return self.norms[2](x + self.ffn(x))


class DualStreamNetwork(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, frames, locations):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, frames * locations, embed_dim) * 0.02)
        self.bus_blocks = nn.ModuleList([ACMSTBlock(embed_dim, num_heads, "temporal") for _ in range(num_layers)])
        self.ceus_blocks = nn.ModuleList([ACMSTBlock(embed_dim, num_heads, "spatial") for _ in range(num_layers)])

    def forward(self, bus, ceus):
        b, c, t, h, w = bus.shape
        n = h * w
        def tokenize(x):
            return x.permute(0, 2, 3, 4, 1).reshape(b, t * n, c) + self.position
        bus, ceus = tokenize(bus), tokenize(ceus)
        for bus_block, ceus_block in zip(self.bus_blocks, self.ceus_blocks):
            previous_bus, previous_ceus = bus, ceus
            bus = bus_block(previous_bus, previous_ceus, t, n)
            ceus = ceus_block(previous_ceus, previous_bus, t, n)
        return (bus + ceus).mean(1)


class AsyCMST(nn.Module):
    def __init__(self, embed_dim=384, num_classes=2, num_layers=6, num_heads=6,
                 sample_duration=16, sample_size=224, use_mask=False, pretrained=False):
        super().__init__()
        locations = (sample_size // 32) ** 2
        self.use_mask = use_mask
        self.bus_encoder = FrameFeatureEncoder(embed_dim, pretrained)
        self.ceus_encoder = FrameFeatureEncoder(embed_dim, pretrained)
        self.frame_sorting = FrameSelfSorting(embed_dim)
        self.mask_decoder = MaskDecoder(embed_dim) if use_mask else None
        self.dual_stream = DualStreamNetwork(embed_dim, num_layers, num_heads, sample_duration, locations)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, bus, ceus, return_aux=False):
        bus_features, ceus_features = self.bus_encoder(bus), self.ceus_encoder(ceus)
        logits = self.classifier(self.dual_stream(bus_features, ceus_features))
        if not return_aux:
            return logits
        output = {"logits": logits, "frame_scores": self.frame_sorting(ceus_features)}
        if self.use_mask:
            output["mask_logits"] = self.mask_decoder(bus_features, bus.shape[-2:])
        return output


def dice_loss(logits, targets, eps=1e-6):
    probabilities, targets = torch.sigmoid(logits), targets.float()
    dims = tuple(range(1, targets.ndim))
    intersection = (probabilities * targets).sum(dims)
    denominator = probabilities.sum(dims) + targets.sum(dims)
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()
