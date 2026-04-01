import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FrameFeatureEncoding(nn.Module):
    def __init__(self, embed_dim=384):
        super(FrameFeatureEncoding, self).__init__()
        # Load ResNet18 without the final fully connected layer
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        # 1x1 conv to reduce channels to embed_dim
        self.conv1x1 = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, 3, T, H, W) where T=16, H=224, W=224
        B, C, T, H, W = x.shape
        # Reshape to (B*T, 3, H, W) for frame-wise processing
        x = x.view(B * T, C, H, W)
        # Encode each frame
        features = self.encoder(x)  # (B*T, 512, 7, 7)
        # Apply 1x1 conv
        features = self.conv1x1(features)  # (B*T, embed_dim, 7, 7)
        # Reshape back to (B, embed_dim, T, 7*7)
        features = features.view(B, self.embed_dim, T, -1)  # (B, embed_dim, 16, 49)
        return features

class AsymmetricCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mode, T=16, HW=49):
        super(AsymmetricCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mode = mode
        self.T = T
        self.HW = HW
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        B, seq_len, _ = q.shape
        T = self.T
        HW = self.HW
        assert seq_len == T * HW

        if self.mode == 'branch1':
            # Q[t,n] attends to K[:,n] BUS-to-CEUS
            q_reshaped = q.view(B, T, HW, self.embed_dim)
            k_reshaped = k.view(B, T, HW, self.embed_dim)
            v_reshaped = v.view(B, T, HW, self.embed_dim)

            # Flatten for attention: (B*HW, T, embed_dim)
            q_flat = q_reshaped.permute(0, 2, 1, 3).contiguous().view(B * HW, T, self.embed_dim)
            k_flat = k_reshaped.permute(0, 2, 1, 3).contiguous().view(B * HW, T, self.embed_dim)
            v_flat = v_reshaped.permute(0, 2, 1, 3).contiguous().view(B * HW, T, self.embed_dim)

            # Projections
            q_p = self.q_proj(q_flat).view(B * HW, T, self.num_heads, self.head_dim).transpose(1, 2)
            k_p = self.k_proj(k_flat).view(B * HW, T, self.num_heads, self.head_dim).transpose(1, 2)
            v_p = self.v_proj(v_flat).view(B * HW, T, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention
            scores = torch.matmul(q_p, k_p.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, v_p)

            # Reshape back
            context = context.transpose(1, 2).contiguous().view(B * HW, T, self.embed_dim)
            output = self.out_proj(context)
            output = output.view(B, HW, T, self.embed_dim).permute(0, 2, 1, 3).contiguous().view(B, seq_len, self.embed_dim)

        elif self.mode == 'branch2':
            # Q[t,n] attends to K[t,:] CEUS-to-BUS
            q_reshaped = q.view(B, T, HW, self.embed_dim)
            k_reshaped = k.view(B, T, HW, self.embed_dim)
            v_reshaped = v.view(B, T, HW, self.embed_dim)

            # Flatten for attention: (B*T, HW, embed_dim)
            q_flat = q_reshaped.view(B * T, HW, self.embed_dim)
            k_flat = k_reshaped.view(B * T, HW, self.embed_dim)
            v_flat = v_reshaped.view(B * T, HW, self.embed_dim)

            # Projections
            q_p = self.q_proj(q_flat).view(B * T, HW, self.num_heads, self.head_dim).transpose(1, 2)
            k_p = self.k_proj(k_flat).view(B * T, HW, self.num_heads, self.head_dim).transpose(1, 2)
            v_p = self.v_proj(v_flat).view(B * T, HW, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention
            scores = torch.matmul(q_p, k_p.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, v_p)

            # Reshape back
            context = context.transpose(1, 2).contiguous().view(B * T, HW, self.embed_dim)
            output = self.out_proj(context)
            output = output.view(B, T, HW, self.embed_dim).view(B, seq_len, self.embed_dim)

        return output

class ACMST_TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mode, T=16, HW=49, ff_dim=2048, dropout=0.1):
        super(ACMST_TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = AsymmetricCrossAttention(embed_dim, num_heads, mode, T, HW)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cross_k, cross_v):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Cross-attention
        cross_out = self.cross_attn(x, cross_k, cross_v)
        x = self.norm2(x + cross_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x

class DualStreamNetwork(nn.Module):
    def __init__(self, embed_dim=384, num_layers=6, num_heads=6, T=16, HW=49):
        super(DualStreamNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.T = T
        self.HW = HW

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, T * HW, embed_dim))

        # Two branches
        self.branch1 = nn.ModuleList([ACMST_TransformerBlock(embed_dim, num_heads, mode='branch1', T=T, HW=HW) for _ in range(num_layers)])
        self.branch2 = nn.ModuleList([ACMST_TransformerBlock(embed_dim, num_heads, mode='branch2', T=T, HW=HW) for _ in range(num_layers)])

    def forward(self, feat1, feat2):
        # feat1, feat2: (B, embed_dim, T, HW)
        B, embed_dim, T, HW = feat1.shape
        seq_len = T * HW

        # Reshape to (B, seq_len, embed_dim)
        feat1 = feat1.view(B, embed_dim, seq_len).transpose(1, 2)  # (B, seq_len, embed_dim)
        feat2 = feat2.view(B, embed_dim, seq_len).transpose(1, 2)

        # Add positional encoding
        feat1 = feat1 + self.pos_embed
        feat2 = feat2 + self.pos_embed

        # Process through layers
        for i in range(self.num_layers):
            feat1 = self.branch1[i](feat1, feat2, feat2)
            feat2 = self.branch2[i](feat2, feat1, feat1)

        # Average the two branches
        output = (feat1 + feat2) / 2  # (B, seq_len, embed_dim)

        # Global average pooling
        output = output.mean(dim=1)  # (B, embed_dim)

        return output


class AsyCMST(nn.Module):
    def __init__(self, embed_dim=384, num_classes=2, num_layers=6, num_heads=6, sample_duration=16, sample_size=224):
        super(AsyCMST, self).__init__()
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.T = sample_duration
        self.H = self.W = sample_size
        self.HW = (self.H // 32) ** 2  # Assuming ResNet18 downsampling to 7x7 for 224x224
        self.frame_encoder = FrameFeatureEncoding(embed_dim)
        self.dual_stream = DualStreamNetwork(embed_dim, num_layers, num_heads, self.T, self.HW)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x1, x2):
        feat1 = self.frame_encoder(x1) # BUS
        feat2 = self.frame_encoder(x2) # CEUS
        features = self.dual_stream(feat1, feat2)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    # Test
    from datetime import datetime
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AsyCMST(embed_dim=384, num_classes=2, sample_duration=16, sample_size=224).to(device)
    for i in range(1000):
        start_time = datetime.now()       
        batch_size = 16
        x1 = torch.randn(batch_size, 3, 16, 224, 224).to(device)
        x2 = torch.randn(batch_size, 3, 16, 224, 224).to(device)
        output = model(x1, x2)
        end_time = datetime.now()
        # time_ms = float((end_time - start_time).seconds) * 1000 / batch_size
        time_ms = (end_time - start_time).total_seconds() * 1000 / batch_size

        print(output.shape, time_ms)  # Should be (batch_size, num_classes)
