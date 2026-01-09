import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SelfAttention2D(nn.Module):
    """Lightweight spatial self-attention for CNN feature maps."""
    def __init__(self, channels: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Conv2d(channels, inner, 1, bias=False)
        self.to_k = nn.Conv2d(channels, inner, 1, bias=False)
        self.to_v = nn.Conv2d(channels, inner, 1, bias=False)
        self.proj = nn.Conv2d(inner, channels, 1, bias=False)
        self.scale = dim_head ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.to_q(x).view(b, self.heads, -1, h*w)
        k = self.to_k(x).view(b, self.heads, -1, h*w)
        v = self.to_v(x).view(b, self.heads, -1, h*w)
        attn = torch.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)  # [B,H,HW,HW]
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, -1, h, w)
        return self.proj(out) + x

class QualityNet(nn.Module):
    """Predict a per-sample reliability (0..1) from an embedding vector."""
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d//2),
            nn.ReLU(inplace=True),
            nn.Linear(d//2, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ResNetBackbone(nn.Module):
    def __init__(self, out_dim: int = 256, use_pretrained: bool = False):
        super().__init__()
        w = ResNet18_Weights.DEFAULT if use_pretrained else None
        base = resnet18(weights=w)
        self.features = nn.Sequential(*list(base.children())[:-2])  # up to conv5
        self.num_ch = base.fc.in_features  # 512 for resnet18
        self.attn = SelfAttention2D(self.num_ch, heads=4, dim_head=32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self.num_ch, out_dim)
        self.quality = QualityNet(out_dim)

    def forward(self, x):
        fmap = self.features(x)
        fmap = self.attn(fmap)
        emb = self.pool(fmap).flatten(1)
        emb = self.proj(emb)
        return emb

    def quality_score(self, emb):
        return self.quality(emb)
