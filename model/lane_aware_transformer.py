import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class LearnableRelPos2D(nn.Module):
    def __init__(self, num_heads, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.num_heads = num_heads
        self.rel_h = nn.Parameter(torch.zeros((2 * height - 1, num_heads)))
        self.rel_w = nn.Parameter(torch.zeros((2 * width - 1, num_heads)))
        nn.init.trunc_normal_(self.rel_h, std=0.02)
        nn.init.trunc_normal_(self.rel_w, std=0.02)

    def forward(self, H, W):
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, H, W)
        coords_flatten = coords.reshape(2, -1)  # (2, H*W)

        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, HW, HW)
        rel_coords[0] += self.height - 1
        rel_coords[1] += self.width - 1

        rh = self.rel_h[rel_coords[0]]  # (HW, HW, heads)
        rw = self.rel_w[rel_coords[1]]
        bias = rh + rw  # (HW, HW, heads)
        return bias.permute(2, 0, 1)  # (heads, HW, HW)


class LaneAwareAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Default H=W=20, for 640x640 → 20x20 feature map
        self.rel_pos_bias = LearnableRelPos2D(num_heads=num_heads, height=20, width=20)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, C//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        rel_bias = self.rel_pos_bias(H, W)
        attn = attn + rel_bias.unsqueeze(0)  # broadcast over batch
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class LaneAwareTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LaneAwareAttention(dim, num_heads=num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)

        x = x + self.mlp(self.norm2(x))
        return x


class LaneAwareTransformerEncoder(nn.Module):
    def __init__(self, dim, depth=4, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            LaneAwareTransformerBlock(dim, num_heads=num_heads)
            for _ in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class ConvHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)
    
    
class LaneGraphFormerPlusPlus(nn.Module):
    def __init__(self, img_size=640, pe=True):
        super().__init__()
        self.feature_dim = 256
        self.spatial_dim = img_size // 32

        self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.reduce_dim = nn.Conv2d(1024, self.feature_dim, kernel_size=1)

        # Transformer with Learnable 2D Relative Position Bias
        self.transformer_encoder = LaneAwareTransformerEncoder(dim=self.feature_dim, depth=4, num_heads=8)

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Conv2d(self.feature_dim * 2, self.feature_dim, kernel_size=1),
            nn.ReLU()
        )

        # Key Outputs
        self.node_heatmap = ConvHead(self.feature_dim, 6)      # 6 types of keypoints
        self.node_type = ConvHead(self.feature_dim, 2)         # way or link
        self.edge_vector = ConvHead(self.feature_dim, 4)       # dx1, dy1, dx2, dy2
        self.edge_prob = ConvHead(self.feature_dim, 1)         # midpoint confidence
        self.edge_class = ConvHead(self.feature_dim, 3)        # way, link, null
        self.topo_tensor = ConvHead(self.feature_dim, 7)       # pr, pv, pm, dx1, dy1, dx2, dy2
        self.segmentation = ConvHead(self.feature_dim, 2)      # lane vs background

    def forward(self, img):
        feat = self.reduce_dim(self.backbone.features(img))
        trans_feat = self.transformer_encoder(feat)

        fused = self.fusion(torch.cat([feat, trans_feat], dim=1))

        return {
            'node_heatmap': torch.sigmoid(self.node_heatmap(fused)),
            'node_type': self.node_type(fused),
            'edge_vector': self.edge_vector(fused),
            'edge_prob': torch.sigmoid(self.edge_prob(fused)),
            'edge_class': self.edge_class(fused),
            'topo_tensor': self.topo_tensor(fused),
            'segmentation': self.segmentation(fused)
        }