import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        # Spatial Attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    # Channel Attention
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        ca = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * ca.unsqueeze(2).unsqueeze(3)

    # Spatial Attention
        avg_out = x.mean(1, keepdim=True)
        max_out = x.amax(1, keepdim=True)
        sa = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * sa

