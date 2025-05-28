import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DFNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes=8):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class LightTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return x

class HybridCFDModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, velocity_scale=5.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )
        self.velocity_scale = velocity_scale
    def forward(self, x):
        return self.net(x) * self.velocity_scale
