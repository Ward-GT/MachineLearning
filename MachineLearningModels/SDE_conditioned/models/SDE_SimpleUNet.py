import torch
import math
import torch.nn as nn
from typing import Optional, Tuple, Union, List
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, has_attention, n_heads, dim_head, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 4, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.act = Swish()
        if has_attention:
            self.attn = AttentionBlock(n_channels=out_ch, n_heads=n_heads, dim_head=dim_head)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.act(self.conv1(x)))
        # Time embedding
        time_emb = self.act(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.act(self.conv2(h)))
        # Attention
        h = self.attn(h)
        # Down or Upsample
        return self.transform(h)

class TimeEmbedding(nn.Module):
    def __init__(self, time_channels: int):
        super().__init__()

        self.time_channels = time_channels
        self.lin1 = nn.Linear(self.time_channels // 4, self.time_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.time_channels, self.time_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.time_channels // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 4, dim_head: int = None, n_groups: int = 32):
        super().__init__()

        if dim_head is None:
            dim_head = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * dim_head * 3)
        self.output = nn.Linear(n_heads * dim_head, n_channels)
        self.scale = dim_head ** -0.5
        self.n_heads = n_heads
        self.dim_head = dim_head

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width = x.shape

        x = x.reshape(batch_size, n_channels, height*width).permute(0, 2, 1)

        qkv = self.projection(x).reshape(batch_size, height*width, self.n_heads, self.dim_head*3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)

        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(batch_size, -1, self.n_heads*self.dim_head)
        res = self.output(res)
        res += x
        return res.permute(0, 2, 1).reshape(batch_size, n_channels, height, width)

class SimpleUNet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, n_channels, image_channels=6, out_dim=3, is_attn=[False, False, True, True], n_heads=4, dim_head=None):
        super().__init__()
        image_channels = image_channels
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = out_dim
        time_emb_dim = n_channels*4

        self.time_emb = TimeEmbedding(time_emb_dim)

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim, has_attention=is_attn[i], n_heads=n_heads, dim_head=dim_head) for i in range(len(down_channels) - 1)])
        # Upsample
        is_attn.reverse()
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True, has_attention=is_attn[i], n_heads=n_heads, dim_head=dim_head) for i in range(len(up_channels) - 1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x_t, y, t):
        x = torch.cat((x_t, y), dim=1)
        # Embedd time
        t = self.time_emb(t)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)