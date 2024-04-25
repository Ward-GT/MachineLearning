import math
import torch
from typing import Optional, Tuple, Union, List
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1))

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        else:
            self.skip = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, dim_head: int = None, n_groups: int = 32):
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

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attention: bool):
        super().__init__()

        self.res = ResidualBlock(in_channels, out_channels, time_channels)

        if has_attention:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res(x, t)
        return self.attn(h)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attention: bool):
        super().__init__()

        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)

        if has_attention:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res(x, t)
        return self.attn(h)

class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res1(x, t)
        h = self.attn(h)
        return self.res2(h, t)

class DownSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.down = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.up(x)

class UNet(nn.Module):
    def __init__(self, input_channels: int = 6, output_channels: int = 3, n_channels: int = 64, ch_mults: List[int] = (1, 2, 2, 4), is_attn: List[bool] = (False, False, True, True), n_blocks: int = 2):
        super().__init__()

        self.image_proj = nn.Conv2d(input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = TimeEmbedding(n_channels*4)
        out_channels = in_channels = n_channels
        n_resolutions = len(ch_mults)
        self.down_blocks = nn.ModuleList()

        for i in range(n_resolutions):

            out_channels = in_channels * ch_mults[i]

            for _ in range(n_blocks):
                self.down_blocks.append(DownBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
                in_channels = out_channels

            if i < n_resolutions - 1:
                self.down_blocks.append(DownSample(in_channels))

        self.middle = MiddleBlock(out_channels, n_channels*4)

        self.up_blocks = nn.ModuleList()

        in_channels = out_channels

        for i in reversed(range(n_resolutions)):

            out_channels = in_channels

            for _ in range(n_blocks):
                self.up_blocks.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))

            out_channels = in_channels // ch_mults[i]
            self.up_blocks.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            in_channels = out_channels

            if i > 0:
                self.up_blocks.append(UpSample(in_channels))

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.out = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)
        h = [x]

        for block in self.down_blocks:
            x = block(x, t)
            h.append(x)

        x = self.middle(x, t)
        for block in self.up_blocks:
            if isinstance(block, UpSample):
                x = block(x, t)
            else:
                x = block(torch.cat([x, h.pop()], dim=1), t)

        return self.out(self.act(self.norm(x)))










