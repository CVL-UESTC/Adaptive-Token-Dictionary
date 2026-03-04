'''
An official Pytorch impl of `Transcending the Limit of Local Window: 
Advanced Super-Resolution Transformer with Adaptive Token Dictionary`.

Arxiv: 'https://arxiv.org/abs/2401.08209'
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import numbers

from basicsr.utils.registry import ARCH_REGISTRY


# Shuffle operation for Categorization and UnCategorization operations.
def index_reverse(index):
    index_r = torch.zeros_like(index)
    
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    # match the shape of x and index
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features, bias=False), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN_td(nn.Module):
    def __init__(self, in_features, hidden_features=None, td_features=0, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        hidden_features += td_features
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x, x_td, x_size):
        x = self.fc1(x)
        x = torch.cat([self.act(x), x_td], dim=-1)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x
    
    
class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, bias=False):
        
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.bias = bias
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim, bias=bias)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, mask=None):
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index_SA.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[0] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, bias={self.bias}'

    def flops(self, n):
        flops = 0
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class ATD_CA(nn.Module):
    def __init__(self, dim, num_tokens=64, reducted_dim=10, bias=False):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.dr = reducted_dim
        self.bias = bias

        self.wq = nn.Linear(dim, reducted_dim, bias=bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=bias)
        self.wv = nn.Linear(dim, dim, bias=bias)

        self.scale = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td, x_size):
        h, w = x_size
        b, n, c = x.shape
        b, m, c = td.shape
        
        # Q: b, n, c
        q = self.wq(x)
        # K: b, m, c
        k = self.wk(td)
        # V: b, m, c
        v = self.wv(td)

        # Q @ K^T
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # b, n, m
        scale = 1 + torch.clamp(self.scale, 0, 3) * np.log(self.num_tokens)
        attn = attn * scale
        attn = self.softmax(attn)
        
        # Attn * V
        x = (attn @ v).reshape(b, n, c)

        return x, attn

    def flops(self, n):
        flops = 0

        # qkv = self.wq(x)
        flops += n * self.dim * self.dr
        # k = self.wk(td)
        flops += self.num_tokens * self.dim * self.dr
        # v = self.wv(td)
        flops += self.num_tokens * self.dim * self.dim

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.dr

        # x = (attn @ v)
        flops += n * self.num_tokens * self.dim

        return flops
    

class AC_MSA(nn.Module):
    def __init__(self, dim, num_heads=4, category_size=128, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.category_size = category_size
        self.proj = nn.Linear(dim, dim, bias=bias)

        self.scale = (dim // self.num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, tk_id, x_size):
        b, n, c3 = qkv.shape
        c = c3 // 3
        gs = min(n, self.category_size)  # group size
        ng = (n + gs - 1) // gs
        
        # sort features by type
        x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1, stable=False)
        tk_id_inv = index_reverse(x_sort_indices)

        # feature categorization
        shuffled_qkv = feature_shuffle(qkv, x_sort_indices)  # b, n, c3
        pad_n = ng * gs - n
        paded_qkv = torch.cat((shuffled_qkv, torch.flip(shuffled_qkv[:, n-pad_n:n, :], dims=[1])), dim=1)
        y = paded_qkv.reshape(b, -1, gs, c3)  # b, ng, gs, c*3

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c//self.num_heads).permute(3, 0, 1, 4, 2, 5)  # 3, b, ng, nh, gs, c//nh
        q, k, v = qkv[0], qkv[1], qkv[2]    # b, ng, nh, gs, c//nh

        attn = (q @ k.transpose(-2, -1))  # b, ng, nh, gs, gs
        attn = attn * self.scale
        attn = self.softmax(attn)  # b, ng, nh, gs, gs

        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n+pad_n, c)[:, :n, :]
        x = feature_shuffle(y, tk_id_inv)
        x = self.proj(x)

        return x


    def flops(self, n):
        flops = 0

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.category_size

        # y = (attn @ v)
        flops += n * self.dim * self.category_size

        # x = self.proj(x)
        flops += n * self.dim * self.dim

        return flops


class ATDTransformerLayer(nn.Module):
    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 bias=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens=num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.reducted_dim = reducted_dim
        self.dim_ffn_td = dim_ffn_td

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3*dim, bias=bias)
        
        self.attn_win = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            bias=bias,
        )
        self.attn_atd = ATD_CA(
            self.dim,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            bias=bias,
        )
        self.attn_aca = AC_MSA(
            self.dim,
            num_heads=num_heads,
            category_size=category_size,
            bias=bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc_td = nn.Linear(dim, dim_ffn_td, bias=bias)
        self.convffn = ConvFFN_td(in_features=dim, hidden_features=mlp_hidden_dim, td_features=dim_ffn_td, kernel_size=convffn_kernel_size, act_layer=act_layer)


    def forward(self, x, td, x_size):
        h, w = x_size
        b, n, c = x.shape
        b, m, c = td.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        # ATD_CA
        x_atd, sim_atd = self.attn_atd(x, td, x_size)  # x_atd: (b, n, c)  sim_atd: (b, n, m)  td: (b, m, c)

        # AC_MSA
        tk_id = torch.argmax(sim_atd, dim=-1, keepdim=False)
        x_aca = self.attn_aca(qkv, tk_id, x_size) 
        x_td = torch.gather(self.fc_td(td), dim=1, index=tk_id.reshape(b, n, 1).expand(-1, -1, self.dim_ffn_td))  # b, n, c

        # SW-MSA
        qkv = qkv.reshape(b, h, w, c3)

        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.calculate_mask(x_size).to(x.device)
        else:
            shifted_qkv = qkv
            attn_mask = None

        x_windows = window_partition(shifted_qkv, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)  # nw*b, window_size*window_size, c
        attn_windows = self.attn_win(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x

        x = shortcut + x_atd + x_win.view(b, n, c) + x_aca

        # FFN
        x = x + self.convffn(self.norm2(x), x_td, x_size)

        return x

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                
        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        # qkv = self.wqkv(x)
        flops += self.dim * 3 * self.dim * h * w

        # SWMSA, ATDCA, ACMSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)
        flops += self.attn_atd.flops(h * w)
        flops += self.attn_aca.flops(h * w)

        # mlp
        flops += h * w * self.dim * (self.dim*2 + self.dim_ffn_td) * self.mlp_ratio
        flops += h * w * (self.dim + self.dim_ffn_td) * self.convffn_kernel_size**2 * self.mlp_ratio

        return flops



class BasicBlock(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 convffn_kernel_size,
                 reducted_dim,
                 mlp_ratio=4.,
                 bias=False,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                ATDTransformerLayer(
                    dim=dim,
                    idx=i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    dim_ffn_td=dim_ffn_td,
                    category_size=category_size,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    reducted_dim=reducted_dim, 
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    norm_layer=norm_layer,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # Token Dictionary
        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def forward(self, x, x_size):
        b, n, c = x.shape
        td = self.td.expand([b, -1, -1])
        idx = 0
        idx_checkpoint = 7
        for layer in self.layers:
            if self.use_checkpoint and self.idx < idx_checkpoint:
                x = checkpoint(layer, x, td, x_size, use_reentrant=False)
            else:
                x = layer(x, td, x_size)
            idx += 1
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class ATDB(nn.Module):
    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 bias=False,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 ):
        super(ATDB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size

        self.block = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            dim_ffn_td=dim_ffn_td,
            category_size=category_size,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            bias=bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x, x_size):
        return self.block(x, x_size)

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.block.flops(input_resolution)

        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.down = nn.Sequential(
                    nn.Conv2d(dim, dim//2, 3, 1, 1, bias=False),
                    nn.PixelUnshuffle(2)
                    )
    
    def forward(self, x, x_size):
        h, w = x_size
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.down(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    
    def flops(self, input_resolution):
        h, w = input_resolution
        flops = h * w * self.dim * self.dim // 2 * 9
        return flops

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.up = nn.Sequential(
                    nn.Conv2d(dim, dim*2, 3, 1, 1, bias=False),
                    nn.PixelShuffle(2)
                    )
    
    def forward(self, x, x_size):
        h, w = x_size
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.up(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    
    def flops(self, input_resolution):
        h, w = input_resolution
        flops = h * w * self.dim * self.dim * 2 * 9
        return flops



@ARCH_REGISTRY.register()
class ATDUNet(nn.Module):
    def __init__(self,
                 img_size=512,
                 patch_size=1,
                 in_chans=1,
                 out_chans=None,
                 embed_dim=40,
                 depths=(6, 8, 8, 6),
                 num_heads=(1, 2, 4, 8),
                 window_size=(16, 16, 16, 8),
                 dim_ffn_td=(4, 8, 16, 16),
                 category_size=(128, 128, 128, 128),
                 num_tokens=(256, 256, 256, 256),
                 reducted_dim=(4, 8, 16, 16),
                 convffn_kernel_size=5,
                 mlp_ratio=4,
                 bias=False,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if out_chans is None:
            out_chans = in_chans
        self.out_chans = out_chans
        self.window_size = window_size
        self.patch_norm = patch_norm
        self.use_checkpoint = use_checkpoint

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.refine_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.reduces = nn.ModuleList()

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1, bias=bias)

        chan = embed_dim
        for idx in range(len(depths) - 1):
            self.encoders.append(
                ATDB(
                    dim=chan,
                    idx=0,
                    input_resolution=(img_size, img_size),
                    depth=depths[idx], 
                    num_heads=num_heads[idx],
                    window_size=window_size[idx],
                    dim_ffn_td=dim_ffn_td[idx],
                    category_size=category_size[idx],
                    num_tokens=num_tokens[idx],
                    reducted_dim=reducted_dim[idx],
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,          
                )
            )
            self.downs.append(Downsample(chan))
            chan = chan * 2

        idx = -1
        self.middle_blks = \
            ATDB(
                dim=chan,
                idx=0,
                input_resolution=(img_size, img_size),
                depth=depths[idx], 
                num_heads=num_heads[idx],
                window_size=window_size[idx],
                dim_ffn_td=dim_ffn_td[idx],
                category_size=category_size[idx],
                num_tokens=num_tokens[idx],
                reducted_dim=reducted_dim[idx],
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                bias=bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,         
            )

        for idx in range(len(depths) - 2, -1, -1):
            self.ups.append(Upsample(chan))
            if idx > 0:
                chan = chan // 2
                self.reduces.append(nn.Linear(chan * 2, chan, bias=bias))
            else:
                self.reduces.append(nn.Identity())
            self.decoders.append(
                ATDB(
                    dim=chan,
                    idx=0,
                    input_resolution=(img_size, img_size),
                    depth=depths[idx], 
                    num_heads=num_heads[idx],
                    window_size=window_size[idx],
                    dim_ffn_td=dim_ffn_td[idx],
                    category_size=category_size[idx],
                    num_tokens=num_tokens[idx],
                    reducted_dim=reducted_dim[idx],
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,      
                )
            )

        idx = 0
        self.refine_blks = \
            ATDB(
                dim=chan,
                idx=0,
                input_resolution=(img_size, img_size),
                depth=depths[idx], 
                num_heads=num_heads[idx]*2,
                window_size=window_size[idx],
                dim_ffn_td=dim_ffn_td[idx],
                category_size=category_size[idx],
                num_tokens=num_tokens[idx],
                reducted_dim=reducted_dim[idx],
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                bias=bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,       
            )

        self.padder_size = window_size[-1] * 2**(len(depths)-1)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=chan,
            embed_dim=chan,
            norm_layer=norm_layer if self.patch_norm else None)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(chan, chan // 2, 3, 1, 1, bias=bias)
        self.conv_last = nn.Conv2d(chan // 2, out_chans, 3, 1, 1, bias=bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        b, c, h, w = x.shape
        x = self.patch_embed(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, (h, w))
            encs.append(x)
            x = down(x, (h, w))
            h, w = h//2, w//2

        x = self.middle_blks(x, (h, w))

        for decoder, up, reduce, enc_skip in zip(self.decoders, self.ups, self.reduces, encs[::-1]):
            x = up(x, (h, w))
            h, w = h*2, w*2
            x = torch.cat([x, enc_skip], -1)
            x = reduce(x)
            x = decoder(x, (h, w))

        x = self.refine_blks(x, (h, w))

        x = self.patch_unembed(x, (h, w))

        return x

    def forward(self, x):
        # print(x.shape)
        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        h_pad = ((h_ori + self.padder_size - 1) // self.padder_size) * self.padder_size - h_ori
        w_pad = ((w_ori + self.padder_size - 1) // self.padder_size) * self.padder_size - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]


        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        # res = self.forward_features(x_first, params)
        x = x + self.conv_last(res)

        # unpadding
        x = x[..., :h_ori, :w_ori]

        return x

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution

        flops += h * w * self.in_chans * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)

        for encoder, down in zip(self.encoders, self.downs):
            flops += encoder.flops((h, w))
            flops += down.flops((h, w))
            h, w = h//2, w//2

        flops += self.middle_blks.flops((h, w))

        for decoder, up in zip(self.decoders, self.ups):
            flops += up.flops((h, w))
            h, w = h*2, w*2
            flops += decoder.flops((h, w))
        
        flops += self.refine_blks.flops((h, w))

        flops += h * w * self.embed_dim * self.embed_dim * 9
        flops += h * w * self.out_chans * self.embed_dim * 9

        return flops


if __name__ == '__main__':

    model = ATDU().cuda()

    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    print(128, 128, model.flops([128, 128]) / 1e9, 'G')
    # print(256, 256, model.flops([256, 256]) / 1e9, 'G')
    # # print(180, 320, model.flops([720//4, 1280//4]) / 1e9, 'G')
    # print(model)

    # Test
    # _input = torch.randn([1, 1, 128, 128]).cuda()
    # output = model(_input)
    # print(output.shape)

