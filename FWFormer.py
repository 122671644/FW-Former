import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from functools import partial
from typing import Tuple, Union
# from Visualizer.visualizer import get_local
# get_local.activate()

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size = (128, 128)):
        B, L, C = x.shape
        H, W = img_size[0], img_size[1]
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops



#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size=(128, 128)):
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops



# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size=(128, 128)):
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops

#########################################
# Retention
def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class VisionRetentionChunk(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        #
        # qr_w = qr.transpose(1, 2)  # (b h n w d1)
        # kr_w = kr.transpose(1, 2)  # (b h n w d1)
        qr_w = q.transpose(1, 2)  # (b h n w d1)
        kr_w = k.transpose(1, 2)  # (b h n w d1)

        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b h n w w)
        qk_mat_w = qk_mat_w + mask_w  # (b h n w w)

        qk_mat_w = torch.softmax(qk_mat_w, -1)  # (b h n w w)
        v = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        qr_h = q.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = k.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        v = v.permute(0, 3, 2, 1, 4)  # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b w n h h)
        qk_mat_h = qk_mat_h + mask_h  # (b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # (b w n h h)
        output = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w n*d2)

        output = output + lepe

        output = self.out_proj(output)
        return output


class VisionRetentionAll(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    # @get_local('qk_mat')

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos

        assert h * w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        qr = q.flatten(2, 3)  # (b n l d1)
        kr = k.flatten(2, 3)  # (b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d2)
        vr = vr.flatten(2, 3)  # (b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2)  # (b n l l)
        qk_mat = qk_mat + mask  # (b n l l)

        qk_mat = torch.softmax(qk_mat, -1)  # (b n l l)
        output = torch.matmul(qk_mat, vr)  # (b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output


##########################################################################
## FFN
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class PModule(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1)
        #self.selayer = SELayer(hidden_dim//2)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim//2, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128, 128)):
        # bs x hw x c
        hh,ww = img_size[0],img_size[1]

        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b h w c -> b c h w ', h=hh, w=ww)

        x1,x2 = self.dwconv(x).chunk(2, dim=1)
        x3 = x1 * x2
        #x4=self.selayer(x3)
        # flaten
        x3 = rearrange(x3, ' b c h w -> b (h w) c', h=hh, w=ww)
        y = self.linear2(x3)
        y = rearrange(y, ' b (h w) c -> b h w c', h=hh, w=ww)
        return y

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        flops += H * W * self.hidden_dim//2
        # fc2
        flops += H * W * self.hidden_dim//2 * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        return flops

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128, 128)):
        # bs x hw x c
        bs, hw, c = x.size()
        # hh = int(math.sqrt(hw))
        hh = img_size[0]
        ww = img_size[1]

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=ww)

        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        return flops

#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.in_features * self.hidden_features
        # fc2
        flops += H * W * self.hidden_features * self.out_features
        print("MLP:{%.2f}" % (flops / 1e9))
        return flops


class MSDWConv(nn.Module):

    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.dw_sizes = dw_sizes
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(dw_sizes)):
            if i == 0:
                channels = dim - dim // len(dw_sizes) * (len(dw_sizes) - 1)
            else:
                channels = dim // len(dw_sizes)
            conv = nn.Conv2d(channels, channels, kernel_size=dw_sizes[i], padding=dw_sizes[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x



class MSConvStar(nn.Module):

    def __init__(self, dim, mlp_ratio=2., dw_sizes=[3]):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = MSDWConv(dim=hidden_dim, dw_sizes=dw_sizes)
        self.fc2 = nn.Conv2d(hidden_dim // 2, dim, 1)
        self.num_head = len(dw_sizes)
        self.act = nn.GELU()

        assert hidden_dim // self.num_head % 2 == 0

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x



class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False,
                 layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:  # 'whole'
            self.retention = VisionRetentionAll(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)
        # self.Pmodule = PModule(dim=embed_dim, hidden_dim=ffn_dim, act_layer=nn.GELU, drop=0.)
        # self.mlp = Mlp(in_features=embed_dim, hidden_features=ffn_dim, act_layer=nn.GELU,
        #                drop=drop_path)
        self.star = MSConvStar(embed_dim, mlp_ratio=2, dw_sizes=[3])
        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
            xm:torch.Tensor,
            img_size = (128,128),
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
    ):
        x = rearrange(x, ' b (h w) c -> b h w c', h=img_size[0], w=img_size[1])
        x = x + self.pos(x)
        shortcut = x
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.star(self.final_layer_norm(x)))
        x = rearrange(x,' b h w c -> b (h w) c',h=img_size[0],w=img_size[1])
        return x



class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))

        # pb_values = -(2 ** (- self.num_heads / torch.linspace(1, self.num_heads, self.num_heads)))
        # self.register_buffer('pb', pb_values.view(self.num_heads, 1))  # shape: [H, 1]
        #
        # self.ps = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))  # [H, D]

        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
        mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  # (n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def generate_1d_depth_decay(self, H, W, xm):
        """
        generate 1d depth decay mask, the result is l*l
        """
        mask = xm[:, :, :, :, None] - xm[:, :, :, None, :]
        mask = mask.abs()

        new_mask = mask.clone()
        new_mask[mask == 0] = -200
        new_mask[mask == 1] = 0

        new_mask = new_mask  * self.decay[:, None, None, None]

        assert new_mask.shape[2:] == (W, H, H)
        return new_mask

    def generate_2d_depth_decay(self, xm):

        B, _, H, W = xm.shape
        L = H * W  # flattened spatial size
        n = self.decay.shape[0]

        # flatten spatial dimensions
        xm_flat = xm.view(B, L)  # shape: (B, H*W)

        # compute pairwise difference
        diff = xm_flat[:, :, None] - xm_flat[:, None, :]  # shape: (B, L, L)
        diff = diff.abs()  # absolute difference

        # construct decay mask with penalties
        mask = diff.clone()
        mask[diff == 0] = -200  # strong suppression of self
        mask[diff == 1] = 0  # optional: no penalty for close neighbors

        # apply learnable decay: shape (n, B, L, L)
        decay_mask = mask[None, :, :, :] * self.decay[:, None, None, None]

        return decay_mask  # (n, B, H*W, H*W)

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)


            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)


            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


class BasicLayer(nn.Module):

    def __init__(self, embed_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 mlp_ratio=4., drop_path=0., chunkwise_recurrent=False,
                 use_checkpoint=False, layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag, embed_dim, num_heads, embed_dim*mlp_ratio,
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

    def forward(self, x, xm, img_size=(128,128)):
        b,l,c = x.shape

        h = img_size[0]
        w = img_size[1]


        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        for blk in self.blocks:
            if self.use_checkpoint:
                tmp_blk = partial(blk, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
                x = checkpoint.checkpoint(tmp_blk, x)
            else:
                x = blk(x, xm, img_size, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
        return x


class FWFormer(nn.Module):

    def __init__(self, in_chans=3, out_chans=3, embed_dims=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],init_value=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 heads_range=[2, 3, 3, 4, 4, 4, 3, 3, 2], mlp_ratio=4, drop_path_rate=0.1, drop_rate=0., chunkwise_recurrent=[True, True, True, True, True, True, True, True, True],
                 use_checkpoint=False, layerscale=False, layer_init_values=1e-5, dowsample=Downsample, upsample=Upsample,**kwargs):
        super().__init__()

        self.embed_dim = embed_dims
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans+1, out_channel=embed_dims, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dims, out_channel=out_chans, kernel_size=3, stride=1)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # Encoder
        self.encoderlayer_0 = BasicLayer(embed_dim=embed_dims,
                                         depth=depths[0],
                                         num_heads=num_heads[0],
                                         init_value=init_value[0],
                                         heads_range=heads_range[0],
                                         mlp_ratio=mlp_ratio,
                                         drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                         chunkwise_recurrent=chunkwise_recurrent[0],
                                         use_checkpoint=use_checkpoint,
                                         layerscale=layerscale,
                                         layer_init_values=layer_init_values)
        self.dowsample_0 = dowsample(embed_dims, embed_dims * 2)
        self.encoderlayer_1 = BasicLayer(embed_dim=embed_dims * 2,
                                         depth=depths[1],
                                         num_heads=num_heads[1],
                                         init_value=init_value[1],
                                         heads_range=heads_range[1],
                                         mlp_ratio=mlp_ratio,
                                         drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                         chunkwise_recurrent=chunkwise_recurrent[1],
                                         use_checkpoint=use_checkpoint,
                                         layerscale=layerscale,
                                         layer_init_values=layer_init_values)
        self.dowsample_1 = dowsample(embed_dims * 2, embed_dims * 4)
        self.encoderlayer_2 = BasicLayer(embed_dim=embed_dims * 4,
                                         depth=depths[2],
                                         num_heads=num_heads[2],
                                         init_value=init_value[2],
                                         heads_range=heads_range[2],
                                         mlp_ratio=mlp_ratio,
                                         drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                         chunkwise_recurrent=chunkwise_recurrent[2],
                                         use_checkpoint=use_checkpoint,
                                         layerscale=layerscale,
                                         layer_init_values=layer_init_values)
        self.dowsample_2 = dowsample(embed_dims * 4, embed_dims * 8)

        # Bottleneck
        self.conv = BasicLayer(embed_dim=embed_dims * 8,
                                         depth=depths[4],
                                         num_heads=num_heads[4],
                                         init_value=init_value[4],
                                         heads_range=heads_range[4],
                                         mlp_ratio=mlp_ratio,
                                         drop_path=conv_dpr,
                                         chunkwise_recurrent=chunkwise_recurrent[4],
                                         use_checkpoint=use_checkpoint,
                                         layerscale=layerscale,
                                         layer_init_values=layer_init_values)

        # # Decoder
        self.upsample_0 = upsample(embed_dims * 8, embed_dims * 4)
        self.decoderlayer_0 = BasicLayer(embed_dim=embed_dims * 8,
                                         depth=depths[6],
                                         num_heads=num_heads[6],
                                         init_value=init_value[6],
                                         heads_range=heads_range[6],
                                         mlp_ratio=mlp_ratio,
                                         drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                         chunkwise_recurrent=chunkwise_recurrent[6],
                                         use_checkpoint=use_checkpoint,
                                         layerscale=layerscale,
                                         layer_init_values=layer_init_values)
        self.upsample_1 = upsample(embed_dims * 8, embed_dims * 2)
        self.decoderlayer_1 = BasicLayer(embed_dim=embed_dims * 4,
                                         depth=depths[7],
                                         num_heads=num_heads[7],
                                         init_value=init_value[7],
                                         heads_range=heads_range[7],
                                         mlp_ratio=mlp_ratio,
                                         drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                         chunkwise_recurrent=chunkwise_recurrent[7],
                                         use_checkpoint=use_checkpoint,
                                         layerscale=layerscale,
                                         layer_init_values=layer_init_values)
        self.upsample_2 = upsample(embed_dims * 4, embed_dims)
        self.decoderlayer_2 = BasicLayer(embed_dim=embed_dims * 2,
                                         depth=depths[8],
                                         num_heads=num_heads[8],
                                         init_value=init_value[8],
                                         heads_range=heads_range[8],
                                         mlp_ratio=mlp_ratio,
                                         drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                         chunkwise_recurrent=chunkwise_recurrent[8],
                                         use_checkpoint=use_checkpoint,
                                         layerscale=layerscale,
                                         layer_init_values=layer_init_values)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, xm, mask=None):
        # Input  Projection
        xi = torch.cat((x, xm), dim=1)
        self.img_size = (x.shape[2], x.shape[3])
        y = self.input_proj(xi)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y, xm, img_size=self.img_size)

        pool0 = self.dowsample_0(conv0, img_size=self.img_size)
        m = nn.MaxPool2d(2)
        xm1 = m(xm)
        self.img_size = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))
        conv1 = self.encoderlayer_1(pool0, xm1, img_size=self.img_size)
        pool1 = self.dowsample_1(conv1, img_size=self.img_size)
        m = nn.MaxPool2d(2)
        xm2 = m(xm1)
        self.img_size = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))
        conv2 = self.encoderlayer_2(pool1, xm2, img_size=self.img_size)
        pool2 = self.dowsample_2(conv2, img_size=self.img_size)
        self.img_size = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))
        m = nn.MaxPool2d(2)
        xm3 = m(xm2)

        # Bottleneck
        conv3 = self.conv(pool2, xm3, img_size=self.img_size)

        # Decoder
        up0 = self.upsample_0(conv3, img_size=self.img_size)

        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))


        deconv0 = torch.cat([up0, conv2], -1)
        deconv0 = self.decoderlayer_0(deconv0, xm2, img_size=self.img_size)

        up1 = self.upsample_1(deconv0, img_size=self.img_size)
        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))
        deconv1 = torch.cat([up1, conv1], -1)
        deconv1 = self.decoderlayer_1(deconv1, xm1, img_size=self.img_size)

        up2 = self.upsample_2(deconv1, img_size=self.img_size)
        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))
        deconv2 = torch.cat([up2, conv0], -1)
        deconv2 = self.decoderlayer_2(deconv2, xm, img_size=self.img_size)

        # Output Projection
        y = self.output_proj(deconv2, img_size=self.img_size) + x
        return y