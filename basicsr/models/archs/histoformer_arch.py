import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math, copy
from inspect import isfunction

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def get_residue(tensor , r_dim = 1):
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel



class DownSample_mask(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim=4, output_dim=1, kernel_size=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.PixelUnshuffle(2),
                                  nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                  )

    def forward(self, x):
        x = self.proj(x)
        return x


class Upsample_mask(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

# ######################################################################交叉注意力模块#############################################################################
# ######################################################################交叉注意力模块#############################################################################
class Attention_Qeury(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_Qeury, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = self.q_dwconv(torch.cat([q, feature], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Attention_Qeury(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_Qeury, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = self.q_dwconv(torch.cat([q, feature], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Attention_Key_Value(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_Key_Value, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

        self.v_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature1, feature2):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        k = self.k_dwconv(torch.cat([k, feature1], dim=1))
        v = self.v_dwconv(torch.cat([v, feature2], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FeedForward_nn(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_nn, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        # print('dim', dim)
        # print('hidden_features', hidden_features)

        self.dwconv1 = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=dim, bias=bias)

        self.project_middle = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x = self.project_in(x)
        x1 = self.dwconv1(x)
        x1 = self.project_middle(x1)
        x1 = F.gelu(x1)
        x2 = self.dwconv2(x1)
        x = self.project_out(x2)
        return x


class TransformerBlock_QKV(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock_QKV, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_Qeury = Attention_Qeury(dim, num_heads, bias)
        self.attn_Key_Value = Attention_Key_Value(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_nn(dim, ffn_expansion_factor, bias)

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature1, feature2, feature3):
        # print('dim',self.dim)
        attn_Qeury = self.attn_Qeury(self.norm1(x), feature1)
        attn_KV = self.attn_Key_Value(self.norm1(x), feature2, feature3)

        x = x + self.fusion(torch.cat([attn_Qeury, attn_KV], dim=1))
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerBlock_Query(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock_Query, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_Qeury = Attention_Qeury(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_nn(dim, ffn_expansion_factor, bias)

    def forward(self, x, feature):
        # print('dim',self.dim)
        x = x + self.attn_Qeury(self.norm1(x), feature)
        x = x + self.ffn(self.norm2(x))

        return x


class Attention_Key_Value(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_Key_Value, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

        self.v_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature1, feature2):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        k = self.k_dwconv(torch.cat([k, feature1], dim=1))
        v = self.v_dwconv(torch.cat([v, feature2], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock_Key_Value(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock_Key_Value, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_Key_Value = Attention_Key_Value(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_nn(dim, ffn_expansion_factor, bias)

    def forward(self, x, feature1, feature2):
        x = x + self.attn_Key_Value(self.norm1(x), feature1, feature2)
        x = x + self.ffn(self.norm2(x))

        return x
#
#
class Global_Perception(nn.Module):
    def __init__(self, dim, dim_pre, num_heads, depth=2, res=(128, 128), pooling_r=4,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True, bias=True,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        #######################################################################################################
        self.global_degregation_aware_restore_aware = global_degregation_aware_restore_aware
        self.global_degregation_aware = global_degregation_aware
        self.global_restore_aware = global_restore_aware

        if self.global_degregation_aware_restore_aware is True:
            self.Attention_qkv = TransformerBlock_QKV(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre, LayerNorm_type='WithBias')
            self.qkv_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 3, kernel_size=1, stride=1),
                                            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim,
                                                      bias=True))
        elif self.global_degregation_aware is True:
            self.Attention_q = TransformerBlock_Query(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre)
            self.q_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 2, kernel_size=1, stride=1),
                                          nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim,
                                                    bias=True))
        elif self.global_restore_aware is True:
            self.Attention_kv = TransformerBlock_Key_Value(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre)
            self.qkv_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 2, kernel_size=1, stride=1),
                                            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim,
                                                      bias=True))

    def forward(self, x, feature_perception=None):
        b, c, h, w = x.size()
        if feature_perception is not None:
            if self.global_degregation_aware_restore_aware is True:
                qkv = self.qkv_dwconv(self.layernorm(feature_perception))
                q_dwconv, k_dwconv, v_dwconv = qkv.chunk(3, dim=1)
                Attention_qkv = self.Attention_qkv(x, feature1=q_dwconv, feature2=k_dwconv,
                                                   feature3=v_dwconv)
                return Attention_qkv
            elif self.global_degregation_aware is True:
                q = self.q_dwconv(self.layernorm(feature_perception))
                Attention_q = self.Attention_q(x, feature=q)
                return Attention_q
            elif self.global_restore_aware is True:
                kv = self.qkv_dwconv(self.layernorm(feature_perception))
                k_dwconv, v_dwconv = kv.chunk(2, dim=1)
                Attention_kv = self.Attention_kv(x, feature1=k_dwconv, feature2=v_dwconv)
                return Attention_kv
        else:
            return x


class Local_Perception(nn.Module):
    def __init__(self, dim, dim_pre, pooling_r=4, local_degregation_aware_restore_aware=True,
                 local_degregation_aware=True,
                 local_restore_aware=True, bias=True):
        super(Local_Perception, self).__init__()
        self.local_degregation_aware_restore_aware = local_degregation_aware_restore_aware
        self.local_degregation_aware = local_degregation_aware
        self.local_restore_aware = local_restore_aware

        if self.local_degregation_aware_restore_aware is True:
            self.degradation = nn.Sequential(
                nn.Conv2d(dim_pre, dim, 1, 1),
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
            ###############################
            self.input = nn.Sequential(
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            )

            self.main_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )

            self.degradation_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )

            self.fusion1 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, bias=bias)
            self.ffn = nn.Sequential(
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            )
        elif self.local_degregation_aware is True:
            self.degradation = nn.Sequential(
                nn.Conv2d(dim_pre, dim, 1, 1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
            ###############################
            self.input = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            )

            self.degradation_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )
            self.ffn = nn.Sequential(
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            )
        elif self.local_restore_aware is True:
            self.degradation = nn.Sequential(
                nn.Conv2d(dim_pre, dim, 1, 1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
            ###############################
            self.input = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            )

            self.main_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )
            self.ffn = nn.Sequential(
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            )
        self.layernorm1 = LayerNorm(dim_pre, LayerNorm_type='WithBias')
        self.layernorm2 = LayerNorm(dim, LayerNorm_type='WithBias')

    def forward(self, x, feature_perception=None):
        b, c, h, w = x.shape
        ############################### Soft_Concert ###############################
        if feature_perception is not None:
            if self.local_degregation_aware_restore_aware is True:
                degradation = self.degradation(self.layernorm1(feature_perception))
                degradation1 = degradation[:, c:, :, :]
                degradation2 = degradation[:, :c, :, :]
                input = self.input(self.layernorm2(x))
                input1 = input[:, c:, :, :]
                input2 = input[:, :c, :, :]
                ###############################
                main_kernel = self.main_kernel(torch.cat([input1, degradation1], dim=1))
                main_kernel = torch.sigmoid(main_kernel)
                main_kernel_mul = torch.mul(main_kernel, degradation1)
                ###############################
                degradation_kernel = self.degradation_kernel(torch.cat([input2, degradation2], dim=1))
                degradation_kernel = torch.sigmoid(degradation_kernel)
                degradation_kernel_mul = torch.mul(degradation_kernel, input2)
                out = self.fusion1(torch.cat([degradation_kernel_mul, main_kernel_mul], dim=1)) + x
                # out = degradation_kernel_mul + main_kernel_mul
                fusion1 = self.ffn(out) + out
                return fusion1
            # self.local_degregation_aware = local_degregation_aware
            # self.local_restore_aware = local_restore_aware

            elif self.local_degregation_aware is True:
                degradation = self.degradation(self.layernorm1(feature_perception))
                input = self.input(self.layernorm2(x))
                ###############################
                degradation_kernel = self.degradation_kernel(torch.cat([input, degradation], dim=1))
                degradation_kernel = torch.sigmoid(degradation_kernel)
                out = torch.mul(degradation_kernel, input) + x
                fusion1 = self.ffn(out) + out
                return fusion1
            elif self.local_restore_aware is True:
                degradation = self.degradation(self.layernorm1(feature_perception))
                input = self.input(self.layernorm2(x))
                main_kernel = self.main_kernel(torch.cat([input, degradation], dim=1))
                main_kernel = F.sigmoid(main_kernel)
                out = torch.mul(main_kernel, degradation) + x
                fusion1 = self.ffn(out) + out
                return fusion1
        else:
            return x


class PromptPGM(nn.Module):
    def __init__(self, dim, num_heads):
        super(PromptPGM, self).__init__()
        self.conv = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1)
        self.G2P = Global_Perception(dim=dim, dim_pre=dim, num_heads=num_heads)
        self.L2P = Local_Perception(dim=dim, dim_pre=dim)

    def forward(self, x, y):
        out1 = self.G2P(x, y)
        out2 = self.L2P(x, y)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv(out) + x
        return out


########################################################################################################################

class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class CSFM(nn.Module):
    def __init__(self, dim, spilt_num=2,scale_ratio=2):
        super(CSFM, self).__init__()
        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')
        self.se = ChannelSELayer(num_channels=dim)
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=False)
        self.act2 = nn.GELU()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.output = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.dim_sp = dim*scale_ratio//spilt_num
        # self.fem = FEM(in_planes=dim,out_planes=dim)
        # self.da = DualAdaptiveNeuralBlock(embed_dim=dim)


        self.mask_in = nn.Sequential(
            nn.Conv2d(1, self.dim_sp, 1),
            nn.GELU()
        )
        self.mask_dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=3 // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.mask_dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp , 1, kernel_size=5, padding=5 // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.mask_out = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.GELU()
        )

    def forward(self, x, mask):
        x = self.norm(x)
        x = self.se(x)

        mask = self.mask_in(mask)
        mask = self.mask_dw_conv_1(mask)
        mask = self.mask_dw_conv_2(mask)

        x_out = x * mask

        x_out = self.output(x_out)
        mask = self.mask_out(mask)
        return x_out,mask
########################################################################################################################
def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.dwcon_q = torch.nn.Conv2d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       groups=in_channels)

        self.kv = torch.nn.Sequential(torch.nn.Linear(1, in_channels // 2),
                                      nn.LayerNorm(in_channels // 2),
                                      nn.LeakyReLU(),
                                      torch.nn.Linear(in_channels // 2, in_channels),
                                      nn.LayerNorm(in_channels),
                                      nn.LeakyReLU(),
                                      torch.nn.Linear(in_channels, in_channels),
                                      nn.LayerNorm(in_channels),
                                      nn.LeakyReLU(),
                                      torch.nn.Linear(in_channels, in_channels * 2),
                                      torch.nn.SiLU()
                                      )

        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, y):  # x: 2 1 1024               y: 2 36 128 128
        b, c, h, w = y.shape
        x = rearrange(x, 'b l d -> b d l ')
        x = self.kv(x)
        k, v = x.chunk(2, dim=-1)  # 2 1024 36
        k = rearrange(k, 'b d l -> b l d')  # 2  36 1024
        # h_=  self.norm(x)
        q = self.dwcon_q(self.q(y))
        q = rearrange(q, 'b c h w -> b (h w) c')  # 2 16384 36

        # compute attention

        w_ = torch.einsum('bij,bjk->bik', q, k)  # 2 116384 1024

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        # q_ = rearrange(q, 'b l d -> b d l')
        # w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', w_, v)
        h_ = rearrange(h_, 'b  (h w) c -> b c h w', h=h, w=w)
        h_ = self.proj_out(h_)

        return y + h_

###################更新后融合模块#############################################################
class ResModule(nn.Module):
    def __init__(self, dim):
        super(ResModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv3(x)
        x1 = self.act(x1)
        x1 = self.conv1(x1)
        out = x1 + x
        return out


class Map_Tensor_Fusion(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Map_Tensor_Fusion, self).__init__()
        self.norm2 = LayerNorm(dim=dim, LayerNorm_type=BiasFree_LayerNorm)
        self.res2 = nn.Sequential(*[ResModule(dim=dim) for i in range(3)])

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax()
        self.dpm = PromptPGM(dim=dim, num_heads=num_heads)

    def forward(self, x, map):

        x2 = self.res2(x)
        x2 = self.norm2(x2)
        x2 = self.dpm(x2, map)
        x2 = self.relu(x2)

        return x2

##########################################################################################################################################
##########################################################################################################################################

class TransformerBlock_map(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_map, self).__init__()

        self.attn_g = Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        # x_out = x + self.dropout(self.ffn(self.norm_ff1(x)))
        x_out = x + self.ffn(self.norm_ff1(x))

        return x_out

##################################################################################################################################################

Conv2d = nn.Conv2d


##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')


def to_3d(x):
    #    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    #    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        #        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5)  # * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def mish(x):
    return x * torch.tanh(F.softplus(x))


##########################################################################
## Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv_5 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1, padding=2,
                               groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated2_1 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1, padding=2,
                                        groups=hidden_features // 4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)

        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)

        return x


##########################################################################
## Dynamic-range Histogram Self-Attention (DHSA)

class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.attn_g = Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        x_out = x + self.ffn(self.norm_ff1(x))

        return x_out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class SkipPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super(SkipPatchEmbed, self).__init__()

        self.proj = nn.Sequential(
            nn.AvgPool2d(2, stride=2, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None),
            Conv2d(in_c, dim, kernel_size=1, bias=bias),
            Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

    def forward(self, x, ):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
class Histoformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):
        super(Histoformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.skip_patch_embed1 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = Conv2d(int(dim * 2 ** 1) + 3, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = Conv2d(int(dim * 2 ** 2) + 3, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = Conv2d(int(dim * 2 ** 3) + 3, int(dim * 2 ** 3), kernel_size=1, bias=bias)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        ###############################################以上代码为新增代码##########################################

        self.feed1 = CSFM(dim =dim)
        self.feed2 = CSFM(dim =dim*2)
        self.feed3 = CSFM(dim =dim*4)


        self.feed4 = CSFM(dim =dim*4)
        self.feed5 = CSFM(dim =dim*2)
        self.feed6 = CSFM(dim =dim*2)


        self.cross_text_image1 = SpatialCrossAttention(in_channels=dim)
        self.cross_text_image2 = SpatialCrossAttention(in_channels=dim * 2)
        self.cross_text_image3 = SpatialCrossAttention(in_channels=dim * 4)

        self.patch_embed2 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_map1 = nn.Sequential(*[
            TransformerBlock_map(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.image_down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_map2 = nn.Sequential(*[
            TransformerBlock_map(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.image_down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_map3 = nn.Sequential(*[
            TransformerBlock_map(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])



        self.dim_up = nn.Conv2d(dim,dim*2,1)

        self.fusion1 = Map_Tensor_Fusion(dim=dim, num_heads=heads[0], bias=bias)
        self.fusion2 = Map_Tensor_Fusion(dim=dim * 2, num_heads=heads[1], bias=bias)
        self.fusion3 = Map_Tensor_Fusion(dim=dim * 4, num_heads=heads[2], bias=bias)

        self.fusion4 = Map_Tensor_Fusion(dim=dim * 4, num_heads=heads[2], bias=bias)
        self.fusion5 = Map_Tensor_Fusion(dim=dim * 2, num_heads=heads[1], bias=bias)
        self.fusion6 = Map_Tensor_Fusion(dim=dim * 2, num_heads=heads[1], bias=bias)


        self.down_rcp1 = DownSample_mask()
        self.down_rcp2 = DownSample_mask()


        self.up_rcp1 = Upsample_mask()
        self.up_rcp2 = Upsample_mask()
    ##########################################以上代码为新增代码##################################################

    def forward(self, inp_img, text_tensor):
        B, C, H, W = inp_img.shape
        text_tensor = text_tensor.float()
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # c,h,w

        ################################-----begin
        mask = get_residue(inp_img)
        image_tensor = self.patch_embed2(inp_img)
        image_tensor1 = self.encoder_map1(image_tensor)
        map1 = self.cross_text_image1(text_tensor, image_tensor1)
        # map_fusion1 = self.fusion1(out_enc_level1, map1)
        # map_fusion1,mask = self.feed1(map_fusion1,mask)
        # out_enc_level1 = out_enc_level1 + map_fusion1
        ################################-----end

        inp_enc_level2 = self.down1_2(out_enc_level1)  # 2c, h/2, w/2
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(torch.cat([inp_enc_level2, skip_enc_level1], 1))
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        ################################-----begin\
        mask = self.down_rcp1(mask)
        image_tensor2 = self.image_down1_2(image_tensor1)
        image_tensor2 = self.encoder_map2(image_tensor2)
        map2 = self.cross_text_image2(text_tensor, image_tensor2)
        # map_fusion2 = self.fusion2(out_enc_level2, map2)
        # map_fusion2,mask= self.feed2(map_fusion2,mask)
        # out_enc_level2 = out_enc_level2 + map_fusion2
        ################################-----end

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(torch.cat([inp_enc_level3, skip_enc_level2], 1))
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        ################################-----begin
        mask = self.down_rcp2(mask)
        image_tensor3 = self.image_down2_3(image_tensor2)
        image_tensor3 = self.encoder_map3(image_tensor3)
        map3 = self.cross_text_image3(text_tensor, image_tensor3)
        # map_fusion3 = self.fusion3(out_enc_level3, map3)
        # map_fusion3,mask = self.feed3(map_fusion3,mask)
        # out_enc_level3 = out_enc_level3 + map_fusion3
        ################################-----end

        inp_enc_level4 = self.down3_4(out_enc_level3)
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(torch.cat([inp_enc_level4, skip_enc_level3], 1))

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)                                              # 4c h/4 w/4
        # ################################-----begin

        map_fusion4 = self.fusion3(out_dec_level3, map3)
        map_fusion4,mask = self.feed4(map_fusion4,mask)
        out_dec_level3 = out_dec_level3 + map_fusion4
        # ################################-----end

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        # ################################-----begin
        mask = self.up_rcp1(mask)
        map_fusion5 = self.fusion5(out_dec_level2, map2)
        map_fusion5,mask= self.feed5(map_fusion5,mask)
        out_dec_level2 = out_dec_level2 + map_fusion5
        # ################################-----end

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        # ################################-----begin
        map1 = self.dim_up(map1)
        mask = self.up_rcp2(mask)
        map_fusion6 = self.fusion6(out_dec_level1, map1)
        map_fusion6,mask= self.feed6(map_fusion6,mask)
        out_dec_level1 = out_dec_level1 + map_fusion6
        # ################################-----end

        out_dec_level1 = self.refinement(out_dec_level1)

        ###########################
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1 + inp_img
