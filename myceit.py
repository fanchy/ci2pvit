import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import default_cfgs, _cfg

import mmobj


class Image2Tokens(nn.Module):
    def __init__(self, in_chans=3, out_chans=64, kernel_size=7, stride=2):
        super(Image2Tokens, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        # depthwise
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features
        )
        # pointwise
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        # self.drop = nn.Dropout(drop)

        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        b, n, k = x.size()
        hw = int(math.sqrt(n))
        x = x.transpose(1, 2).view(b, k, hw, hw)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)

        out = x.view(b, k, hw*hw).transpose(1, 2)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attention_map = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # self.attention_map = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionLCA(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(AttentionLCA, self).__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dim = dim
        self.qkv_bias = qkv_bias
        
    def forward(self, x):

        q_weight = self.qkv.weight[:self.dim, :]
        q_bias = None if not self.qkv_bias else self.qkv.bias[:self.dim]
        kv_weight = self.qkv.weight[self.dim:, :]
        kv_bias = None if not self.qkv_bias else self.qkv.bias[self.dim:]
        
        B, N, C = x.shape
        _, last_token = torch.split(x, [N-1, 1], dim=1)
        
        q = F.linear(last_token, q_weight, q_bias)\
             .reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = F.linear(x, kv_weight, kv_bias)\
              .reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # self.attention_map = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, with_bn=True, 
                 feedforward_type='leff', patchHW=8):
        super().__init__()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.feedforward_type = feedforward_type

        if feedforward_type == 'leff':
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.leff = mmobj.LeFF(dim, mlp_hidden_dim, patchHW)
            #self.leff = LocallyEnhancedFeedForward(
            #    in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
            #    kernel_size=kernel_size, with_bn=with_bn,
            #)
        else:  # LCA
            self.attn = AttentionLCA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.feedforward = Mlp(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
            )

    def forward(self, x):
        #print('att', x.shape, self.norm1)
        if self.feedforward_type == 'leff':
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.leff(self.norm2(x)))
            return x
        else:  # LCA
            _, last_token = torch.split(x, [x.size(1)-1, 1], dim=1)
            x = last_token + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.feedforward(self.norm2(x)))
            return x

class CeIT(nn.Module):
    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 leff_local_size=3,
                 leff_with_bn=True, twoscaleIndex = 6, dimEmbed=192):
        super().__init__()
        print('dimEmbed', dimEmbed)

        self.i2t = nn.Identity()
        if twoscaleIndex >= 0:
            num_patches = (img_size//16)**2#self.i2t.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dimEmbed))
        else:
            num_patches = (img_size//32)**2#self.i2t.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()

        for i in range(depth):
            n = None
            curDim = embed_dim
            patchHW = img_size//32
            curNumHead = num_heads
            if twoscaleIndex >= 0:
                if i < twoscaleIndex:
                    curDim = dimEmbed
                    curNumHead = dimEmbed//(embed_dim//num_heads)
                    patchHW = img_size//16
                elif i == twoscaleIndex:
                    n = mmobj.CNNPatchMerge(dimEmbed, embed_dim*2, embed_dim)
            m = Block(
                dim=curDim, num_heads=curNumHead, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                kernel_size=leff_local_size, with_bn=leff_with_bn,patchHW=patchHW)
            if n:
                self.blocks.append(nn.Sequential(n, m))
            else:    
                self.blocks.append(m)

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
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
        return {'pos_embed'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.i2t(x)
        #print('trace1 x', x.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        #print('trace2 x', x.shape)
        
        for blk in self.blocks:
            #print('trace x', x.shape)
            x= blk(x)
        return self.norm(x)

    def forward(self, img, ci2pvitObj):
        x = ci2pvitObj.to_patch_embedding(img)
        x = self.forward_features(x)
        return (x,)

def ceit_base_patch16_256(dimEmbed=192, pretrained=False, **kwargs):
    model = CeIT(
        img_size=256,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,dimEmbed=dimEmbed,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model
