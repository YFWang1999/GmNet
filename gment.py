
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.jit import Final


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, kernel_size=7,
                 f12_bn=False, g_bn=True, dwconv2_bn=False, act=nn.ReLU,
                 drop_path=0., layer_scale=1e-6):

        super().__init__()
        self.dwconv = ConvBN(dim, dim, kernel_size, 1, (kernel_size-1)//2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio*dim, 1, with_bn=f12_bn)
        self.g = ConvBN(mlp_ratio*dim, dim, 1, with_bn=g_bn)
        self.dwconv2 = ConvBN(dim, dim, kernel_size, 1, (kernel_size - 1) // 2, groups=dim, with_bn=dwconv2_bn)
        self.act = act()
        self.gamma = nn.Parameter(layer_scale * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale > 0 else None
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)

    def forward(self, x):
        input = x
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.f1(x)
        x = self.act(x) * (x)
        x = self.g(x)
        
        
        x = self.dwconv2(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if hasattr(self, "drop_path"):
            x = input + self.drop_path(x)
        else:
            x =  input + x
        return x



class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, with_bn=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.add_module('conv', torch.nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class PermuteLienar(nn.Module):
    def __init__(self, in_planes, out_planes):
        """
        input: [B, C, H, W]
        """
        super().__init__()
        self.layer = nn.Linear(in_planes, out_planes)

    def forward(self, x):
        return self.layer(x.permute(0,2,3,1)).permute(0,3,1,2)


class Model(nn.Module):
    def __init__(self, num_classes=1000,
                 # newwork configuration
                 embed_dim=[32, 64, 128, 256], depths=[2, 2, 8, 2],
                 f12_bn=False, g_bn=False, dwconv2_bn=False, act=nn.ReLU, downsampler_act = nn.ReLU,
                 mlp_ratio=[4, 4, 4, 4], layer_scale=1e-6,
                 drop_path_rate=0.0, kernel_size=7, block=None,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32

        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            act(),
            # nn.Conv2d(self.in_channel, self.in_channel, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(self.in_channel)
        )
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            down_sampler = nn.Sequential(
                ConvBN(self.in_channel, embed_dim[i_layer], 3, 2, 1),
                downsampler_act()
            )
            self.in_channel = embed_dim[i_layer]
            blocks = [
                block(self.in_channel, mlp_ratio=mlp_ratio[i_layer], kernel_size=kernel_size,
                      f12_bn=f12_bn, g_bn=g_bn, dwconv2_bn=dwconv2_bn, act=act,
                      drop_path=dpr[cur+i], layer_scale=layer_scale)
                for i in range(depths[i_layer])]
            cur += depths[i_layer]
            stage = nn.Sequential(down_sampler, *blocks)
            self.stages.append(stage)
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        x = self.stem(x) # [B,in_planes, 112,112]
        for stage in self.stages:
            x = stage(x)
            #pdb.set_trace()
        #pdb.set_trace()
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def convert_model(model):
    # search for all to-be-replaced layers
    for name, layer in model.named_children():
        if isinstance(layer, ConvBN) and layer.kernel_size==1:
            fc_layer = PermuteLienar(in_planes=layer.in_planes, out_planes=layer.out_planes)
            setattr(model, name, fc_layer)
        elif isinstance(layer, nn.Module):
            convert_model(layer)


@register_model
def gmnet_s3(pretrained=False, **kwargs):
    base_dim = 48
    dim_expand = [1, 2, 4, 8]
    embed_dim = [int(base_dim * expand) for expand in dim_expand]
    print(embed_dim)
    depths = [3,3,8,3]   # [1,1,4,2]
    mlp_ratio = [4,4,4,4]
    kernel_size = 7
    model = Model(
        embed_dim=embed_dim, depths=depths, mlp_ratio=mlp_ratio, kernel_size=kernel_size,
        f12_bn=False, g_bn=True, dwconv2_bn=False, act=nn.ReLU6, downsampler_act=nn.Identity,
        block = Block,
        **kwargs
    )
    if pretrained:
        convert_model(model)
    return model

