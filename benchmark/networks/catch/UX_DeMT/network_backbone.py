#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2023

@author: mingliang.yang
"""

from typing import Tuple
import torch.nn as nn
import functools
import torch
import numpy as np

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock,UnetrPrUpBlock
from monai.networks.blocks.upsample import UpSample
# from monai.networks.blocks.selfattention import SABlock
from typing import Union
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from networks.UX_DeMT.uxnet_encoder import uxnet_conv
# from networks.UX_DeMT_3D.DeformableBlock import DeformBasicBlock
from einops import rearrange
from .DHDC.ADHDC_Net import HDC_module,Conv_down,DHDC_module,conv_trans_block_3d,hd
# from .cc_attention.functions import CrissCrossAttention
from .cc_attention.check3d import CrissCrossAttention3D2
# from .heads.UX_DeMT_head import UX_DeMTHead
# from functools import partial


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

def transform_inputs(inputs):
    # in_index=[0, 1]
    inputs = [inputs[i] for i in range(len(inputs))]
    upsampled_inputs = [
        nn.functional.interpolate(
            input=x,
            size=inputs[0].shape[2:],
            mode='trilinear',
            align_corners=False) for x in inputs
    ]
    inputs = torch.cat(upsampled_inputs, dim=1)
    return inputs

class dehc(nn.Module):
    def __init__(self, c1=4, c2=4):
        super(dehc,self).__init__()
        self.c1=c1
        self.c2=c2
        self.activation=nn.LeakyReLU(inplace=False)
        self.block=conv_trans_block_3d(self.c1, self.c2, self.activation)
        self.hdc=HDC_module(self.c2*2, self.c2, self.activation)

    def forward(self,x1,x2):
        x=self.block(x1)
        x = torch.cat((x, x2), dim=1)
        out=self.hdc(x)
        return out

#0722 (64, 128, 256, 512).
class UX_DeMT1(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.activation=nn.LeakyReLU(inplace=False)
        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths[0:2],
            dims=self.feat_size[0:2],
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.d3=nn.Sequential(HDC_module(self.feat_size[1], self.feat_size[1], self.activation),
                                    Conv_down(self.feat_size[1], self.feat_size[2], self.activation))
        self.d4=nn.Sequential(HDC_module(self.feat_size[2], self.feat_size[2], self.activation),
                                    Conv_down(self.feat_size[2], self.feat_size[3], self.activation))
        # self.d5=nn.Sequential(HDC_module(self.feat_size[3], self.feat_size[3], self.activation),
        #                             Conv_down(self.feat_size[3], self.hidden_size, self.activation))
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.encoder2=HDC_module(self.feat_size[0], self.feat_size[1], self.activation)
        # self.encoder3=HDC_module(self.feat_size[1], self.feat_size[2], self.activation)
        # self.encoder4=HDC_module(self.feat_size[2], self.feat_size[3], self.activation)
        # self.encoder5=HDC_module(self.feat_size[3], self.hidden_size, self.activation)

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.bridge=DHDC_module(self.hidden_size,self.hidden_size,self.activation)#self.feat_size[3]
        self.bridge=DHDC_module(self.feat_size[3],self.feat_size[3],self.activation)
        self.decoder5=dehc(self.feat_size[3], self.feat_size[2])
        self.decoder4 = dehc(self.feat_size[2], self.feat_size[1])

        # self.decoder3 = dehc(self.feat_size[2], self.feat_size[1])
        # self.decoder2 = dehc(self.feat_size[1], self.feat_size[0])

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.hidden_size,
        #     out_channels=self.feat_size[3],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[2],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0]//2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        #syn
        self.encoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2b = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1]*2,
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.attetion1=CrissCrossAttention3D2()
        # self.attetion2 = CrissCrossAttention3D2()

        self.branch1a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0]//2,#self.feat_size[0]
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0]//2,
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.hd1a = hd(2, 1)
        self.hd2a = hd(4, 2)

        self.hd1b = hd(2, 1)
        self.hd2b = hd(4, 1)

        self.branch2a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_chans)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=1)  # Act.PRELU   nn.Tanh()
        self.act = nn.Tanh()
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        x2 = outs[0]
        x3 = outs[1]
        x4 = self.d3(x3)#outs[2]
        x5 = self.d4(x4)#outs[3]

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(x2)
        # enc3 = self.encoder3(x3)
        # enc4 = self.encoder4(x4)#
        # enc_hidden = self.encoder5(x5)
        bridge=self.bridge(x5)
        dec3 = self.decoder5(bridge, x4)
        dec2 = self.decoder4(dec3, x3)#seg enc3
        # dec3 = self.decoder5(enc_hidden, enc4)
        # dec2 = self.decoder4(dec3, enc3)#seg
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # dec0b = self.decoder2b(dec1, enc1)

        outa = self.decoder1(dec0)#公用特征，


        b1a = self.branch1a(outa)
        b1b = self.branch1b(outa)

        repeat1 = x_in.repeat(1, 8, 1, 1, 1)
        Vessel1 = b1b - repeat1
        cta1 = b1a + repeat1

        b2a=self.branch2a(torch.cat((b1a,Vessel1),1))
        b2b = self.branch2b(torch.cat((b1b,cta1),1))
        repeat2 = x_in.repeat(1, 16, 1, 1, 1)
        Vessel2=b2b-repeat2
        cta2=b2a+repeat2

        b3a=self.branch3a(torch.cat((b2a,Vessel2),1))
        b3b = self.branch3b(torch.cat((b2b,cta2),1))
        repeat3 = x_in.repeat(1, 32, 1, 1, 1)
        Vessel3=b3b-repeat3
        cta3=b3a+repeat3

        outa=self.out1(torch.cat((b3a,Vessel3),1))
        outb = self.out2(torch.cat((b3b,cta3),1))
        # feat = self.conv_proj(dec4)
        return outa, outb  # (self.act(self.out2(out2))+1)/2

#0725 0801  (64, 128, 256, 512)8+分别块通道注意力
class UX_DeMT2(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.activation=nn.LeakyReLU(inplace=False)
        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths[0:2],
            dims=self.feat_size[0:2],
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.d3=nn.Sequential(HDC_module(self.feat_size[1], self.feat_size[1], self.activation),
                                    Conv_down(self.feat_size[1], self.feat_size[2], self.activation))
        self.d4=nn.Sequential(HDC_module(self.feat_size[2], self.feat_size[2], self.activation),
                                    Conv_down(self.feat_size[2], self.feat_size[3], self.activation))
        # self.d5=nn.Sequential(HDC_module(self.feat_size[3], self.feat_size[3], self.activation),
        #                             Conv_down(self.feat_size[3], self.hidden_size, self.activation))
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.encoder2=HDC_module(self.feat_size[0], self.feat_size[1], self.activation)
        # self.encoder3=HDC_module(self.feat_size[1], self.feat_size[2], self.activation)
        # self.encoder4=HDC_module(self.feat_size[2], self.feat_size[3], self.activation)
        # self.encoder5=HDC_module(self.feat_size[3], self.hidden_size, self.activation)

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.bridge=DHDC_module(self.hidden_size,self.hidden_size,self.activation)#self.feat_size[3]
        self.bridge=DHDC_module(self.feat_size[3],self.feat_size[3],self.activation)
        self.decoder5=dehc(self.feat_size[3], self.feat_size[2])
        self.decoder4 = dehc(self.feat_size[2], self.feat_size[1])

        # self.decoder3 = dehc(self.feat_size[2], self.feat_size[1])
        # self.decoder2 = dehc(self.feat_size[1], self.feat_size[0])

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.hidden_size,
        #     out_channels=self.feat_size[3],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[2],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        #syn
        self.encoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2b = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1]*2,
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.attetion1=CrissCrossAttention3D2()
        # self.attetion2 = CrissCrossAttention3D2()

        self.branch1a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8,#self.feat_size[0]
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.hd1a = hd(2, 1)
        self.hd2a = hd(4, 2)

        self.hd1b = hd(2, 1)
        self.hd2b = hd(4, 1)

        self.branch2a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_chans)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=1)  # Act.PRELU   nn.Tanh()
        self.act = nn.Tanh()
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        x2 = outs[0]
        x3 = outs[1]
        x4 = self.d3(x3)#outs[2]
        x5 = self.d4(x4)#outs[3]

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(x2)
        # enc3 = self.encoder3(x3)
        # enc4 = self.encoder4(x4)#
        # enc_hidden = self.encoder5(x5)
        bridge=self.bridge(x5)
        dec3 = self.decoder5(bridge, x4)
        dec2 = self.decoder4(dec3, x3)#seg enc3
        # dec3 = self.decoder5(enc_hidden, enc4)
        # dec2 = self.decoder4(dec3, enc3)#seg
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # dec0b = self.decoder2b(dec1, enc1)

        outa = self.decoder1(dec0)#公用特征，
        outb = self.decoder1b(dec0)
        s0a, s1a, s2a, s3a = outa.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1a = self.hd1a(s1a, s2a)
        l2a = torch.cat([s1a, s2a], dim=1)
        l3a = self.hd2a(s3a, l2a)
        outa = torch.cat([s0a, l1a, s2a, l3a], dim=1)

        s0b, s1b, s2b, s3b = outb.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1b = self.hd1b(s1b, s2b)
        l2b = torch.cat([s1b, s2b], dim=1)
        l3b = self.hd2b(s3b, l2b)
        outb = torch.cat([s0b, l1b, s2b, l3b], dim=1)

        # x=torch.cat((outa,outb),dim=1)
        # outa=self.attetion1(x,x,outa)
        # outb = self.attetion2(x, x, outb)

        b1a = self.branch1a(outa)
        b1b = self.branch1b(outb)

        repeat1 = x_in.repeat(1, 8, 1, 1, 1)
        Vessel1 = b1b - repeat1
        cta1 = b1a + repeat1

        b2a=self.branch2a(torch.cat((b1a,Vessel1),1))
        b2b = self.branch2b(torch.cat((b1b,cta1),1))
        repeat2 = x_in.repeat(1, 16, 1, 1, 1)
        Vessel2=b2b-repeat2
        cta2=b2a+repeat2

        b3a=self.branch3a(torch.cat((b2a,Vessel2),1))
        b3b = self.branch3b(torch.cat((b2b,cta2),1))
        repeat3 = x_in.repeat(1, 32, 1, 1, 1)
        Vessel3=b3b-repeat3
        cta3=b3a+repeat3

        outa=self.out1(torch.cat((b3a,Vessel3),1))
        outb = self.out2(torch.cat((b3b,cta3),1))
        # feat = self.conv_proj(dec4)
        return outa, outb  # (self.act(self.out2(out2))+1)/2

#0726 (64, 128, 256, 512)+公用块注意力模块权重.扩展了注意力通道数目
class UX_DeMT4(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.activation=nn.LeakyReLU(inplace=False)
        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths[0:2],
            dims=self.feat_size[0:2],
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.d3=nn.Sequential(HDC_module(self.feat_size[1], self.feat_size[1], self.activation),
                                    Conv_down(self.feat_size[1], self.feat_size[2], self.activation))
        self.d4=nn.Sequential(HDC_module(self.feat_size[2], self.feat_size[2], self.activation),
                                    Conv_down(self.feat_size[2], self.feat_size[3], self.activation))
        # self.d5=nn.Sequential(HDC_module(self.feat_size[3], self.feat_size[3], self.activation),
        #                             Conv_down(self.feat_size[3], self.hidden_size, self.activation))
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.encoder2=HDC_module(self.feat_size[0], self.feat_size[1], self.activation)
        # self.encoder3=HDC_module(self.feat_size[1], self.feat_size[2], self.activation)
        # self.encoder4=HDC_module(self.feat_size[2], self.feat_size[3], self.activation)
        # self.encoder5=HDC_module(self.feat_size[3], self.hidden_size, self.activation)

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.bridge=DHDC_module(self.hidden_size,self.hidden_size,self.activation)#self.feat_size[3]
        self.bridge=DHDC_module(self.feat_size[3],self.feat_size[3],self.activation)
        self.decoder5=dehc(self.feat_size[3], self.feat_size[2])
        self.decoder4 = dehc(self.feat_size[2], self.feat_size[1])

        # self.decoder3 = dehc(self.feat_size[2], self.feat_size[1])
        # self.decoder2 = dehc(self.feat_size[1], self.feat_size[0])

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.hidden_size,
        #     out_channels=self.feat_size[3],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[2],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0]//4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        #syn
        self.encoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2b = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1]*2,
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0]//4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.attetion1=CrissCrossAttention3D2()
        # self.attetion2 = CrissCrossAttention3D2()
        self.hd1a = hd(4, 1)
        self.hd2a = hd(8, 4)

        self.branch1a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0]//4,#self.feat_size[0]
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0]//4,
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.branch2a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_chans)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=1)  # Act.PRELU   nn.Tanh()
        self.act = nn.Tanh()
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        x2 = outs[0]
        x3 = outs[1]
        x4 = self.d3(x3)#outs[2]
        x5 = self.d4(x4)#outs[3]

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(x2)
        # enc3 = self.encoder3(x3)
        # enc4 = self.encoder4(x4)#
        # enc_hidden = self.encoder5(x5)
        bridge=self.bridge(x5)
        dec3 = self.decoder5(bridge, x4)
        dec2 = self.decoder4(dec3, x3)#seg enc3
        # dec3 = self.decoder5(enc_hidden, enc4)
        # dec2 = self.decoder4(dec3, enc3)#seg
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # dec0b = self.decoder2b(dec1, enc1)

        outa = self.decoder1(dec0)#公用特征，
        outb = self.decoder1b(dec0)
        s0a, s1a, s2a, s3a = outa.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1a = self.hd1a(s1a, s2a)
        l2a = torch.cat([s1a, s2a], dim=1)
        l3a = self.hd2a(s3a, l2a)
        outa = torch.cat([s0a, l1a, s2a, l3a], dim=1)

        s0b, s1b, s2b, s3b = outb.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1b = self.hd1a(s1b, s2b)
        l2b = torch.cat([s1b, s2b], dim=1)
        l3b = self.hd2a(s3b, l2b)
        outb = torch.cat([s0b, l1b, s2b, l3b], dim=1)

        # x=torch.cat((outa,outb),dim=1)
        # outa=self.attetion1(x,x,outa)
        # outb = self.attetion2(x, x, outb)

        b1a = self.branch1a(outa)
        b1b = self.branch1b(outb)

        repeat1 = x_in.repeat(1, 8, 1, 1, 1)
        Vessel1 = b1b - repeat1
        cta1 = b1a + repeat1

        b2a=self.branch2a(torch.cat((b1a,Vessel1),1))
        b2b = self.branch2b(torch.cat((b1b,cta1),1))
        repeat2 = x_in.repeat(1, 16, 1, 1, 1)
        Vessel2=b2b-repeat2
        cta2=b2a+repeat2

        b3a=self.branch3a(torch.cat((b2a,Vessel2),1))
        b3b = self.branch3b(torch.cat((b2b,cta2),1))
        repeat3 = x_in.repeat(1, 32, 1, 1, 1)
        Vessel3=b3b-repeat3
        cta3=b3a+repeat3

        outa=self.out1(torch.cat((b3a,Vessel3),1))
        outb = self.out2(torch.cat((b3b,cta3),1))
        # feat = self.conv_proj(dec4)
        return outa, outb  # (self.act(self.out2(out2))+1)/2

#0803
class UX_DeMT44(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.activation=nn.LeakyReLU(inplace=False)
        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths[0:2],
            dims=self.feat_size[0:2],
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.d3=nn.Sequential(HDC_module(self.feat_size[1], self.feat_size[1], self.activation),
                                    Conv_down(self.feat_size[1], self.feat_size[2], self.activation))
        self.d4=nn.Sequential(HDC_module(self.feat_size[2], self.feat_size[2], self.activation),
                                    Conv_down(self.feat_size[2], self.feat_size[3], self.activation))
        # self.d5=nn.Sequential(HDC_module(self.feat_size[3], self.feat_size[3], self.activation),
        #                             Conv_down(self.feat_size[3], self.hidden_size, self.activation))
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.encoder2=HDC_module(self.feat_size[0], self.feat_size[1], self.activation)
        # self.encoder3=HDC_module(self.feat_size[1], self.feat_size[2], self.activation)
        # self.encoder4=HDC_module(self.feat_size[2], self.feat_size[3], self.activation)
        # self.encoder5=HDC_module(self.feat_size[3], self.hidden_size, self.activation)

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.bridge=DHDC_module(self.hidden_size,self.hidden_size,self.activation)#self.feat_size[3]
        self.bridge=DHDC_module(self.feat_size[3],self.feat_size[3],self.activation)
        self.decoder5=dehc(self.feat_size[3], self.feat_size[2])
        self.decoder4 = dehc(self.feat_size[2], self.feat_size[1])

        # self.decoder3 = dehc(self.feat_size[2], self.feat_size[1])
        # self.decoder2 = dehc(self.feat_size[1], self.feat_size[0])

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.hidden_size,
        #     out_channels=self.feat_size[3],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[2],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        #syn
        self.encoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2b = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1]*2,
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.attetion1=CrissCrossAttention3D2()
        # self.attetion2 = CrissCrossAttention3D2()

        self.branch1a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8,#self.feat_size[0]
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.hd1a = hd(2, 1)
        self.hd2a = hd(4, 2)

        self.hd1b = hd(2, 1)
        self.hd2b = hd(4, 1)

        self.branch2a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_chans)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=1)  # Act.PRELU   nn.Tanh()
        self.act = nn.Tanh()
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        x2 = outs[0]
        x3 = outs[1]
        x4 = self.d3(x3)#outs[2]
        x5 = self.d4(x4)#outs[3]

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(x2)
        # enc3 = self.encoder3(x3)
        # enc4 = self.encoder4(x4)#
        # enc_hidden = self.encoder5(x5)
        bridge=self.bridge(x5)
        dec3 = self.decoder5(bridge, x4)
        dec2 = self.decoder4(dec3, x3)#seg enc3
        # dec3 = self.decoder5(enc_hidden, enc4)
        # dec2 = self.decoder4(dec3, enc3)#seg
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # dec0b = self.decoder2b(dec1, enc1)

        outa = self.decoder1(dec0)#公用特征，
        outb = self.decoder1b(dec0)
        s0a, s1a, s2a, s3a = outa.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1a = self.hd1a(s1a, s2a)
        l2a = torch.cat([s1a, s2a], dim=1)
        l3a = self.hd2a(s3a, l2a)
        outa = torch.cat([s0a, l1a, s2a, l3a], dim=1)

        s0b, s1b, s2b, s3b = outb.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1b = self.hd1a(s1b, s2b)
        l2b = torch.cat([s1b, s2b], dim=1)
        l3b = self.hd2a(s3b, l2b)
        outb = torch.cat([s0b, l1b, s2b, l3b], dim=1)

        # x=torch.cat((outa,outb),dim=1)
        # outa=self.attetion1(x,x,outa)
        # outb = self.attetion2(x, x, outb)

        b1a = self.branch1a(outa)
        b1b = self.branch1b(outb)

        repeat1 = x_in.repeat(1, 8, 1, 1, 1)
        Vessel1 = b1b - repeat1
        cta1 = b1a + repeat1

        b2a=self.branch2a(torch.cat((b1a,Vessel1),1))
        b2b = self.branch2b(torch.cat((b1b,cta1),1))
        repeat2 = x_in.repeat(1, 16, 1, 1, 1)
        Vessel2=b2b-repeat2
        cta2=b2a+repeat2

        b3a=self.branch3a(torch.cat((b2a,Vessel2),1))
        b3b = self.branch3b(torch.cat((b2b,cta2),1))
        repeat3 = x_in.repeat(1, 32, 1, 1, 1)
        Vessel3=b3b-repeat3
        cta3=b3a+repeat3

        outa=self.out1(torch.cat((b3a,Vessel3),1))
        outb = self.out2(torch.cat((b3b,cta3),1))
        # feat = self.conv_proj(dec4)
        return outa, outb  # (self.act(self.out2(out2))+1)/2

#0727  (64, 128, 256, 512)8+分别常规通道注意力
class UX_DeMT55(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 512,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.activation=nn.LeakyReLU(inplace=False)
        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths[0:2],
            dims=self.feat_size[0:2],
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.d3=nn.Sequential(HDC_module(self.feat_size[1], self.feat_size[1], self.activation),
                                    Conv_down(self.feat_size[1], self.feat_size[2], self.activation))
        self.d4=nn.Sequential(HDC_module(self.feat_size[2], self.feat_size[2], self.activation),
                                    Conv_down(self.feat_size[2], self.feat_size[3], self.activation))
        # self.d5=nn.Sequential(HDC_module(self.feat_size[3], self.feat_size[3], self.activation),
        #                             Conv_down(self.feat_size[3], self.hidden_size, self.activation))
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.encoder2=HDC_module(self.feat_size[0], self.feat_size[1], self.activation)
        # self.encoder3=HDC_module(self.feat_size[1], self.feat_size[2], self.activation)
        # self.encoder4=HDC_module(self.feat_size[2], self.feat_size[3], self.activation)
        # self.encoder5=HDC_module(self.feat_size[3], self.hidden_size, self.activation)

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.bridge=DHDC_module(self.hidden_size,self.hidden_size,self.activation)#self.feat_size[3]
        self.bridge=DHDC_module(self.feat_size[3],self.feat_size[3],self.activation)
        self.decoder5=dehc(self.feat_size[3], self.feat_size[2])
        self.decoder4 = dehc(self.feat_size[2], self.feat_size[1])

        # self.decoder3 = dehc(self.feat_size[2], self.feat_size[1])
        # self.decoder2 = dehc(self.feat_size[1], self.feat_size[0])

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.hidden_size,
        #     out_channels=self.feat_size[3],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[2],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        #syn
        self.encoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2b = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1]*2,
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.attetion1=CrissCrossAttention3D2()
        self.attetion2 = CrissCrossAttention3D2()

        self.branch1a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8,#self.feat_size[0]
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.hd1a = hd(2, 1)
        self.hd2a = hd(4, 2)

        self.hd1b = hd(2, 1)
        self.hd2b = hd(4, 1)

        self.branch2a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_chans)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=1)  # Act.PRELU   nn.Tanh()
        self.act = nn.Tanh()
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

        self.hidden_size=8
        self.linear1a = nn.Sequential(nn.Linear(8, self.hidden_size), nn.LayerNorm(self.hidden_size))
        self.linear1b = nn.Sequential(nn.Linear(8, self.hidden_size), nn.LayerNorm(self.hidden_size))
        self.task_fusion = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, dropout=0.)
        self.smlp1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size))
        self.smlp2a = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size))
        self.smlp2b = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size))

        self.task_querya = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, dropout=0.)
        self.task_queryb = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, dropout=0.)

        # self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 4,batch_first=True)
        # self.multihead_attn = MultiHeadAttention(self.hidden_size, 4,8,8)
    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        x2 = outs[0]
        x3 = outs[1]
        x4 = self.d3(x3)#outs[2]
        x5 = self.d4(x4)#outs[3]

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(x2)
        # enc3 = self.encoder3(x3)
        # enc4 = self.encoder4(x4)#
        # enc_hidden = self.encoder5(x5)
        bridge=self.bridge(x5)
        dec3 = self.decoder5(bridge, x4)
        dec2 = self.decoder4(dec3, x3)#seg enc3
        # dec3 = self.decoder5(enc_hidden, enc4)
        # dec2 = self.decoder4(dec3, enc3)#seg
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # dec0b = self.decoder2b(dec1, enc1)

        outa = self.decoder1(dec0)#公用特征，
        outb = self.decoder1b(dec0)

        b, c, h, w, d = outa.shape
        outa = self.linear1a(outa.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        outb = self.linear1b(outb.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        outa = rearrange(outa, "b c h w d-> b (h w d) c").contiguous()
        outb = rearrange(outb, "b c h w d-> b (h w d) c").contiguous()

        task_cat = torch.cat([outa,outb], dim=0)  # concaten in batch. having someing trick

        task_cat = self.task_fusion(task_cat, task_cat, task_cat)[0]
        task_cat = self.smlp1(task_cat)
        aa=self.task_querya(outa, task_cat, task_cat)[0]
        outa=outa + self.smlp2a(self.task_querya(outa, task_cat, task_cat)[0])
        outa=rearrange(outa, "b (h w d) c -> b c h w d", h=h, w=w).contiguous()

        outb = outb + self.smlp2b(self.task_queryb(outb, task_cat, task_cat)[0])
        outb = rearrange(outb, "b (h w d) c -> b c h w d", h=h, w=w).contiguous()

        # x=torch.cat((outa,outb),dim=1)
        # outa=self.attetion1(x,x,outa)
        # outb = self.attetion2(x, x, outb)
        # a=self.multihead_attn(outa,outa,outa)

        b1a = self.branch1a(outa)
        b1b = self.branch1b(outb)

        repeat1 = x_in.repeat(1, 8, 1, 1, 1)
        Vessel1 = b1b - repeat1
        cta1 = b1a + repeat1

        b2a=self.branch2a(torch.cat((b1a,Vessel1),1))
        b2b = self.branch2b(torch.cat((b1b,cta1),1))
        repeat2 = x_in.repeat(1, 16, 1, 1, 1)
        Vessel2=b2b-repeat2
        cta2=b2a+repeat2

        b3a=self.branch3a(torch.cat((b2a,Vessel2),1))
        b3b = self.branch3b(torch.cat((b2b,cta2),1))
        repeat3 = x_in.repeat(1, 32, 1, 1, 1)
        Vessel3=b3b-repeat3
        cta3=b3a+repeat3

        outa=self.out1(torch.cat((b3a,Vessel3),1))
        outb = self.out2(torch.cat((b3b,cta3),1))
        # feat = self.conv_proj(dec4)
        return outa, outb  # (self.act(self.out2(out2))+1)/2

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class Self_Cross_Attn(nn.Module):
    """ Self attention Layer"""
    #GAN self-attention
    def __init__(self, in_dim, activation):
        super(Self_Cross_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x1,x2):
        """
            inputs :
                x1 : input feature maps( B  C W  H)
                x2 ：symmetry input
            returns :
                out : self attention value + input feature
                attention: B x N x N (N is Width*Height)
        """
        # b, c, h, w, d
        m_batchsize, C, width, height,depth = x1.size()#m_batchsize, C, height，width
        proj_query = self.query_conv(x1).view(m_batchsize, -1, width * height*depth).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x2).view(m_batchsize, -1, width * height*depth)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x2).view(m_batchsize, -1, width * height*depth)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x1#要不要这一步需要验证
        return out, attention

class UX_DeMT_deformable(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2,2, 2],#, 2, 2
            feat_size=[48, 96,192, 384],#, 192, 384
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 16,#768
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims
        self.tasks=['seg','syn']

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.in_channel=336#720#self.feat_size[0]*(len(self.feat_size)-1)
        self.linear1 = nn.Sequential(nn.Linear(self.in_channel, self.hidden_size), nn.LayerNorm(self.hidden_size))
        # self.defor_mixer1=DeformBasicBlock(dc,dc)
        # self.defor_mixer2 = DeformBasicBlock(dc,dc)
        self.defor_mixers = nn.ModuleList([DeformBasicBlock(self.hidden_size,self.hidden_size)  for t in range (len(self.tasks))])
        self.task_fusion = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, dropout=0.)
        self.smlp1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size))
        self.smlp2 = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size))  for t in range (len(self.tasks))])

        self.task_querys = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, dropout=0.)  for t in range (len(self.tasks))])

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.branch1a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0]//6,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0]//6,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0]//3,
            out_channels=self.feat_size[0]//3,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0]//3,
            out_channels=self.feat_size[0]//3,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_chans)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=1)  # Act.PRELU   nn.Tanh()
        self.act = nn.Tanh()
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        inp = transform_inputs(outs)  # bchw
        b, c, h, w ,d= inp.shape
        inp = self.linear1(inp.permute(0, 2, 3,4,1)).permute(0, 4, 1, 2,3)
        outs = []
        for ind, defor_mixer in enumerate(self.defor_mixers):
            out = defor_mixer(inp)
            out = rearrange(out, "b c h w d-> b (h w d) c").contiguous()
            outs.append(out)  #
        task_cat = torch.cat(outs, dim=0)  # concaten in batch. having someing trick

        task_cat = self.task_fusion(task_cat, task_cat, task_cat)[0]
        task_cat = self.smlp1(task_cat)

        outs_ls = []
        for ind, task_query in enumerate(self.task_querys):
            inp = outs[ind] + self.smlp2[ind](task_query(outs[ind], task_cat, task_cat)[0])
            outs_ls.append(rearrange(inp, "b (h w) c -> b c h w", h=h, w=w).contiguous())

        enc1 = self.encoder1(x_in)
        # print(enc1.size())
        x2 = outs[0]
        enc2 = self.encoder2(x2)

        dec2 = self.encoder2(outs[0])
        out = self.encoder2(outs[0])

        b1a = self.branch1a(out)
        b1b = self.branch1b(out)
        repeat1 = x_in.repeat(1, 8, 1, 1, 1)
        Vessel1 = b1b - repeat1
        cta1 = b1a + repeat1

        b2a=self.branch2a(torch.cat((b1a,Vessel1),1))
        b2b = self.branch2b(torch.cat((b1b,cta1),1))
        repeat2 = x_in.repeat(1, 16, 1, 1, 1)
        Vessel2=b2b-repeat2
        cta2=b2a+repeat2

        b3a=self.branch3a(torch.cat((b2a,Vessel2),1))
        b3b = self.branch3b(torch.cat((b2b,cta2),1))
        repeat3 = x_in.repeat(1, 32, 1, 1, 1)
        Vessel3=b3b-repeat3
        cta3=b3a+repeat3

        outa=self.out1(torch.cat((b3a,Vessel3),1))
        outb = self.out2(torch.cat((b3b,cta3),1))

        # feat = self.conv_proj(dec4)
        return outa, outb  # (self.act(self.out2(out2))+1)/2

class Discriminator_3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d, use_sigmoid=False):
        super(Discriminator_3D, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = (2,4,4)
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        x=self.model(input)
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)

#0724 0727  0802 0803(64, 128, 256, 512)+公用块通道注意力模块权重.
#dice+ssim=1.4833, dice=0.8510
class UX_DeMT(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.activation=nn.LeakyReLU(inplace=False)
        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths[0:2],
            dims=self.feat_size[0:2],
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.d3=nn.Sequential(HDC_module(self.feat_size[1], self.feat_size[1], self.activation),
                                    Conv_down(self.feat_size[1], self.feat_size[2], self.activation))
        self.d4=nn.Sequential(HDC_module(self.feat_size[2], self.feat_size[2], self.activation),
                                    Conv_down(self.feat_size[2], self.feat_size[3], self.activation))
        # self.d5=nn.Sequential(HDC_module(self.feat_size[3], self.feat_size[3], self.activation),
        #                             Conv_down(self.feat_size[3], self.hidden_size, self.activation))
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.encoder2=HDC_module(self.feat_size[0], self.feat_size[1], self.activation)
        # self.encoder3=HDC_module(self.feat_size[1], self.feat_size[2], self.activation)
        # self.encoder4=HDC_module(self.feat_size[2], self.feat_size[3], self.activation)
        # self.encoder5=HDC_module(self.feat_size[3], self.hidden_size, self.activation)

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.bridge=DHDC_module(self.hidden_size,self.hidden_size,self.activation)#self.feat_size[3]
        self.bridge=DHDC_module(self.feat_size[3],self.feat_size[3],self.activation)
        self.decoder5=dehc(self.feat_size[3], self.feat_size[2])
        self.decoder4 = dehc(self.feat_size[2], self.feat_size[1])

        # self.decoder3 = dehc(self.feat_size[2], self.feat_size[1])
        # self.decoder2 = dehc(self.feat_size[1], self.feat_size[0])

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.hidden_size,
        #     out_channels=self.feat_size[3],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[2],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        #syn
        self.encoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2b = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1]*2,
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.attetion1=CrissCrossAttention3D2()
        # self.attetion2 = CrissCrossAttention3D2()

        self.branch1a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8,#self.feat_size[0]
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.hd1a = hd(2, 1)
        self.hd2a = hd(4, 2)

        self.hd1b = hd(2, 1)
        self.hd2b = hd(4, 1)

        self.branch2a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch3b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_chans)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=1)  # Act.PRELU   nn.Tanh()
        self.act = nn.Tanh()
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        x2 = outs[0]
        x3 = outs[1]
        x4 = self.d3(x3)#outs[2]
        x5 = self.d4(x4)#outs[3]

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(x2)
        # enc3 = self.encoder3(x3)
        # enc4 = self.encoder4(x4)#
        # enc_hidden = self.encoder5(x5)
        bridge=self.bridge(x5)
        dec3 = self.decoder5(bridge, x4)
        dec2 = self.decoder4(dec3, x3)#seg enc3
        # dec3 = self.decoder5(enc_hidden, enc4)
        # dec2 = self.decoder4(dec3, enc3)#seg
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # dec0b = self.decoder2b(dec1, enc1)

        outa = self.decoder1(dec0)#公用特征，
        outb = self.decoder1b(dec0)
        s0a, s1a, s2a, s3a = outa.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1a = self.hd1a(s1a, s2a)
        l2a = torch.cat([s1a, s2a], dim=1)
        l3a = self.hd2a(s3a, l2a)
        outa = torch.cat([s0a, l1a, s2a, l3a], dim=1)

        s0b, s1b, s2b, s3b = outb.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1b = self.hd1a(s1b, s2b)
        l2b = torch.cat([s1b, s2b], dim=1)
        l3b = self.hd2a(s3b, l2b)
        outb = torch.cat([s0b, l1b, s2b, l3b], dim=1)

        # x=torch.cat((outa,outb),dim=1)
        # outa=self.attetion1(x,x,outa)
        # outb = self.attetion2(x, x, outb)

        b1a = self.branch1a(outa)
        b1b = self.branch1b(outb)

        repeat1 = x_in.repeat(1, 8, 1, 1, 1)
        Vessel1 = b1b - repeat1
        cta1 = b1a + repeat1

        b2a=self.branch2a(torch.cat((b1a,Vessel1),1))
        b2b = self.branch2b(torch.cat((b1b,cta1),1))
        repeat2 = x_in.repeat(1, 16, 1, 1, 1)
        Vessel2=b2b-repeat2
        cta2=b2a+repeat2

        b3a=self.branch3a(torch.cat((b2a,Vessel2),1))
        b3b = self.branch3b(torch.cat((b2b,cta2),1))
        repeat3 = x_in.repeat(1, 32, 1, 1, 1)
        Vessel3=b3b-repeat3
        cta3=b3a+repeat3

        outa=self.out1(torch.cat((b3a,Vessel3),1))
        outb = self.out2(torch.cat((b3b,cta3),1))
        # feat = self.conv_proj(dec4)
        return outa, outb  # (self.act(self.out2(out2))+1)/2

if  __name__=='__main__':
    input1=torch.Tensor(np.random.rand(4,1,96,96,48))#2,192,192,4
    input2=torch.Tensor(np.random.rand(4,1,96,96,48))#2,192,192,4
    # a=torch.where((input1>0)&(input1<0.4),2,1)
    # b=1
    model = UX_DeMT(1,2)
    x,y=model(input1)
    x=1
    # model = Generator_xx(1, 1)