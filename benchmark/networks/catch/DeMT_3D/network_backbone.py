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
from typing import Union
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from einops import rearrange
from .DHDC.ADHDC_Net import HDC_module,Conv_down,DHDC_module,conv_trans_block_3d,hd

from .sync_batchnorm import SynchronizedBatchNorm3d

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

class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class Conv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class DConv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation, d=1):
        super(DConv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(d, d, 0), dilation=d, bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

device1 = torch.device("cuda")

def hdc(image, num=2):
    x1 = torch.Tensor([]).to(device1) #
    for i in range(num):
        for j in range(num):
            for k in range(num):
                x3 = image[:, :, k::num, i::num, j::num]
                x1 = torch.cat((x1, x3), dim=1)
    return x1

class DeMT_seg(nn.Module):
    def __init__(self, in_chans=1, out_chans=5, num_filters=32):
        super(DeMT_seg, self).__init__()
        self.in_dim = in_chans
        self.out_dim = out_chans
        self.n_f = num_filters
        self.activation = nn.ReLU(inplace=False)
        # down

        self.conv_3x3x3 = Conv_3x3x3(8, self.n_f, self.activation)  #
        self.conv_1 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_1 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_2 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_2 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_3 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_3 = Conv_down(self.n_f, self.n_f, self.activation)
        # bridge
        self.bridge = DHDC_module(self.n_f, self.n_f, self.activation)
        # up
        self.up_1 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_4 = HDC_module(self.n_f * 2, self.n_f, self.activation)
        self.up_2 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_5 = HDC_module(self.n_f * 2, self.n_f, self.activation)
        self.up_3 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_6 = HDC_module(self.n_f * 2, self.n_f, self.activation)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.out = nn.Conv3d(self.n_f, 8, kernel_size=1, stride=1, padding=0)

        self.hd1 = hd(2, 1)
        self.hd2 = hd(4, 1)
        self.conv = Conv_3x3x3(8, self.out_dim, self.activation)#
        self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            # elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = hdc(x)
        # x=self.conv(x)
        x = self.conv_3x3x3(x)
        x1 = self.conv_1(x)
        x = self.down_1(x1)
        x2 = self.conv_2(x)
        x = self.down_2(x2)
        x3 = self.conv_3(x)
        x = self.down_3(x3)
        x = self.bridge(x)
        x = self.up_1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.conv_4(x)
        x = self.up_2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.conv_5(x)
        x = self.up_3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_6(x)
        x = self.upsample(x)
        x = self.out(x)

        s0, s1, s2, s3 = x.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1 = self.hd1(s1, s2)
        l2 = torch.cat([s1, s2], dim=1)
        l3 = self.hd2(s3, l2)
        y = torch.cat([s0, l1, s2, l3], dim=1)
        y=self.conv(y)
        y = self.softmax(y)
        return y

class DeMT(nn.Module):
    def __init__(self, in_chans=1, out_chans=5, num_filters=32):
        super(DeMT, self).__init__()
        self.in_dim = in_chans
        self.out_dim = out_chans
        self.n_f = num_filters
        self.activation = nn.ReLU(inplace=False)
        # down
        norm_name: Union[Tuple, str] = "instance"
        res_block: bool = True
        spatial_dims = 3

        self.conv_3x3x3 = Conv_3x3x3(8, self.n_f, self.activation)  #
        self.conv_1 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_1 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_2 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_2 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_3 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_3 = Conv_down(self.n_f, self.n_f, self.activation)
        # bridge
        self.bridge = DHDC_module(self.n_f, self.n_f, self.activation)
        # up
        self.up_1 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_4 = HDC_module(self.n_f * 2, self.n_f, self.activation)
        self.up_2 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_5 = HDC_module(self.n_f * 2, self.n_f, self.activation)
        self.up_3 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_6 = HDC_module(self.n_f * 2, self.n_f, self.activation)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.out = nn.Conv3d(self.n_f, 8, kernel_size=1, stride=1, padding=0)
        self.hd1 = hd(2, 1)
        self.hd2 = hd(4, 1)

        self.deca = Conv_3x3x3(8, 8, self.activation)#
        self.decb = Conv_3x3x3(8, 8, self.activation)  #
        self.hd1a = hd(2, 1)
        self.hd2a = hd(4, 2)

        # self.hd1b = hd(2, 1)
        # self.hd2b = hd(4, 1)
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

        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_dim)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=1)  # Act.PRELU   nn.Tanh()
        self.act = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_in):
        x = hdc(x_in)
        # x=self.conv(x)
        x = self.conv_3x3x3(x)
        x1 = self.conv_1(x)
        x = self.down_1(x1)
        x2 = self.conv_2(x)
        x = self.down_2(x2)
        x3 = self.conv_3(x)
        x = self.down_3(x3)
        x = self.bridge(x)
        x = self.up_1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.conv_4(x)
        x = self.up_2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.conv_5(x)
        x = self.up_3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_6(x)
        x = self.upsample(x)
        x = self.out(x)

        s0, s1, s2, s3 = x.chunk(4, dim=1)#torch.chunk(tensor, chunks, dim=0)。在给定维度(轴)上将输入张量进行分块儿
        l1 = self.hd1(s1, s2)
        l2 = torch.cat([s1, s2], dim=1)
        l3 = self.hd2(s3, l2)
        y = torch.cat([s0, l1, s2, l3], dim=1)
        outa=self.deca(y)
        outb = self.decb(y)

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
    model = DeMT_seg(1,2)
    x,y=model(input1)
    x=1
    # model = Generator_xx(1, 1)