#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:04:06 2022

@author: leeh43
"""

from typing import Tuple
import torch.nn as nn
import functools
import torch
import numpy as np

import torch.nn.functional as F


class Expand(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unsqueeze(x, dim=0)

class BN_block3d(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.bn_block(x)

class downward_layer1(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels,n_convolutions):
        super().__init__()
        block=[]
        for _ in range(n_convolutions):
            block+=[nn.Conv3d(in_channels, out_channels//2, kernel_size=(3,3,3), padding=1),
                    # nn.InstanceNorm3d(out_channels),
                    nn.PReLU()]
        self.bn_block1 = nn.Sequential(*block)

        self.bn_block2=nn.Sequential(
            nn.Conv3d(out_channels//2, out_channels, kernel_size=(2,2,2),stride=(2,2,2), padding=1),
            nn.PReLU())

    def forward(self, x,y):
        add = self.bn_block1(x)+y
        down_data=self.bn_block2(add)
        return down_data,add

class downward_layer(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels,n_convolutions):
        super().__init__()
        block=[]
        for _ in range(n_convolutions):
            block+=[nn.Conv3d(in_channels, out_channels//2, kernel_size=(3,3,3), padding=1),
                    # nn.InstanceNorm3d(out_channels),
                    nn.PReLU()]
        self.bn_block1 = nn.Sequential(*block)

        self.bn_block2=nn.Sequential(
            nn.Conv3d(out_channels//2, out_channels, kernel_size=(2,2,2),stride=(2,2,2)),
            nn.PReLU())

    def forward(self, x,y):
        add = self.bn_block1(x)+y
        down_data=self.bn_block2(add)
        return down_data,add

class upward_layer1(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels,n_convolutions):
        super().__init__()
        block=[]
        for _ in range(n_convolutions):
            block+=[nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=1),
                    # nn.InstanceNorm3d(out_channels),
                    nn.PReLU()]
        self.bn_block1 = nn.Sequential(*block)

        self.bn_block2=nn.Sequential(
            nn.ConvTranspose3d(out_channels, out_channels,kernel_size=(2,2,2), stride=(2, 2, 2)),
                                 # nn.InstanceNorm3d(out_channels),
                                 nn.PReLU())

    def forward(self, x):
        add = self.bn_block1(x)+x
        down_data=self.bn_block2(add)
        return down_data

class upward_layer2(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels,n_convolutions):
        super().__init__()
        block=[]
        for _ in range(n_convolutions):
            block+=[nn.Conv3d(in_channels, out_channels*2, kernel_size=(3,3,3), padding=1),
                    # nn.InstanceNorm3d(out_channels),
                    nn.PReLU()]
            in_channels=out_channels*2
        self.bn_block1 = nn.Sequential(*block)

        self.bn_block2=nn.Sequential(
            nn.ConvTranspose3d(out_channels*2, out_channels,kernel_size=(2,2,2), stride=(2, 2, 2)),
                                 # nn.InstanceNorm3d(out_channels),
                                 nn.PReLU())

    def forward(self, x,y):
        add = self.bn_block1(torch.cat((x,y),1))
        down_data=self.bn_block2(add+x)
        return down_data

class out_layer(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_block=nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            # nn.InstanceNorm3d(out_channels),
            nn.PReLU())

    def forward(self, x,y):
        xx=torch.cat((x,y),1)
        add = self.bn_block(xx)
        add2=add+x
        return add2

class branch_layer(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels,n_convolutions):
        super().__init__()
        block=[]
        for _ in range(n_convolutions):
            block+=[nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                    # nn.InstanceNorm3d(out_channels),
                    nn.PReLU()]
            in_channels=out_channels
        self.bn_block1 = nn.Sequential(*block)

    def forward(self, x):
        out= self.bn_block1(x)
        return out

class outa(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels,n_convolutions):
        super().__init__()
        self.bn_block=nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
                    # nn.InstanceNorm3d(out_channels),
                    nn.PReLU())

    def forward(self, x):
        out= self.bn_block(x)
        # out=self.s(out)
        return out

class outb(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels,n_convolutions):
        super().__init__()
        self.bn_block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
                                      # nn.InstanceNorm3d(out_channels),
                                      nn.PReLU())

    def forward(self, x):
        out= self.bn_block(x)
        return out

class SVSUP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer= nn.InstanceNorm3d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # self.input1a = nn.Sequential(nn.Conv3d(in_channels, 16, kernel_size=3, padding=1,bias=use_bias),
        #          nn.PReLU(True)) #norm_layer(16),

        self.d1 = downward_layer(in_channels, 32,1)
        self.d2=downward_layer(32,64,2)
        self.u1 = upward_layer1(64, 64,3)
        self.u2=upward_layer2(64+32,32,2)
        self.out = out_layer(48, 32)

        self.b1a=branch_layer(32,8,3)
        self.b1b = branch_layer(32, 8, 3)
        self.b2a=branch_layer(16,16,2)
        self.b2b = branch_layer(16,16, 2)
        self.b3a=branch_layer(32,32,2)
        self.b3b = branch_layer(32, 32, 2)

        self.outa=outa(64,2,1)
        self.outb = outb(64, 1, 1)

    def forward(self, x):
        repeat=x.repeat(1,16,1,1,1)
        out1, left1 = self.d1(x,repeat)
        out2, left2 = self.d2(out1,out1)
        out3 = self.u1(out2)
        out4 = self.u2(out3,left2)
        out5=self.out(out4,left1)

        b1a=self.b1a(out5)
        b1b = self.b1b(out5)
        repeat1 = x.repeat(1, 8, 1, 1, 1)
        Vessel1=b1b-repeat1
        cta1=b1a+repeat1

        b2a=self.b2a(torch.cat((b1a,Vessel1),1))
        b2b = self.b2b(torch.cat((b1b,cta1),1))
        repeat2 = x.repeat(1, 16, 1, 1, 1)
        Vessel2=b2b-repeat2
        cta2=b2a+repeat2

        b3a=self.b3a(torch.cat((b2a,Vessel2),1))
        b3b = self.b3b(torch.cat((b2b,cta2),1))
        repeat3 = x.repeat(1, 32, 1, 1, 1)
        Vessel3=b3b-repeat3
        cta3=b3a+repeat3

        outa=self.outa(torch.cat((b3a,Vessel3),1))
        outb = self.outb(torch.cat((b3b,cta3),1))
        return outa,outb

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

if  __name__=='__main__':
    # input1=torch.Tensor(np.random.rand(1,1,64,64,64))#2,192,192,4
    # input2=torch.Tensor(np.random.rand(1,1,64,64,64))#2,192,192,4
    input1=torch.Tensor(np.random.rand(1,1,96,96,96))#2,192,192,4
    input2=torch.Tensor(np.random.rand(1,1,96,96,96))#2,192,192,4
    # a=torch.where((input1>0)&(input1<0.4),2,1)
    # b=1
    model = SVSUP(1,1)
    outa,outb=model(input1)
    x=1
    # model = Generator_xx(1, 1)