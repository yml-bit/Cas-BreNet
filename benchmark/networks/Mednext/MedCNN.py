import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from monai.networks.blocks.dynunet_block import UnetOutBlock

from .blocks import *

class Base3DCNN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Base3DCNN, self).__init__()
        self.dense1 = MedNeXtBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                exp_r=2,
                kernel_size=3,
                do_res=True,
                norm_type='group',
                dim='3d',
                grn=False
            )
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x=self.dense1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class fist_layer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(fist_layer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class final_layer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(final_layer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * 6 * 12 * 12, out_channels*2)  # 调整线性层的输入大小以匹配最小尺度输出
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(out_channels*2, out_channels)  #
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        x = self.relu(x)
        return x

class MedCNN(nn.Module):
    def __init__(self, num_classes):
        super(MedCNN, self).__init__()
        in_channels=1
        n_channels=32
        self.stem1 = fist_layer(in_channels,n_channels)
        self.base_cnn1 = Base3DCNN(n_channels,n_channels*2)
        self.base_cnn2 = Base3DCNN(n_channels*2, n_channels*4)
        self.base_cnn3 = Base3DCNN(n_channels*4, n_channels*8)
        self.final1=final_layer(n_channels*8,1536)

        self.fusion_fc1 = nn.Linear(1536, 512)  # 融合层，根据基础模型输出调整
        self.fusion_fc2 = nn.Linear(512, num_classes)  # 融合层，根据基础模型输出调整
    def forward(self, patch1):
        x=self.stem1(patch1)
        x=self.base_cnn1(x)
        x=self.base_cnn2(x)
        x=self.base_cnn3(x)
        x=self.final1(x)
        output1 = self.fusion_fc1(x)
        output = self.fusion_fc2(output1)
        return output
