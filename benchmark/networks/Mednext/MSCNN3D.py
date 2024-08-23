import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.denseblock import DenseBlock,ConvDenseBlock
from monai.networks.blocks import Convolution

from .blocks import *

class LesionClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=6):
        super(LesionClassifier, self).__init__()
        hidden_size=1024#2048
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels,hidden_size),
            # nn.ReLU(),  # 添加激活函数，如ReLU，以增加非线性
            # nn.Dropout(p=0.5),
            # nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x=self.classifier(x)
        return x

class Base3DCNN1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Base3DCNN1, self).__init__()
        growth_rate = in_channels  # 每个DenseLayer输出通道数
        num_layers = 4  # 每个DenseBlock的层数
        layers_in_first_block = [
            Convolution(spatial_dims=3,
                        in_channels=in_channels + growth_rate * i,
                        out_channels=in_channels,  # 修改为out_channels
                        kernel_size=3,
                        strides=1,
                        norm='batch')
            for i in range(num_layers)
        ]
        self.dense1 = DenseBlock(layers_in_first_block)
        self.conv1 = nn.Conv3d(in_channels*(num_layers+1), out_channels, kernel_size=3, stride=1, padding=1)
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

class Base3DCNN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Base3DCNN, self).__init__()
        self.b1 = MedNeXtBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                exp_r=2,
                kernel_size=3,
                do_res=True,
                norm_type='group',
                dim='3d',
                grn=False
            )
        self.b2 = MedNeXtDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=2,
            kernel_size=3,
            do_res=True,
            norm_type='group',
            dim='3d',
        )

    def forward(self, x):
        x=self.b1(x)
        x = self.b2(x)
        return x

def resize(patch):
    assert isinstance(patch, torch.Tensor), "Input must be a PyTorch tensor."
    resized_patch = F.interpolate(patch, size=(40, 128, 128), mode='nearest')
    return resized_patch

class fist_layer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(fist_layer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.GroupNorm(out_channels,out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x

class final_layer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(final_layer, self).__init__()
        # hidden_size = 2048
        self.avg=nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels, out_channels)  # 调整线性层的输入大小以匹配最小尺度输出
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc_out = nn.Linear(out_channels*2, out_channels)  #
    def forward(self,x):
        x=self.avg(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        # x = self.fc_out(x)
        # x = self.relu(x)
        return x

class MSCNN3D(nn.Module):
    def __init__(self, num_classes):
        super(MSCNN3D, self).__init__()
        in_channels=3
        n_channels=32
        final_size=1024
        self.stem1 = fist_layer(in_channels,n_channels)
        # self.final0 = final_layer(n_channels, final_size)
        self.final0 = LesionClassifier(n_channels, num_classes)
        self.base_cnn1 = Base3DCNN(n_channels,n_channels*2)
        self.base_cnn2 = Base3DCNN(n_channels*2, n_channels*4)
        self.base_cnn3 = Base3DCNN(n_channels*4, n_channels*8)
        # self.final1=final_layer(n_channels*8,final_size)
        self.final1 = LesionClassifier(n_channels*8, num_classes)

        self.stem11 = fist_layer(in_channels,n_channels)
        self.base_cnn11 = Base3DCNN(n_channels,n_channels*2)
        self.base_cnn22 = Base3DCNN(n_channels*2, n_channels*4)
        self.base_cnn33 = Base3DCNN(n_channels*4, n_channels*8)
        # self.final11 = final_layer(n_channels*8,final_size)
        self.final11 = LesionClassifier(n_channels * 8, num_classes)

        self.stem111 = fist_layer(in_channels,n_channels)
        self.base_cnn111 = Base3DCNN(n_channels,n_channels*2)
        self.base_cnn222 = Base3DCNN(n_channels*2, n_channels*4)
        self.base_cnn333 = Base3DCNN(n_channels*4, n_channels*8)
        # self.final111=final_layer(n_channels*8,final_size)
        self.final111 = LesionClassifier(n_channels * 8, num_classes)

        self.fusion_fc1 = nn.Linear(final_size*4, final_size*2)  # 融合层，根据基础模型输出调整
        self.dropout = nn.Dropout(p=0.5)
        self.fusion_fc2 = nn.Linear(final_size*2, num_classes)  # 融合层，根据基础模型输出调整
    def forward(self, patch):
        # patch1=resize(patch)
        # patch2=patch[:,:,6: -6, 24: -24, 24: -24]
        # patch3 = resize(patch[:,:,12: -12, 48:-48, 48:-48])
        patch2=patch[:,:,6: -6, 24: -24, 24: -24]
        patch3 = patch[:,:,12: -12, 48:-48, 48:-48]
        features = []
        x=self.stem1(patch)
        features.append(self.final0(x))
        x=self.base_cnn1(x)
        x=self.base_cnn2(x)
        x=self.base_cnn3(x)
        x=self.final1(x)
        features.append(x)

        x=self.stem11(patch2)
        x=self.base_cnn11(x)
        x=self.base_cnn22(x)
        x=self.base_cnn33(x)
        x=self.final11(x)
        features.append(x)

        x=self.stem111(patch3)
        x=self.base_cnn111(x)
        x=self.base_cnn222(x)
        x=self.base_cnn333(x)
        x=self.final111(x)
        features.append(x)
        # concatenated_features = torch.stack(features)
        # output=torch.mean(concatenated_features,dim=0)
        concatenated_features = torch.cat(features, dim=1)
        output1 = self.fusion_fc1(concatenated_features)
        output1=self.dropout(output1)
        output = self.fusion_fc2(output1)
        return output
