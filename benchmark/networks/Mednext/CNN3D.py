import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from monai.networks.blocks.dynunet_block import UnetOutBlock

from .blocks import *

class LesionClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=6):
        super(LesionClassifier, self).__init__()
        hidden_size=2048
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels,hidden_size),
            nn.ReLU(),  # 添加激活函数，如ReLU，以增加非线性
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x=self.classifier(x)
        return x

class LesionClassifier1(nn.Module):
    def __init__(self, input_nc, num_classes):
        super(LesionClassifier1, self).__init__()

        # 第一部分：3D卷积层块
        self.conv1 = nn.Conv3d(input_nc, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        # 全连接层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), inplace=True)

        # 平展3D特征图到一维向量
        # x = x.view(-1, 128 * (4 // 8) ** 3)
        # out = self.fc(x)
        x = self.classifier(x)
        return x

class CNN3D(nn.Module):
    def __init__(self, in_channels, out_channels,num_classes):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = nn.Conv3d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Conv3d(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels*4)
        self.pool3 = nn.MaxPool3d(kernel_size=2)


        self.conv4 = nn.Conv3d(out_channels*4, out_channels*8, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(out_channels*8)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.classification_head = LesionClassifier(out_channels*8, num_classes)
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(out_channels * 6 * 20 * 20, 1536)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(1536, 512)  # 融合层，根据基础模型输出调整
        # self.fc2 = nn.Linear(512, num_classes)  # 融合层，根据基础模型输出调整

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x=self.pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.pool2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.pool3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        # x = self.pool4(x)
        x = self.relu(x)

        # x = self.flatten(x)
        # x = self.fc(x)
        # x = self.relu(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.classification_head(x)
        return x



if __name__ == "__main__":
    network = MedNeXt(
        in_channels=1,
        n_channels=32,
        n_classes=13,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
        # exp_r = 2,
        kernel_size=3,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=True,
        # block_counts = [2,2,2,2,2,2,2,2,2],
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style=None,
        dim='2d',
        grn=True

    ).cuda()


    # network = MedNeXt_RegularUpDown(
    #         in_channels = 1,
    #         n_channels = 32,
    #         n_classes = 13,
    #         exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
    #         kernel_size=3,                     # Can test kernel_size
    #         deep_supervision=True,             # Can be used to test deep supervision
    #         do_res=True,                      # Can be used to individually test residual connection
    #         block_counts = [2,2,2,2,2,2,2,2,2],
    #
    #     ).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1, 1, 64, 64, 64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())

    with torch.no_grad():
        print(network)
        x = torch.zeros((1, 1, 128, 128, 128)).cuda()
        print(network(x)[0].shape)
