from monai.networks.nets import BasicUNet
import torch.nn as nn
from torchvision.models import resnet34

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.encoder = resnet34(pretrained=False)  # 不预训练，因为我们可能需要微调以适应MRI数据特性

        # 取消最后的全局平均池化层和全连接层
        modules = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels_encoder, num_classes_segmentation=2):
        super(SegmentationDecoder, self).__init__()

        self.up_convs = nn.ModuleList()
        for i in range(5):  # 假设我们有5个上采样阶段，根据实际需求调整
            up_conv = nn.ConvTranspose3d(in_channels=in_channels_encoder // (2 ** i),
                                         out_channels=in_channels_encoder // (2 ** (i + 1)),
                                         kernel_size=2,
                                         stride=2,
                                         padding=0)
            batch_norm = nn.BatchNorm3d(in_channels_encoder // (2 ** (i + 1)))
            relu = nn.ReLU(inplace=True)

            self.up_convs.append(nn.Sequential(up_conv, batch_norm, relu))

        self.final_conv = nn.Conv3d(in_channels=in_channels_encoder // (2 ** 5),
                                    out_channels=num_classes_segmentation,
                                    kernel_size=1)

    def forward(self, x):
        for up_conv in self.up_convs:
            x = up_conv(x)

        segmentation_output = self.final_conv(x)
        return segmentation_output

class LesionClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=12):
        super(LesionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            # nn.Linear(in_channels,1024),
            # nn.Dropout(p=0.5),
            nn.Linear(in_channels, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

class MultiTaskNetwork(nn.Module):
    def __init__(self, in_channels=1, seg_out_channels=2, class_num=1, base_num_features=32):
        super(MultiTaskNetwork, self).__init__()
        self.shared_encoder = SharedEncoder()
        self.segmentation_decoder = SegmentationDecoder(base_num_features, seg_out_channels)
        self.classification_head = LesionClassifier(base_num_features, class_num)

    def forward(self, x):
        shared_features = self.shared_encoder(x)
        seg_output = self.segmentation_decoder(shared_features)
        class_output = self.classification_head(shared_features)
        return seg_output, class_output