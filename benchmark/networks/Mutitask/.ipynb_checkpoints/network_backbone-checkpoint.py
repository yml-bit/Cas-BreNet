from monai.networks.nets import BasicUNet
import torch.nn as nn
from torchvision.models.video.resnet import r3d_18 as resnet18
# from torchvision.models import resnet18

class SharedEncoder(nn.Module):
    def __init__(self, in_channels=1, base_num_features=32):
        super(SharedEncoder, self).__init__()
        self.encoder = BasicUNet(in_channels=in_channels, out_channels=base_num_features)

    def forward(self, x):
        return self.encoder(x)

# class SharedEncoder(nn.Module):
#     def __init__(self, in_channels=1, base_model=resnet18):
#         super(SharedEncoder, self).__init__()
#
#         self.base_model = base_model(pretrained=True)
#         self.base_model.stem[0] = nn.Conv3d(in_channels=in_channels,
#                                           out_channels=64,
#                                           kernel_size=(7, 7, 7),
#                                           stride=(2, 2, 2),
#                                           padding=(3, 3, 3),
#                                           bias=False)
#
#         # Remove the last pooling and fully connected layers
#         modules = list(self.base_model.children())[:-2]
#
#         self.encoder = nn.Sequential(*modules)
#
#     def forward(self, x):
#         return self.encoder(x)

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super(Conv3DBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        if activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if hasattr(self, 'relu'):
            x = self.relu(x)
        return x

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super(SegmentationDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()  # Assuming binary segmentation (lesion vs background)
        )

    def forward(self, x):
        return self.decoder(x)

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
        x=self.classifier(x)
        return x

class unet(nn.Module):
    def __init__(self, in_channels=1, seg_out_channels=2, class_num=12, base_num_features=32):
        super(unet, self).__init__()
        self.shared_encoder = SharedEncoder(in_channels, base_num_features)#in_channels, base_num_features
        self.segmentation_decoder = SegmentationDecoder(base_num_features, seg_out_channels)
        self.classification_head = LesionClassifier(base_num_features, class_num)

    def forward(self, x):
        shared_features = self.shared_encoder(x)
        seg_output = self.segmentation_decoder(shared_features)
        class_output = self.classification_head(shared_features)
        return seg_output, class_output.squeeze(dim=-1)