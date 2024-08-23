from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union

__all__ = ["SegResNet", "CrossSeg"]

class SegResNet(nn.Module):
    """
    SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module does not include the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )#using (3x3x1),使用更多的残差块
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        # if self.use_conv_final:
        #     x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x

class CrossSeg(SegResNet):
    """
    SegResNetVAE based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module contains the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Args:
        input_image_size: the size of images to input into the network. It is used to
            determine the in_features of the fc layer in VAE.
        vae_estimate_std: whether to estimate the standard deviations in VAE. Defaults to ``False``.
        vae_default_std: if not to estimate the std, use the default value. Defaults to 0.3.
        vae_nz: number of latent variables in VAE. Defaults to 256.
            Where, 128 to represent mean, and 128 to represent std.
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.
    """

    def __init__(
        self,
        input_image_size: Sequence[int],
        spatial_dims: int = 3,
        init_filters: int = 64,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        act: Union[str, tuple] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        norm_name: Union[Tuple, str] = "instance",
        res_block: bool = True,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            use_conv_final=use_conv_final,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode=upsample_mode,
        )
        self.out_chans = out_channels
        self.input_image_size = input_image_size
        self.smallest_filters = 16

        zoom = 2 ** (len(self.blocks_down) - 1)
        self.fc_insize = [s // (2 * zoom) for s in self.input_image_size]

        self.feat_size = feat_size
        self.branch1a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0] // 6,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch1b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0] // 6,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2a = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0] // 3,
            out_channels=self.feat_size[0] // 3,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.branch2b = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0] // 3,
            out_channels=self.feat_size[0] // 3,
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

    def _prepare_vae_modules(self):
        zoom = 2 ** (len(self.blocks_down) - 1)
        v_filters = self.init_filters * zoom
        total_elements = int(self.smallest_filters * np.prod(self.fc_insize))

        self.vae_down = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, v_filters, self.smallest_filters, stride=2, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.smallest_filters),
            self.act_mod,
        )
        self.vae_fc1 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc2 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc3 = nn.Linear(self.vae_nz, total_elements)

        self.vae_fc_up_sample = nn.Sequential(
            get_conv_layer(self.spatial_dims, self.smallest_filters, v_filters, kernel_size=1),
            get_upsample_layer(self.spatial_dims, v_filters, upsample_mode=self.upsample_mode),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
        )

    def forward(self, x_in):
        x, down_x = self.encode(x_in)
        down_x.reverse()
        out = self.decode(x, down_x)

        b1a = self.branch1a(out)
        b1b = self.branch1b(out)
        repeat1 = x_in.repeat(1, 8, 1, 1, 1)
        Vessel1 = b1b - repeat1
        cta1 = b1a + repeat1

        b2a = self.branch2a(torch.cat((b1a, Vessel1), 1))
        b2b = self.branch2b(torch.cat((b1b, cta1), 1))
        repeat2 = x_in.repeat(1, 16, 1, 1, 1)
        Vessel2 = b2b - repeat2
        cta2 = b2a + repeat2

        b3a = self.branch3a(torch.cat((b2a, Vessel2), 1))
        b3b = self.branch3b(torch.cat((b2b, cta2), 1))
        repeat3 = x_in.repeat(1, 32, 1, 1, 1)
        Vessel3 = b3b - repeat3
        cta3 = b3a + repeat3

        outa = self.out1(torch.cat((b3a, Vessel3), 1))
        outb = self.out2(torch.cat((b3b, cta3), 1))

        # feat = self.conv_proj(dec4)
        return outa, outb  # (self.act(self.out2(out2))+1)/2
