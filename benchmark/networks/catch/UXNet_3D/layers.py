from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

scale_eval = False

alpha = 0.02
beta = 0.00002

resnet_n_blocks = 1

norm = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
norm3D = partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
align_corners = False
up_sample_mode = 'bilinear'

def custom_init(m):
    m.data.normal_(0.0, alpha)

def get_init_function(activation, init_function, **kwargs):
    """Get the initialization function from the given name."""
    a = 0.0
    if activation == 'leaky_relu':
        a = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']

    gain = 0.02 if 'gain' not in kwargs else kwargs['gain']
    if isinstance(init_function, str):
        if init_function == 'kaiming':
            activation = 'relu' if activation is None else activation
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation, mode='fan_in')
        elif init_function == 'dirac':
            return torch.nn.init.dirac_
        elif init_function == 'xavier':
            activation = 'relu' if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function == 'normal':
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function == 'orthogonal':
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function == 'zeros':
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_function is None:
        if activation in ['relu', 'leaky_relu']:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ['tanh', 'sigmoid']:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function

def get_activation(activation, **kwargs):
    """Get the appropriate activation from the given name"""
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky_relu':
        negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return None

class Conv3D(torch.nn.Module):
    """Defines a basic convolution layer.
    The general structure is as follow:

    Conv -> Norm (optional) -> Activation -----------> + --> Output
                                         |            ^
                                         |__ResBlcok__| (optional)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, **kwargs):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.resnet_block = ResnetTransformer3D(out_channels, resnet_n_blocks, init_func) if use_resnet else None
        self.norm = norm3D(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)
        # Initialize the weights
        init_ = get_init_function(activation, init_func)
        init_(self.conv3d.weight)
        if self.conv3d.bias is not None:
            self.conv3d.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm3d):
            nn.init.normal_(self.norm.weight.data, 0.0, 1.0)
            nn.init.constant_(self.norm.bias.data, 0.0)

    def forward(self, x):
        x = self.conv3d(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x

class DownBlock3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, skip=True, refine=False, pool=True,
                 pool_size=(2,2,1), **kwargs):
        super(DownBlock3D, self).__init__()
        self.conv_0 = Conv3D(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                           activation=activation, init_func=init_func, use_norm=use_norm, callback=None,
                           use_resnet=use_resnet, **kwargs)
        self.conv_1 = None
        if refine:
            self.conv_1 = Conv3D(out_channels, out_channels, kernel_size, stride, padding, bias=bias,
                               activation=activation, init_func=init_func, use_norm=use_norm, callback=None,
                               use_resnet=use_resnet, **kwargs)
        self.skip = skip
        self.pool = None
        if pool:
            self.pool = nn.MaxPool3d(kernel_size=pool_size)

    def forward(self, x):
        x = skip = self.conv_0(x)
        if self.conv_1 is not None:
            x = skip = self.conv_1(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.skip:
            return x, skip
        else:
            return x

class ResnetTransformer3D(torch.nn.Module):
    def __init__(self, dim, n_blocks, init_func):
        super(ResnetTransformer3D, self).__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock3D(dim, padding_type='replicate', norm3D=norm3D, use_dropout=False,
                            use_bias=True)]
        self.model = nn.Sequential(*model)

        init_ = get_init_function('relu', init_func)

        def init_weights(m):
            if type(m) == nn.Conv3d:
                init_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if type(m) == nn.BatchNorm3d:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class ResnetBlock3D(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm3D, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm3D, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm3D, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm3D          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm3D(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm3D(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out