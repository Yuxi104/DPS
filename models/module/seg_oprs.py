import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os.path as osp
BatchNorm2d = nn.BatchNorm2d

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=1, Fch=16, scale=4, branch=2, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

    def forward(self, fm):
        # fm is already a concatenation of multiple scales
        fm = self.conv_1x1(fm)
        return fm
        # fm_se = self.channel_attention(fm)
        # output = fm + fm * fm_se
        # return output


class Head(nn.Module):
    def __init__(self, in_planes, out_planes=19, Fch=16, scale=4, branch=2, is_aux=False, norm_layer=nn.BatchNorm2d):
        super(Head, self).__init__()
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            if is_aux:
                mid_planes = in_planes
            else:
                mid_planes = in_planes
        else:
            # in_planes > 256:
            if is_aux:
                mid_planes = in_planes // 2
            else:
                mid_planes = in_planes // 2
        self.conv_3x3 = ConvBnRelu(in_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_1x1 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        return output


class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native nn.Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, width_mult_list=[1.]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        self.conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias),
            # nn.BatchNorm2d(C_out),
            BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        assert x.size()[1] == self.C_in, "{} {}".format(x.size()[1], self.C_in)
        x = self.conv(x)
        return x


class BasicResidual1x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, width_mult_list=[1.]):
        super(BasicResidual1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        # self.bn1 = nn.BatchNorm2d(C_out)
        self.bn1 = BatchNorm2d(C_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_1x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, width_mult_list=[1.]):
        super(BasicResidual_downup_1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
        # self.bn1 = nn.BatchNorm2d(C_out)
        self.bn1 = BatchNorm2d(C_out)


    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out = self.relu(out)
        return out


class BasicResidual2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, width_mult_list=[1.]):
        super(BasicResidual2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        # self.bn1 = nn.BatchNorm2d(C_out)
        self.bn1 = BatchNorm2d(C_out)
        self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
        # self.bn2 = nn.BatchNorm2d(C_out)
        self.bn2 = BatchNorm2d(C_out)
    

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, width_mult_list=[1.]):
        super(BasicResidual_downup_2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
        # self.bn1 = nn.BatchNorm2d(C_out)
        self.bn1 = BatchNorm2d(C_out)
        self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
        # self.bn2 = nn.BatchNorm2d(C_out)
        self.bn2 = BatchNorm2d(C_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out = self.relu(out)
        return out


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride=1, width_mult_list=[1.]):
        super(FactorizedReduce, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
        if stride == 2:
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
            self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
            self.bn = BatchNorm2d(C_out)


    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1)
            out = self.bn(out)
            out = self.relu(out)
            return out
        else:
            return x