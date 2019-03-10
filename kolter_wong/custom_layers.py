import torch
import math
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair


class Conv2dUntiedBias(nn.Module):
    def __init__(self, height, width, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels, height, width))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = F.conv2d(input, self.weight, None, self.stride,
                          self.padding, self.dilation, self.groups)
        # add untied bias
        output += self.bias  # Explicit broadcasting is unnecessary: .unsqueeze(0).repeat(input.size(0), 1, 1, 1)
        return output
