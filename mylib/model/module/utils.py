import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

def kaiming_init(m):
    # Weight initial method, more detial search for torch.nn.init.kaiming_normal_
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    # Weight initial method, more detial search for torch.nn.init.normal_
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias= True)

class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual # short cut

        return out

class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn = False):
        super(_bn_relu_conv, self).__init__()
        self.has_bn = bn
        #self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        #if self.has_bn:
        #    x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x