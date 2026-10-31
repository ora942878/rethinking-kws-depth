"""PyTorch port of the classification path in hyperconnect/TC-ResNet.

Official layout: stem16 -> 24,24,32,32,48,48 -> pool -> dropout -> classifier.
TensorFlow SAME padding, projection ReLU and Slim BN eps/decay are preserved.
The unused auxiliary two-output ranges head is omitted.
Source: https://github.com/hyperconnect/TC-ResNet/blob/master/audio_nets/tc_resnet.py
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,bias=False)

    def forward(self,x):
        total=max((math.ceil(x.shape[-1]/self.stride)-1)*self.stride+self.kernel_size-x.shape[-1],0)
        return self.conv(F.pad(x,(total//2,total-total//2)))


def conv_bn(in_channels,out_channels,kernel,stride=1,activate=True):
    layers=[SameConv(in_channels,out_channels,kernel,stride),
            nn.BatchNorm1d(out_channels,eps=1e-3,momentum=0.003)]
    if activate: layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class TCBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.transition=in_channels!=out_channels
        stride=2 if self.transition else 1
        self.conv1=conv_bn(in_channels,out_channels,9,stride)
        self.conv2=conv_bn(out_channels,out_channels,9,activate=False)
        # The upstream Slim projection uses the default BN + ReLU activation.
        self.shortcut=conv_bn(in_channels,out_channels,1,2) if self.transition else nn.Identity()

    def forward(self,x):
        return F.relu(self.conv2(self.conv1(x))+self.shortcut(x))


class TCResNet14(nn.Module):
    def __init__(self,num_classes=12):
        super().__init__()
        channels=[16,24,24,32,32,48,48]
        self.stem=conv_bn(40,16,3)
        self.blocks=nn.ModuleList([TCBlock(a,b) for a,b in zip(channels,channels[1:])])
        self.dropout=nn.Dropout(0.5)
        self.classifier=nn.Linear(48,num_classes,bias=False)
        for m in self.modules():
            if isinstance(m,(nn.Conv1d,nn.Linear)): nn.init.xavier_uniform_(m.weight)

    def features(self,x,layer_list=None):
        x=self.stem(x)
        for i in (range(1,7) if layer_list is None else layer_list): x=self.blocks[i-1](x)
        return x.mean(-1)

    def forward(self,x,layer_list=None):
        return self.classifier(self.dropout(self.features(x,layer_list)))
