from torch import nn
import torch


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x

class CHAM(nn.Module):

    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CHAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        inputs = x
        x = self.channelattention(x)

        out = (1 - self.alpha) * x + self.alpha * inputs

        return out
