from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from torchsummary import summary

# 一个继承了nn.Module的卷积块类conv_block的定义，用于构建神经网络中的卷积操作
# 该类包含了一个卷积神经网络，由两个卷积层和两个批量标准化层组成。
# 具体来说，该卷积块接受一个输入张量x
# 然后在其上应用两个3x3大小的卷积层
# 每个卷积层之后都有一个批量标准化层和ReLU激活函数
# 最终，该卷积块返回一个输出张量
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        # 一个包含两个卷积层的Sequential对象
        # 两个卷积层、两个批归一化层和两个ReLU激活函数
        # in_ch代表输入的通道数
        # out_ch代表输出的通道数
        self.conv = nn.Sequential(
            # 使用了一个nn.Conv2d函数，
            # 定义了一个包含out_ch个输出通道，
            # 3x3的卷积核大小，步长为1，
            # padding为1的卷积操作。
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d函数进行批归一化操作
            nn.BatchNorm2d(out_ch),
            # ReLU激活函数
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    # 实现了卷积块的前向传递
    def forward(self, x):

        # 将输入张量x传入卷积块中的Sequential对象
        x = self.conv(x)
        # 得到输出张量x，最后返回该张量
        return x

# 上采样卷积块
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    # 输入通道数in_ch和输出通道数out_ch
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()

        # 使用 nn.Sequential 构建了一个上采样的操作序列
        # 包括上采样、卷积、批归一化和 ReLU 激活函数
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    # 在 forward 函数中，输入一个张量，进行上采样卷积操作，并返回输出张量。
    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        # 表示 U-Net 网络中第一层卷积核的数量为 64
        n1 = 64
        # filters包含了5个整数，分别是n1、n1*2、n1*4、n1*8、n1*16
        # 这些整数用于定义U-Net模型中的特征通道数，第一层特征通道数为n1
        # 每经过一层下采样，特征通道数翻倍
        # 每经过一层上采样，特征通道数减半
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        # 定义了4个最大池化层，用于对输入特征图进行下采样操作，即将特征图的分辨率降低
        # 每个最大池化层的kernel_size为2（卷积大小为2*2），stride为2(卷积核每次移动两个像素进行卷积)
        # 表示在2x2的窗口内取最大值，然后向下移动2个像素进行采样，这样可以将特征图的大小减半
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义了5个卷积块
        # 包括卷积层、批量归一化（batch normalization）和激活函数（ReLU）
        # 输入的特征通道数从较低的数量逐渐增加到较高的数量
        # 具体来说，输入的图像（或特征图）通过Conv1卷积块得到一组特征图，再通过Maxpool1池化层进行下采样；接下来，这组特征图通过Conv2卷积块得到一组更多的特征图，再通过Maxpool2池化层进行下采样；这样的过程一直持续到Conv5卷积块得到一组最丰富的特征图，然后开始通过上采样和跳跃式连接来进行解码操作。
        
        # (3,64)
        self.Conv1 = conv_block(in_ch, filters[0])
        # (64,128)
        self.Conv2 = conv_block(filters[0], filters[1])
        # (128,256)
        self.Conv3 = conv_block(filters[1], filters[2])
        # (256,512)
        self.Conv4 = conv_block(filters[2], filters[3])
        # (512,1024)
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out