from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
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
        x = self.conv(x)
        # 得到输出张量x，最后返回该张量
        return x

# 上采样卷积块
class up_conv(nn.Module):

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

class Attention_block(nn.Module):
    
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        # 定义 W_g，将低层特征 F_l 通过 1x1 卷积转换为 F_int 通道数，添加批标准化
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 定义 W_x，将高层特征 F_g 通过 1x1 卷积转换为 F_int 通道数，添加批标准化
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 定义 psi，将 W_g 和 W_x 的输出特征相加后通过 1x1 卷积压缩为一个通道，
        # 添加批标准化，再通过 Sigmoid 函数映射到 [0, 1] 之间，得到 attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), # 在每层输入之前对其进行标准化，可以加速网络训练，提高泛化性能
            nn.Sigmoid()
        )

        # 定义激活函数 ReLU
        # 将所有小于零的值变为零，大于等于零的值不变
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 通过 W_g 得到低层特征 g1，W_x 得到高层特征 x1
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 将 g1 和 x1 按元素相加并经过激活函数得到 attention map
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # 将输入 x 乘以 attention map 得到加权后的输出
        out = x * psi
        return out


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # print(f'x.shape:{x.shape}') [1, 3, 256, 256]
        # 编码器部分（下采样）
        e1 = self.Conv1(x)
        # print(f'e1.shape:{e1.shape}') [1, 64, 256, 256]
        # print(e5.shape) 
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        # print(f'e2.shape:{e2.shape}') [1, 128, 128, 128]

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [1, 1024, 16, 16]

        # 解码器部分（上采样）

        d5 = self.Up5(e5)
        # print(d5.shape) [1, 512, 32, 32]
        # print(e4.shape) [1, 512, 32, 32]
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out
