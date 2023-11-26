import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
from torch.nn.functional import one_hot

transform_RGB = transforms.Compose([
    transforms.ToTensor(),
    # 对数据进行归一化
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform = transforms.Compose([
    transforms.ToTensor(),
    # 对数据进行归一化
    # transforms.Normalize(mean=[0.5], std=[0.5])
])



class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # self.name 包含gt目录下所有文件的文件名，是一个列表
        self.name = os.listdir(os.path.join(path, 'gt'))

    def __len__(self):
        # 返回文件的长度（标签文件）
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png

        # 标签和对应图片的路径
        segment_path = os.path.join(self.path, 'gt', segment_name)
        image_path = os.path.join(self.path, 'tp', segment_name)

        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open_rgb(image_path)
        return transform_RGB(image), transform(segment_image)


class DatasetForRun(Dataset):
    def __init__(self, path):
        self.path = path
        # self.name 包含gt目录下所有文件的文件名，是一个列表
        self.name = os.listdir(os.path.join(path))

    def __len__(self):
        # 返回文件的长度（标签文件）
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png

        # 标签和对应图片的路径
        image_path = os.path.join(self.path, segment_name)

        image = keep_image_size_open_rgb(image_path)
        return transform_RGB(image)


if __name__ == '__main__':

    data = MyDataset('data')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)
