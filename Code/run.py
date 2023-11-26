import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm
import numpy as np
from utils import *
from torchvision import transforms

from sklearn.metrics import precision_score, recall_score, f1_score

from  torch import nn,optim
import os
from data import DatasetForRun

# 导入Unet网络
from unet import U_Net
from att_unet import AttU_Net

# 设置模型权重的存储路径
weight_path = '/Users/lihaji/Desktop/Image_check/params/model.pth'

# 加载权重
model = AttU_Net(3,1)
if os.path.exists(weight_path) :
    model.load_state_dict(torch.load(weight_path,map_location='cpu'))
    print("发现权重数据，已加载.")
    
else : 
    print("未找到权重数据.")
    

# torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model.to(device)
torch.no_grad()
model.eval()


input_path = '/Users/lihaji/Desktop/Image_check/input_image'
output_path = '/Users/lihaji/Desktop/Image_check/output_image'

data_loader = DataLoader(DatasetForRun(input_path),1,False)

for i,image in enumerate(tqdm.tqdm(data_loader)):
    image = image.to(device)
    out_image = model(image)

    _image = image[0]
    _out_image = out_image[0]
    _out_image = torch.cat([_out_image]*3, dim=0)
    img= torch.stack([_image,_out_image],dim=0)
    save_image(img,f'{output_path}/resImage-{i}.png')