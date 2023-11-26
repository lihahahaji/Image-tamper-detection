import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
import numpy as np

from  torch import nn,optim
import os
from data import MyDataset

# 导入Unet网络
from unet import U_Net
from att_unet import AttU_Net

# 设置训练设备

if(torch.cuda.is_available()):
    print("cuda is available.")
else:
    print("cuda is not available.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'

batch_size = 4  # 批大小，即每个训练迭代中处理的样本数量
epoch = 500     # 训练轮数
shuffle =True
lr = 0.01

# 设置数据集的存储路径
t_data = "C:\\Users\\lihaji\\Desktop\\Image_check\\DataSet\\CASIA\\train_test\\train"

# 设置模型权重的存储路径
weight_path = 'C:\\Users\\lihaji\\Desktop\\Image_check\\params\\model.pth'

# 训练过程中生成图片的存储路径
save_path = 'C:\\Users\\lihaji\\Desktop\\Image_check\\gen_images\\'

# 设置 Dataloder
data_loader = DataLoader(MyDataset(t_data),batch_size,shuffle)

# 初始化模型
model = AttU_Net(3,1)
model.to(device)

# 加载权重数据
if os.path.exists(weight_path) :
    model.load_state_dict(torch.load(weight_path))
    print("发现权重数据，已加载.")
else : 
    print("未找到权重数据.")

# 设置优化器
opt = optim.Adam(model.parameters(),lr)
# 初始化学习率调整器
scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

loss_fun = nn.BCELoss()

# 设置训练损失
min_train_loss = np.Inf
running_loss = 0.0
train_length = 0
haji_Loss = np.inf

# 开始训练
print("开始训练.")
for ep in range(epoch): 

    running_loss = 0.0
    train_length = 1

    print(f'epoch = {ep}/{epoch}, training...')

    for i,(image,image_label) in enumerate(tqdm.tqdm(data_loader)):

        # 获取输入的数据和标签
        image = image.to(device)
        image_label = image_label.to(device)

        # 将梯度清零
        opt.zero_grad()

        # 前向传播
        out_image = model(image)

        # 计算损失
        s_out_image = torch.sigmoid(out_image)
        s_image_label = torch.sigmoid(image_label)
        train_loss = loss_fun(s_out_image,s_image_label)

        # 反向传播
        train_loss.backward()

        # 更新权重
        opt.step()

        # print(train_loss.item())
        if(haji_Loss>train_loss.item()):
            haji_Loss = train_loss.item()
        running_loss += train_loss.item()*image.size(0)
        train_length += 1

        # 选取第一张图片输出
        _image = image[0]
        _image_l = image_label[0]
        _out_image = out_image[0]
        _out_image = torch.cat([_out_image]*3, dim=0)
        # 图像拼接
        img= torch.stack([_image,_out_image],dim=0)

        # 保存生成的图片
        # save_image(img,f'{save_path}/image_epoch{ep}-{i}.png')

    scheduler.step()

    epoch_loss = running_loss / train_length
    if(epoch_loss < min_train_loss):
        print(f'train loss update: {min_train_loss} to {epoch_loss}')
        min_train_loss = epoch_loss
        print('save model')
        # 保存当前的权重
        torch.save(model.state_dict(),weight_path)

    
    print(f'epoch:{ep}-trainloss : {min_train_loss}')
    print(f'epochMInLoss:{haji_Loss}')
