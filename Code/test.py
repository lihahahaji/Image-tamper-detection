import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

from  torch import nn,optim
import os
from data import MyDataset

# 导入Unet网络
from unet import U_Net
from att_unet import AttU_Net



def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)

f = open('log.txt', 'w')

# 加载测试数据
test_data = "/Users/lihaji/Desktop/Image_check/DataSet/CASIA/train_test/test"
data_loader = DataLoader(MyDataset(test_data),1,False)

# 设置模型权重的存储路径
weight_path = '/Users/lihaji/Desktop/Image_check/params/model.pth'
res_path = '/Users/lihaji/Desktop/Image_check/test_result'

# 加载权重
model = AttU_Net(3,1)
if os.path.exists(weight_path) :
    model.load_state_dict(torch.load(weight_path,map_location='cpu'))
    print("发现权重数据，已加载.")
    f.write('发现权重数据，已加载.')
else : 
    print("未找到权重数据.")
    f.write('发现权重数据，已加载.')



# torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model.to(device)
torch.no_grad()

model.eval()
correct = 0
total = 0
dice_score = 0.0

y_true = []
y_pred = []

out_num = 1
for i,(image,image_label) in enumerate(tqdm.tqdm(data_loader)):
    image = image.to(device)
    image_label = image_label.to(device)
    out_image = model(image)

    outputs = torch.sigmoid(out_image)
    outputs[outputs > 0.5] = 1
    outputs[outputs <= 0.5] = 0

    y_true.extend(image_label.cpu().detach().numpy().flatten())
    y_pred.extend(outputs.cpu().detach().numpy().flatten())

    

    haji_dice = dice_coefficient(out_image.float(), image_label.float())
    dice_score += haji_dice

    predicted = torch.round(out_image)
    total += image_label.numel()
    correct += (predicted == image_label).sum().item()

    _image = image[0]
    _image_l = image_label[0]
    _out_image = out_image[0]
    

    _out_image = torch.cat([_out_image]*3, dim=0)
    _image_l = torch.cat([_image_l]*3, dim=0)
    img= torch.stack([_image,_image_l,_out_image],dim=0)

    # save_image(img,f'{res_path}/image_result-{i}.png')
    
    if(haji_dice > 0.5):
        # print(f'dice-{i}:{haji_dice}')
        save_image(img,f'{res_path}/image_result-{out_num}.png')
        out_num +=1

# dice_score /= len(data_loader)
# print(f'dice_res:{dice_score}')

# precision = precision_score(y_true, y_pred, average='binary')
# recall = recall_score(y_true, y_pred, average='binary')
# f1 = f1_score(y_true, y_pred, average='binary')

# print(f'precision:{precision}\nrecall:{recall}\nf1:{f1}\n')


# accuracy = 100 * correct / total
# print(f'Test Accuracy: {accuracy:.2f}%')