import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os
import torchvision as tv
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models

# 注意这里一定要用cpu

# 从txt中获取类型
def getClassFromTxt(txtPath):
    index_classes={}
    with open(txtPath, 'r', encoding='utf-8') as f:
        txts=f.readlines()
        txts=[t[:-1] for t in txts]
        for t in txts:
            label,name=t.split(' ')[1].split('.')
            index_classes[int(label)-1]=name
    return index_classes


def getResult(txtClassPath,modelPath,path):
    # 测试验证集预处理
    preprocessVal_Test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]
        ),
    ])

    with torch.no_grad():
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # txtClassPath='../data/bird/classes.txt'
        # path='../data/test/011.Rusty_Blackbird_242.jpg'
        # modelPath='../model/bird_v1_densenet121_76_58.pt'

        img =  Image.open(path).convert('RGB')
        img_tensor = preprocessVal_Test(img)
        img_tensor=torch.unsqueeze(img_tensor,0)
        # print(img_tensor.size())

        # img_tensor=img_tensor.to(device)
        # img_tensor=img_tensor.expand(64,3,224,224)
        # print(img_tensor.size())

        # denseNetModel = torch.load(modelPath)
        denseNetModel = torch.load(modelPath,map_location='cpu')
        denseNetModel.eval()
        index_to_classes=getClassFromTxt(txtClassPath)

        output = denseNetModel(img_tensor)
        prediction = torch.argmax(output, 1)

        # 转化为概率
        output=F.softmax(output,dim=1)
        values, indices = torch.topk(output, k=5, dim=1, largest=True, sorted=True)

        np.set_printoptions(suppress=True)
        values=values.squeeze().numpy()*100
        indices=indices.squeeze().numpy()
        classes=[index_to_classes.get(ind) for ind in indices]
        # print("values: ", values)
        # print("indices: ", indices)
        # print("classes: ", classes)
        for i in range(5):
            print(f'对{classes[i]}识别的置信程度为{values[i]}%')
        # result=[]
        # for i in range(5):
        #     # print(f'对{classes[i]}识别的置信程度为{values[i]}%')
        #     result.append(f'对{classes[i]}识别的置信程度为{values[i]}%')
        # print(result)
        # print(prediction)
        # name=index_to_classes.get(prediction.item())
        # print(name)

txtClassPath='../data/bird/classes.txt'
path='../data/test/011.Rusty_Blackbird_242.jpg'
modelPath='../model/bird_v1_densenet121_91_66.pt'
getResult(txtClassPath,modelPath,path)
