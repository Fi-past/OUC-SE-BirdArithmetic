from getDatas import *
from modelNet import getImageNet

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
from torch.utils.data import DataLoader
import torchvision

# def getImgPath(imgPath):
#     imgs=[]
#     for imgName in os.listdir(imgPath):
#         imgs.append(imgName)
#     return imgs

preprocessVal_Test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    ),
])

if __name__ == "__main__":

    imgTestPath='../data/bird/test'
    path='../model/bird_v1_densenet121_91_66.pt'

    # img_names=getImgPath(imgTestPath)

    test_dataset =torchvision.datasets.ImageFolder(imgTestPath,transform=preprocessVal_Test)
    test_loader =DataLoader(test_dataset,batch_size=1, shuffle=False,num_workers=4)
    
    denseNetModel = torch.load(path)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss() 
    # 优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, denseNetModel.parameters()),  lr=0.001)
    # 准备模型训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    denseNetModel=denseNetModel.to(device)

    # 进行测试
    denseNetModel.eval()
    img_name_list=[]
    result_list=[]

    # pa=10
    with torch.no_grad():
        for i,(data,_) in enumerate(test_loader):
            data=data.to(device)
            one_img_path, _ = test_loader.dataset.samples[i]
            one_img_path=one_img_path.rsplit('\\',1)[1]
            # sample_fname, _ = data.samples
            # print(one_img_path)
            

            output = denseNetModel(data)
            prediction = torch.argmax(output, 1).item()+1

            img_name_list.append(one_img_path)
            result_list.append(prediction)

            # pa-=1
            # if pa<0:
            #     break

            # print(prediction)
            # batchCorrect = (prediction == target).sum().float()
            # batchSize=len(target)
    dictionary = dict(zip(img_name_list, result_list))
    # dictionary=sorted(dictionary.items(),key=lambda item:sorted(item[0],key=lambda x:int(x.split('.')[0])))
    dictionary=sorted(dictionary.items(),key=lambda item:int(item[0].split('.')[0]))
    # print(dictionary)
    with open("../data/submitAi/key.csv", 'w') as f:
      for line in dictionary:
          f.write("{},{}\n".format(line[0], line[1]))