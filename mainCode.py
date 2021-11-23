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


# 看看是不是样本不均衡的问题 在dataset里面看吧

if __name__ == "__main__":

    # 获取dataset
    imgTrainPath='../data/bird/train_set'
    imgValPath='../data/bird/val_set'
    txtClassPath='../data/bird/classes.txt'

    bird_data_Trainset=birdTrainDataSet(imgTrainPath,txtClassPath,True)
    bird_data_Valset=birdTrainDataSet(imgValPath,txtClassPath,False)

    # 获取dataloader
    trainloader = DataLoader(bird_data_Trainset,batch_size=180,shuffle=True,num_workers=6)
    valloader = DataLoader(bird_data_Valset,batch_size=180,shuffle=False,num_workers=6)

    # 获取预训练的denseNet模型
    denseNetModel = getImageNet()
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss() 
    # 优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, denseNetModel.parameters()),  lr=0.001)

    # 准备模型训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path='../model/bird_v1.pt'
    if os.path.exists(path):
        denseNetModel = torch.load(path)
        print('加载先前模型成功')
    else:
        print('未加载原有模型训练')

    epochNum=20
    index_to_classes=getClassFromTxt(txtClassPath)
    denseNetModel=denseNetModel.to(device)


    for epoch in range(epochNum):

        denseNetModel.train()
        train_loss=0
        train_correct,train_total=0,0

        for batch, (data, target) in enumerate(trainloader):
            data=data.to(device)
            target=target.to(device)

            optimizer.zero_grad()
            output = denseNetModel(data)

            prediction = torch.argmax(output, 1)

            batchCorrect = (prediction == target).sum().float()
            batchSize=len(target)

            train_correct += batchCorrect
            train_total += batchSize

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

            if batch%10==0:
                print(f'单batch：第{epoch+1}个epoch中第{batch}次迭代,训练集准确率为{100*batchCorrect/batchSize}%')
            
        print('-'*35)
        print(f'epoch：第{epoch+1}次迭代,训练集准确率为{100*train_correct/train_total}%,loss为{train_loss}')

        # 进行测试
        denseNetModel.eval()
        test_loss=0
        test_correct,test_total=0,0
        # 注意这个maxValAcc是在内存中的
        maxValAcc,valAcc=0,0

        with torch.no_grad():
            for batch, (data, target) in enumerate(valloader):
                data=data.to(device)
                target=target.to(device)

                output = denseNetModel(data)
                prediction = torch.argmax(output, 1)
                batchCorrect = (prediction == target).sum().float()
                batchSize=len(target)

                loss = criterion(output, target)
                test_loss+=loss
                test_correct += batchCorrect
                test_total += batchSize

        valAcc=100*test_correct/test_total
        print(f'epoch：第{epoch+1}次迭代,验证集准确率为{valAcc}%，loss为{test_loss}')
        print('-'*35)

        if valAcc > maxValAcc:
            maxValAcc=valAcc
            # 保存模型
            torch.save(denseNetModel, path)
            print()
            print('模型更新成功~')
            print()