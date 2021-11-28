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


if __name__ == "__main__":
    # 准备模型训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取dataset
    imgTrainPath='../data/bird/train_set'
    imgValPath='../data/bird/val_set'
    txtClassPath='../data/bird/classes.txt'

    bird_data_Trainset=birdTrainDataSet(imgTrainPath,txtClassPath,True)
    bird_data_Valset=birdTrainDataSet(imgValPath,txtClassPath,False)

    # weights=[]
    # for data, label in bird_data_Trainset:
    #     weights.append(bird_data_Trainset.class_num_list[label])

    # # 注意这里的weights应为所有样本的权重序列，其长度为所有样本长度
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(bird_data_Trainset),replacement=True)

    # 获取dataloader
    b_s=16
    trainloader = DataLoader(bird_data_Trainset,batch_size=b_s,num_workers=2)
    # trainloader = DataLoader(bird_data_Trainset,batch_size=b_s,num_workers=2, sampler = sampler)
    valloader = DataLoader(bird_data_Valset,batch_size=b_s,shuffle=False,num_workers=2)

    # 获取预训练的denseNet模型
    denseNetModel = getImageNet()
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(bird_data_Trainset.class_num_list).to(device).float())
    # 优化器
    # optimizer=optim.SGD(denseNetModel.parameters(), lr=0.001, momentum=0.9, factor=0.6)
    
    lr=0.0015
    optimizer = optim.Adam([
            {"params":denseNetModel.classifier.parameters(),"lr":lr},
            {"params":denseNetModel.features.parameters(),"lr":1e-6},
            ],
            lr=lr, #默认参数
        )

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, denseNetModel.parameters()),lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)


    path='../model/bird_v1.pt'
    path_new='../model/bird_new.pt'
    if os.path.exists(path):
        denseNetModel = torch.load(path)
        print('加载先前模型成功')
    else:
        print('未加载原有模型训练')

    epochNum=20

    # 在内存里面
    maxValAcc=0
    index_to_classes=getClassFromTxt(txtClassPath)
    denseNetModel=denseNetModel.to(device)


    for epoch in range(epochNum):

        denseNetModel.train()
        train_loss=0
        train_correct,train_total=0,0
        train_num=0

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

            # 梯度累积
            if batch%3!=0:
                continue

            optimizer.step()

            train_loss+=loss.item()
            train_num+=1

            if batch%5==0 and batch!=0:
                print(f'单batch：第{epoch+1}个epoch中训练到第{batch*b_s}个图片,训练集准确率为{100*train_correct/train_total}%')
            
        print('-'*35)
        print(f'epoch：第{epoch+1}次迭代,训练集准确率为{100*train_correct/train_total}%,loss为{train_loss/train_num}')

        # 进行测试
        denseNetModel.eval()
        test_loss=0
        test_correct,test_total=0,0

        valAcc=0
        val_num=0

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
                val_num+=1
                test_correct += batchCorrect
                test_total += batchSize


        valAcc=100*test_correct/test_total

        print(f'epoch：第{epoch+1}次迭代,验证集准确率为{valAcc}%，loss为{test_loss/val_num}')
        print('-'*35)

        if valAcc > maxValAcc:
            maxValAcc=valAcc
            # 保存模型
            torch.save(denseNetModel, path)
            print()
            print('模型更新成功~')
            print()
        # elif abs(valAcc-maxValAcc)<0.5:
        #   lr=lr*0.8
        #   optimizer = optim.Adam(filter(lambda p: p.requires_grad, denseNetModel.parameters()),lr=lr)
        #   print('优化器更新成功')
        # else:
        #   pass
        scheduler.step()
        # 打印当前学习率
        print('当前学习率为：',end=' ')
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        torch.save(denseNetModel,path_new)