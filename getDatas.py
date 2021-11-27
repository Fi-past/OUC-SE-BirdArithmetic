import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os
import torchvision as tv
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

# 训练集预处理
preprocessTrain = transforms.Compose([
    # transforms.Resize([230, 230]),
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.08),
    # transforms.RandomRotation(15),
    # 功能：修改修改亮度、对比度和饱和度
    transforms.ColorJitter(brightness=0.05, contrast=0.05, hue=0.05),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    ),
])

# 测试验证集预处理
preprocessVal_Test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    ),
])

# 从文件中读取数据
def defaultLoader(path,ifTrain):
    img_pil =  Image.open(path).convert('RGB')
    # print(len(img_pil))
    if ifTrain==True:
        img_tensor = preprocessTrain(img_pil)
    else:
        img_tensor = preprocessVal_Test(img_pil)
    return img_tensor


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


# 获取图片的路径
def getImgPath(imgPath):
    imgs=[]
    for imgName in os.listdir(imgPath):
        imgs.append(imgName)
    return imgs

# 判断样本是否平衡，传入dataloader
def ifBalance(train_loader):
    class_num_list=[0 for i in range(200)]
    for i, (data, labels) in enumerate(train_loader):
        for label in labels:
            class_num_list[label]+=1
    print(class_num_list)


# dataset类
class birdTrainDataSet(Dataset):
    def __init__(self,imgTrainPath,txtClassPath, ifTrain):
        self.class_num_list=[0 for i in range(200)]
        self.txtClassPath=txtClassPath
        self.imgTrainPath=imgTrainPath
        self.ifTrain=ifTrain
        self.index_classes=getClassFromTxt(txtClassPath)
        # print(self.index_classes)
        self.imgNames=getImgPath(imgTrainPath)
        # 统计类的种类
        self.getClassNum()

    def __getitem__(self, index):
        imgName=self.imgNames[index]
        num,name=imgName.split('.',1)
        label=int(num)
        img=defaultLoader(os.path.join(self.imgTrainPath,imgName),self.ifTrain)
        label=label-1
        return img,label

    def __len__(self):
        return len(self.imgNames)

    def getClassNum(self):
        for imgname in self.imgNames:
            num,name=imgname.split('.',1)
            label=int(num)-1
            self.class_num_list[label]+=1
        self.class_num_list=1/np.array(self.class_num_list)

        # print(self.class_num_list)
        # 不确定：根据torch种平衡样本的语法，应该取倒数
        # self.class_num_list=1/np.array(self.class_num_list)


if __name__ == '__main__' : 

    imgTrainPath='../data/bird/train_set'
    txtClassPath='../data/bird/classes.txt'

    train_dataset=birdTrainDataSet(imgTrainPath,txtClassPath,True)
    # 测试可知样本不均衡
    # print(train_dataset.class_num_list)


    # show=ToPILImage()
    # (data, label) = bd[100]
    # print(label)
    # data=show((data+1)/2)
    # # print(type(data))
    # # print(data)
    # plt.imshow(data)
    # plt.title('image') # 图像题目
    # plt.show()

    # 不准

    weights=[]
    for data, label in train_dataset:
        weights.append(train_dataset.class_num_list[label])

    batch_size=64
    # 注意这里的weights应为所有样本的权重序列，其长度为所有样本长度
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset),replacement=True) 
    trainloader = DataLoader(train_dataset, batch_size = batch_size, sampler = sampler)
    # trainloader = DataLoader(train_dataset, batch_size = batch_size)
    ifBalance(trainloader)

    # iterloader=iter(trainloader)
    # images,label=iterloader.next()
    # print(images.size())
    # print(label)