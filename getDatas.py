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
    # transforms.Resize([224, 224]),
    transforms.Resize([230, 230]),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),

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


# dataset类
class birdTrainDataSet(Dataset):
    def __init__(self,imgTrainPath,txtClassPath, ifTrain):
        self.txtClassPath=txtClassPath
        self.imgTrainPath=imgTrainPath
        self.ifTrain=ifTrain
        self.index_classes=getClassFromTxt(txtClassPath)
        # print(self.index_classes)
        self.imgNames=getImgPath(imgTrainPath)

    def __getitem__(self, index):
        imgName=self.imgNames[index]
        num,name=imgName.split('.',1)
        label=int(num)
        img=defaultLoader(os.path.join(self.imgTrainPath,imgName),self.ifTrain)
        label=label-1
        return img,label

    def __len__(self):
        return len(self.imgNames)


if __name__ == '__main__' : 

    imgTrainPath='../data/bird/train_set'
    txtClassPath='../data/bird/classes.txt'

    bd=birdTrainDataSet(imgTrainPath,txtClassPath,True)

    show=ToPILImage()
    (data, label) = bd[100]
    print(label)
    data=show((data+1)/2)
    # print(type(data))
    # print(data)
    plt.imshow(data)
    plt.title('image') # 图像题目
    plt.show()

    trainloader = DataLoader(bd,batch_size=64,shuffle=True)

    iterloader=iter(trainloader)
    images,label=iterloader.next()
    print(images.size())
    print(label)