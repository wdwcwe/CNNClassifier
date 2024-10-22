import os
import torch
from torch import nn
import torch.optim as optim

from torchvision import transforms, models, datasets


from src.model import initialize_model
from src.train import train_model

#基于花的卷积网络模型搭建，使用resnet神经网络框架进行迁移学习
#1.读取数据
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
#2.数据预处理
#2.1数据增强  增加原始数据的多样性和泛化能力，并将图片数据归一化
data_transforms = {
    #训练数据
    'train': transforms.Compose([transforms.RandomRotation(45)，#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224)，#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5)，#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5)，#随机垂直翻转
        transforms.ColorJitter(brightness=0.2， contrast=0.1， saturation=0.1， hue=0.1)，#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025)，#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406]， [0.229, 0.224, 0.225])#均值，标准差
    ]),
    #测试数据
    'valid': transforms.Compose([transforms.Resize(256)，
        transforms.CenterCrop(224)，
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406]， [0.229, 0.224, 0.225])
    ]),
}
#2.2创建数据加载器   使用DataLoader模块
batch_size=8   #批次大小
#将不同的数据用ImageFolder加载并按照data_transforms的参数进行预处理后存入image_datasets
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
#构建数据加载器
dataloaders={x: torch.utils.data.DataLoader(image_datasets[x],batch_size,shuffle=True) for x in ['train','valid']}
datasets_sizes={x:len(image_datasets[x]) for x in ['train','valid']}

model_name='resnet'  #可选的比较多 ['resnet'， 'alexnet'， 'vgg'， 'squeezenet'， 'densenet'， 'inception']
#是否用别人训练好的权重
feature_extract=True
#是否用GPU进行训练
train_on_gpu=torch.cuda.is_available()
if not train_on_gpu :
    print('CUDA is not available.  Training on CPU....')
else:
    print('CUDA is available     Train on GPU....'  )
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#检查训练了哪些层是否和自己初始化的一样只有全连接层
model_ft,input_size=initialize_model(model_name,102,feature_extract,use_pretrained=True)
model_ft=model_ft.to(device)#将模型放在GPU、
# print(model_ft)
filename='CNNFlower.pth'
params_to_update=model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update=[]
    for name,param in model_ft.named_parameters():
        if param.requires_grad==True:
            params_to_update.append(param)
            print("\t"，name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad==True:
            print("\t"，name)
#优化器和损失函数设置   传入params_to_update只对需要更新的层进行更新
optimizer_ft=optim.Adam(params_to_update,lr=1e-2)
#每7次epoch，学习率衰减为原来的1/10
scheduler=optim.lr_scheduler.StepLR(optimizer_ft,step_size=7，gamma=0.1)
#因为resnet网络最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion=nn.NLLLoss()
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer_ft,scheduler,device,10, model_name=="inception"，filename)


