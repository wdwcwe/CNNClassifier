import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder

from src.model import initialize_model
from torchvision import transforms, models, datasets
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作   保证输入的大小是一致的
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # 相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))

    return img


def im_convert(tensor):
    """ 展示数据"""

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image
data_dir = '../flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
model_name='resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用别人训练好的权重
feature_extract=True
#是否用GPU进行训练
train_on_gpu=torch.cuda.is_available()
if not train_on_gpu :
    print('CUDA is not available.  Training on CPU....')
else:
    print('CUDA is available     Train on GPU....'  )
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transforms = {
    #训练数据
    'train': transforms.Compose([transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    #测试数据
    'valid': transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
#2.2创建数据加载器   使用DataLoader模块
batch_size=8   #批次大小
#将不同的数据用ImageFolder加载并按照data_transforms的参数进行预处理后存入image_datasets
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
#构建数据加载器
dataloaders={x: torch.utils.data.DataLoader(image_datasets[x],batch_size,shuffle=True) for x in ['train','valid']}
img_dataset = ImageFolder("../flower_data/train")
#获取不同类别文件夹的名字
class_names=img_dataset.classes
#3.加载模型
model, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU模式
model = model.to(device)
filename="../CNNFlower.pth"
# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model.load_state_dict(checkpoint['state_dict'])#让模型按照训练好的权重参数加载
#从数据加载器中获取数据进行预测
dataiter=iter(dataloaders['valid'])
images,labels=next(dataiter)
model.eval()
if train_on_gpu:
    output = model(images.cuda())
else:
    output = model(images)
#获取预测结果
_,preds_tensor=torch.max(output,1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
#根据结果画图展示
fig=plt.figure(figsize=(20,20))
columns=4
rows=2
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
for idx in range(columns*rows):
    ax=fig.add_subplot(rows,columns,idx+1,xticks=[],yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.show()