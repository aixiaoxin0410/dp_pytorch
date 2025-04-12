#准备数据集
from os import write

import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_test import *

train_data = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

#用dataloader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# #搭建神经网络(放在model文件中，规范性）
# class Nn_net(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.model = nn.Sequential(nn.Conv2d(3,32,5,1,2),
#                                    nn.MaxPool2d(2),
#                                    nn.Conv2d(32,32,5,1,2),
#                                    nn.MaxPool2d(2),
#                                    nn.Conv2d(32,64,5,1,2),
#                                    nn.MaxPool2d(2),
#                                    nn.Flatten(),
#                                    nn.Linear(64*4*4,64),
#                                    nn.Linear(64,10))
#
#     def forward(self,x):
#         x = self.model(x)
#         return x

#创建网络模型
nn_net = Nn_net()

#创建损失函数
loss_fc = nn.CrossEntropyLoss()

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(nn_net.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_train_step = 0 #记录训练的次数
total_test_step = 0 #记录测试的次数
epoch = 10 #训练的轮数

#添加tensorboard
writer = SummaryWriter("logs")


for i in range(10):
    print("第 {} 轮训练开始".format(i+1))

    # 训练步骤开始
    nn_net.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = nn_net(imgs)

        #计算结果与真实值的差距
        loss = loss_fc(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad() #梯度清零
        loss.backward() #反向传播
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    nn_net.eval()
    total_test_loss = 0
    total_accuracy = 0 #整体的正确率
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = nn_net(imgs)
            loss = loss_fc(outputs,targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(nn_net,"nn_net_{}.pth".format(i))
    torch.save(nn_net.state_dict(),"nn_net_{}.pth".format(i))

    print("模型已保存")

writer.close()