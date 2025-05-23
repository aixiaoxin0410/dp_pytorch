import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights=None)

#保存方法1：模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")

#保存方法2：模型参数（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#陷阱1
class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x

test = Test()
torch.save(test,"test_method1.pth")