import torch
import torchvision
from torch import nn
from model_save import * #更方便一点，不需要复制模型

# #方式1-> 保存方式1，加载模型
# model = torch.load("vgg16_method1.pth",weights_only=False)
# print(model)

# #方式2-> 保存方式2，加载字典
# model1 = torch.load("vgg16_method2.pth")
# print(model1)

# #方式2-> 保存方式2，加载模型
# vgg16 = torchvision.models.vgg16(weights=None)
# vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

#陷阱1
# class Test(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3,64,kernel_size=3)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         return x
model = torch.load("test_method1.pth",weights_only=False)
# print(model)

