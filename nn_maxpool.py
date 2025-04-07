import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_conv2d import dataloader, targets, output

dataset = torchvision.datasets.CIFAR10("CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("logs")

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]
#                       ],dtype=torch.float32)
#
# kernel = torch.tensor([[1,2,1],
#                        [0,1,0],
#                        [2,1,0]
#                        ])

# input = torch.reshape(input,(-1,1,5,5,))
# print(input.shape)

class Test_MaxPool(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)  # 2 3 5 1
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False) # 2

    def forward(self,input):
        output = self.maxpool1(input)
        return output

test_maxpool = Test_MaxPool()

# output = test_maxpool(input)
# print(output)
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("inputs", imgs, step)
    output = test_maxpool(imgs)
    writer.add_images("maxpool_inputs", output, step)
    step = step + 1

writer.close()
