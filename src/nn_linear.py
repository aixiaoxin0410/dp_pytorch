import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset,batch_size=64,drop_last=True) #加上drop_last=True不报错。

class Test(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output

test = Test()

for data in dataloader:
    imgs,target = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)

    output = test(output)
    print(output.shape)