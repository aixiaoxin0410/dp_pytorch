import torch
import torchvision
from torch import nn
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# input = torch.tensor([[-1, -0.5],
#                      [-1, 3]])
#
# input = torch.reshape(input,(-1,1,2,2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Test_Nnrelu(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


test_nnrelu = Test_Nnrelu()
# output = test_nnrelu(input)
# print(output)

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs,targets = data
    writer.add_images("initial", imgs, step)
    output = test_nnrelu(imgs)
    writer.add_images("sigmoid_initial", output, step)
    step = step + 1

writer.close()