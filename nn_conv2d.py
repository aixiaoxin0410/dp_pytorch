import torchvision
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.ToTensor()
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset", train=False, transform=dataset_transform, download=True)

dataloader = DataLoader(test_set, batch_size=64)

class My_nn(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

test_nn = My_nn()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    output = test_nn(imgs)
    print(imgs.shape)    # torch.Size([64, 3, 32, 32])
    print(output.shape)  # torch.Size([64, 6, 30, 30])
    # [64, 6, 30, 30] -> [xxx, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))  # 第一个值不知道是多少的时候，填-1，会根据后面三个值自己计算

    writer.add_images("test_nn_input", imgs, step)
    writer.add_images("test_nn_output", output, step)

    step = step + 1

writer.close()

