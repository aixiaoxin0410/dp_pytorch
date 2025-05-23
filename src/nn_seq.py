import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Test_Seq(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.conv1 = Conv2d(3,32,5,1,2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5,padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten1 = Flatten()
        # self.linear1 = Linear(1024,64)
        # self.linear2 = Linear(64,10)

        self.model1 = Sequential(Conv2d(3,32,5,1,2),
                                 MaxPool2d(2),
                                 Conv2d(32,32,5,padding=2),
                                 MaxPool2d(2),
                                 Conv2d(32, 64, 5,padding=2),
                                 MaxPool2d(2),
                                 Flatten(),
                                 Linear(1024,64),
                                 Linear(64,10))

    def forward(self,x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten1(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)

        return x

test_seq = Test_Seq()
print(test_seq)

input = torch.ones((64,3,32,32))
output = test_seq(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(test_seq,input)
writer.close()