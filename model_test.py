import torch
from torch import nn


class Nn_net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(nn.Conv2d(3,32,5,1,2),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32,32,5,1,2),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32,64,5,1,2),
                                   nn.MaxPool2d(2),
                                   nn.Flatten(),
                                   nn.Linear(64*4*4,64),
                                   nn.Linear(64,10))

    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    nn_net = Nn_net()
    input = torch.ones((64,3,32,32))
    output = nn_net(input)
    print(output.shape)