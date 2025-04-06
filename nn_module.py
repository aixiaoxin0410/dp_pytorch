import torch
from torch import nn


class nn_test(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,input):
        output = input + 1
        return output

test_nn = nn_test()
x = torch.tensor(1.0)
output = test_nn(x)
print(output)