import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset", transform=torchvision.transforms.ToTensor(),train=False, download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Test_Seq(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        x = self.model1(x)

        return x

test_seq = Test_Seq()
lose_cross = CrossEntropyLoss()

for data in dataloader:
    imgs,targets = data
    outputs = test_seq(imgs)
    result_losecross = lose_cross(outputs,targets)
    # print(result_losecross)
    # result_losecross.backward()
    print("ok")
    # print(outputs)
    # print(targets)