import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_tranforms import writer

#准备测试数据集
dataset_transform = torchvision.transforms.ToTensor()

test_set = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset", transform=dataset_transform,train=False, download=True)

test_loader = DataLoader(dataset=test_set,batch_size=4,shuffle=True,num_workers=0,drop_last=False)

img,target = test_set[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step = step + 1

writer.close()