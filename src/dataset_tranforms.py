import torchvision
from torch.utils.tensorboard import SummaryWriter

from tb_test import writer

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                                    ])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset", transform=dataset_transform,train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset", transform=dataset_transform,train=False, download=True)

# print(test_set[0])
# img,target = test_set[0]
# print(img)
# print(target)
#
# img.show()

print(test_set[0])

writer = SummaryWriter("logs")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()