import torchvision.datasets

vgg16_false = torchvision.models.vgg16(pretrain=False)
vgg16_true = torchvision.models.vgg16(pretrain=True)

print(vgg16_true)