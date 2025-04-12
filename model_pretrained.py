import torchvision.models
from torch import nn

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))

# print(vgg16_true)

# print(vgg16_false)

vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)