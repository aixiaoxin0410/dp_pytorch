import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../imgs/dog.png"
image = Image.open(image_path)
# print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])

image = transform(image)
# print(image.shape)

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

model = Nn_net()
model.load_state_dict(torch.load("../src/nn_net_0.pth"))
# print(model)

image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))