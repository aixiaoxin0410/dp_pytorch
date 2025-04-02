from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

# print(img)

tensor_train = transforms.ToTensor()

writer = SummaryWriter("logs")

tensor_img = tensor_train(img)
# print(tensor_img)

writer.add_image("tensor_img",tensor_img)

writer.close()