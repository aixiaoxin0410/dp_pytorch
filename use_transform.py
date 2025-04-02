from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img = Image.open("dataset/train/ants/6240338_93729615ec.jpg")

#Tensor
tensor_train = transforms.ToTensor()
img_tensor = tensor_train(img)
writer.add_image("totensor",img_tensor)
print(img_tensor[0][0][0])

#Normalize
trans_norm = transforms.Normalize([1,3,5],[3,2,1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("normalize",img_norm,2)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = tensor_train(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)


#Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, tensor_train])
img_resize_2 = trans_compose(img)
writer.add_image("resize_compose",img_resize_2,0)

#RandomCrop
trans_random = transforms.RandomCrop(50)
trans_compose_2 = transforms.Compose([trans_random,tensor_train])
for i in range(10):
     img_crop = trans_compose_2(img)
     writer.add_image("img_crop",img_crop,i)

writer.close()