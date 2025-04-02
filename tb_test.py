from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import cv2
import numpy as np

writer = SummaryWriter("logs")

img_path = "dataset/train/ants/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
img_array = np.array(img)

print(img_array.shape)
writer.add_image("test",img_array,2,dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=3x",3*i,i)

writer.close()







