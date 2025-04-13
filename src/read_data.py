from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

# VideoWidth = 640
# VideoHeight = 480
#
# cap = cv2.VideoCapture(1)
# cap.set(3,VideoWidth)
# cap.set(4,VideoHeight)
# cap.set(10,150)
#
# while(True):
#     success,img = cap.read()
#     cv2.imshow("test",img)
#     if cv2.waitKey(1) & 0xFF ==ord("q"):
#         break
#
# cap.release()

# img_path = "dataset/train/ants/0013035.jpg"
# imgtest = cv2.imread(img_path)
# cv2.imshow("test",imgtest)
#
# cv2.waitKey(0)

# img_path = "C:/Users/zhongweixin/PycharmProjects/dp_pytorch/dataset/train/ants/0013035.jpg"
#
# img = Image.open(img_path)
#
# dir_path = "dataset/train/ants"
#
# img_path_list = os.listdir(dir_path)

class Mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_name_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_name_path)
        label = self.label_dir
        return img,label

    def __long__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = Mydata(root_dir,ants_label_dir)
bees_dataset = Mydata(root_dir,bees_label_dir)

img,label = ants_dataset[0]
img.show(img)