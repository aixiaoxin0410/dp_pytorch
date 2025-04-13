# 步骤 1：准备环境
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

# 步骤 2：定义数据集类
class ImageDataset(Dataset):
    def __init__(self, txt_file, image_dir, transform=None, has_labels=True):
        if has_labels:
            self.data = pd.read_csv(txt_file, sep='\s+', header=None, names=['filename', 'label'])
        else:
            self.data = pd.read_csv(txt_file, sep='\s+', header=None, names=['filename'])
            self.data['label'] = None
        self.image_dir = image_dir
        self.transform = transform
        self.has_labels = has_labels
        self.label_map = {
            "高风险": 0,
            "中风险": 1,
            "低风险": 2,
            "无风险": 3,
            "非楼道": 4
        }
        reverse_label_map = {}  # 创建一个空字典

        # 反转 label_map
        for k, v in self.label_map.items():
            reverse_label_map[v] = k

        self.reverse_label_map = reverse_label_map  # 将结果赋值给 reverse_label_map

        if has_labels:
            self.data['label'] = self.data['label'].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        if self.has_labels:
            return image, label
        else:
            return image, self.data.iloc[idx]['filename']

# 步骤 3：加载和预处理数据
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(txt_file="train.txt", image_dir="train/", transform=transform, has_labels=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

a_dataset = ImageDataset(txt_file="A.txt", image_dir="A/", transform=transform, has_labels=False)
a_loader = DataLoader(a_dataset, batch_size=32, shuffle=False)

# 步骤 4：构建和训练模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 五种类别

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 步骤 5：对 A 数据进行预测
model.eval()
predictions = []
with torch.no_grad():
    for images, filenames in a_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        # 转换数字标签为中文标签
        predicted_labels = []
        for p in predicted:
            number_label = p.item()  # 将张量转换为整数
            text_label = train_dataset.reverse_label_map[number_label]  # 查找中文标签
            predicted_labels.append(text_label)

        # 将文件名和中文标签配对
        for i in range(len(filenames)):
            filename = filenames[i]  # 获取文件名
            label = predicted_labels[i]  # 获取中文标签
            predictions.append((filename, label))  # 添加到预测结果列表

# 步骤 6：保存预测结果
with open("A.txt", "w", encoding="utf-8") as f:
    for filename, label in predictions:
        f.write(f"{filename} {label}\n")

print("预测完成，结果已写入 A.txt")