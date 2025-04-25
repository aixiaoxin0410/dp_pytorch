# 步骤 1：准备环境
# ----------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import os
import numpy as np
from torch.optim.lr_scheduler import StepLR

# 步骤 2：定义数据集类
# ----------------------------------------
class ImageDataset(Dataset):
    def __init__(self, txt_file, image_dir, transform=None, has_labels=True):
        if not os.path.exists(txt_file):
            print(f"错误：找不到文件 {txt_file}")
            exit(1)
        if not os.path.exists(image_dir):
            print(f"错误：找不到目录 {image_dir}")
            exit(1)

        try:
            if has_labels:
                self.data = pd.read_csv(txt_file, sep='\s+', header=None, names=['filename', 'label'])
            else:
                self.data = pd.read_csv(txt_file, sep='\s+', header=None, names=['filename'])
                self.data['label'] = None
        except Exception as e:
            print(f"错误：无法读取文件 {txt_file}，错误信息：{e}")
            exit(1)

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

        self.reverse_label_map = {}
        for key, value in self.label_map.items():
            self.reverse_label_map[value] = key

        valid_data = []
        for idx in range(len(self.data)):
            filename = self.data.iloc[idx]['filename']
            img_path = os.path.join(self.image_dir, filename)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_data.append(self.data.iloc[idx])
            except Exception as e:
                print(f"跳过无效图片 {img_path}：{e}")

        self.data = pd.DataFrame(valid_data, columns=self.data.columns)

        if has_labels:
            for label in self.data['label']:
                if label not in self.label_map:
                    print(f"错误：文件 {txt_file} 中有无效标签：{label}")
                    exit(1)
            self.data['label'] = self.data['label'].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx]['filename'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图片失败 {img_path}：{e}")
            return None

        label = self.data.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            return image, label
        else:
            return image, self.data.iloc[idx]['filename']

# 步骤 3：加载和预处理数据
# ----------------------------------------
base_path = "../spring_camp_intelligent_knights/data/"

if not os.path.exists(base_path):
    print(f"错误：路径 {base_path} 不存在")
    exit(1)

# 训练数据的 transform（添加增强）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试数据的 transform（不添加增强）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练数据
train_dataset = ImageDataset(
    txt_file=os.path.join(base_path, "train.txt"),
    image_dir=os.path.join(base_path, "train/"),
    transform=train_transform,
    has_labels=True
)

# 划分训练集和验证集
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)
split = int(0.8 * dataset_size)
train_indices, val_indices = indices[:split], indices[split:]

def collate_fn_train(batch):
    images = []
    labels = []
    for item in batch:
        if item is None:
            continue
        image, label = item
        images.append(image)
        labels.append(label)
    return images, labels

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=SubsetRandomSampler(train_indices),
    collate_fn=collate_fn_train
)

val_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=SubsetRandomSampler(val_indices),
    collate_fn=collate_fn_train
)

# 加载测试数据
a_dataset = ImageDataset(
    txt_file=os.path.join(base_path, "A.txt"),
    image_dir=os.path.join(base_path, "A/"),
    transform=test_transform,
    has_labels=False
)

def collate_fn_test(batch):
    images = []
    filenames = []
    for item in batch:
        if item is None:
            continue
        image, filename = item
        images.append(image)
        filenames.append(filename)
    return images, filenames

a_loader = DataLoader(
    a_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn_test
)

print(f"训练数据集大小：{len(train_indices)} 张图片")
print(f"验证数据集大小：{len(val_indices)} 张图片")
print(f"测试数据集大小：{len(a_dataset)} 张图片")

# 步骤 4：构建和训练模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 加权损失
class_counts = torch.tensor([500, 600, 700, 800, 92], dtype=torch.float32)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * 5
class_weights = class_weights.to(device)
print(f"类别权重：{class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加 L2 正则化
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # 改回 StepLR

# 训练和验证循环
num_epochs = 15
best_val_acc = 0.0
best_model_path = "best_model.pth"
patience = 5
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_count = 0
    for images, labels in train_loader:
        if not images or not labels:
            continue
        images = torch.stack(images).to(device)
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_count += 1
    if batch_count > 0:
        print(f"第 {epoch+1}/{num_epochs} 轮，训练损失：{running_loss/batch_count:.4f}")

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    val_batch_count = 0
    with torch.no_grad():
        for images, labels in val_loader:
            if not images or not labels:
                continue
            images = torch.stack(images).to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_batch_count += 1
            _, predicted_labels = torch.max(outputs, 1)
            total += labels.size(0)
            is_correct = (predicted_labels == labels)
            num_correct = is_correct.sum().item()
            correct += num_correct

    val_accuracy = (correct / total) * 100 if total > 0 else 0
    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
    print(f"第 {epoch + 1}/{num_epochs} 轮，验证集准确率：{val_accuracy:.2f}%，验证损失：{avg_val_loss:.4f}")

    scheduler.step()  # StepLR 不需要验证损失

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"保存最佳模型，验证准确率：{val_accuracy:.2f}%")
        counter = 0
    else:
        counter += 1
        print(f"验证准确率未提升，计数器：{counter}/{patience}")
        if counter >= patience:
            print("早停：验证准确率不再提升，停止训练")
            break

# 加载最佳模型进行预测
model.load_state_dict(torch.load(best_model_path))
print(f"加载最佳模型，验证准确率：{best_val_acc:.2f}%")

# 步骤 5：对测试数据进行预测
# ----------------------------------------
model.eval()
predictions = []
with torch.no_grad():
    for images, filenames in a_loader:
        if not images or not filenames:
            continue
        images = torch.stack(images).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        predicted_labels = []
        for p in predicted:
            number_label = p.item()
            text_label = train_dataset.reverse_label_map[number_label]
            predicted_labels.append(text_label)

        for i in range(len(filenames)):
            filename = filenames[i]
            label = predicted_labels[i]
            predictions.append((filename, label))

# 步骤 6：保存预测结果
# ----------------------------------------
output_file = os.path.join(base_path, "predictions.txt")
with open(output_file, "w", encoding="utf-8", newline="\n") as f:
    for filename, label in predictions:
        if "/" in filename or "\\" in filename:
            print(f"警告：文件名 {filename} 包含路径，已自动修正")
            filename = os.path.basename(filename)
        f.write(f"{filename}\t{label}\n")

print(f"预测完成，结果已写入 {output_file}")