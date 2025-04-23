# 步骤1：准备环境
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# 导入需要的库
import logging
import os.path
import pandas as pd
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

# 步骤2：定义数据集类
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# 这个类用来加载图和标签，处理训练和测试数据
class ImageDataset(Dataset):
    def __init__(self, txt_file, image_dir, transform=None, has_labels=True):
        # 检查txt文件是否存在
        if not os.path.exists(txt_file):
            print(f"错误：找不到文件{txt_file}")
            exit(1)
        # 检查图片目录是否存在
        if not os.path.exists(image_dir):
            print(f"错误：找不到目录{image_dir}")
            exit(1)

        # 读取txt
        # 如果 has_labels=True（训练数据），文件格式是”文件名 标签“
        # 如果 has_labels=False（测试数据），文件格式是”文件名“
        try:
            if has_labels:
                self.data = pd.read_csv(txt_file, sep="\s+",header=None,
                                        names = ['filename', 'label'])
            else:
                self.data = pd.read_csv(txt_file, sep="\s+", header=None,
                                        names=['filename'])
                self.data['label'] = None # 测试数据没有标签，设为None

        except Exception as e:
            print(f"错误：无法读取文件{txt_file}，错误信息：{e}")
            exit(1)

        self.image_dir = image_dir
        self.transform = transform
        self.has_labels = has_labels

        # 定义标签映射（中文标签转数字）
        self.label_map = {
            "高风险": 0,
            "中风险": 1,
            "低风险": 2,
            "无风险": 3,
            "非楼道": 4
        }

        # 创建反向映射（数字转中文标签），用于预测时显示中文
        self.reverse_label_map = {}
        for key,value in self.label_map.items():
            self.reverse_label_map[value] = key

        # 验证图片文件是否有效，只保留有效的图片
        valid_data = []
        for idx in range(len(self.data)):
            filename = self.data.iloc[idx]['filename']
            img_path = os.path.join(self.image_dir,filename)
            try:
                # 打开图片并验证
                with Image.open(img_path) as img:
                    img.verify()
                valid_data.append(self.data.iloc[idx])
            except Exception as e:
                print(f"跳过无效图片：{img_path}")

        # 更新数据，只保留有效图片
        self.data = pd.DataFrame(valid_data,columns=self.data.columns)

        # 如果是训练数据，检查标签是否有效，并将中文标签转换为数字
        if has_labels:
            for label in self.data['label']:
                if label not in self.label_map:
                    print(f"错误：文件 {txt_file} 中有无效标签：{label}")
                    exit(1)
            self.data['label'] = self.data['label'].map(self.label_map)

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图片路径
        img_path = os.path.join(self.image_dir,self.data.iloc[idx]['filename'])
        # 尝试打开图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"记载图片失败{img_path}:{e}")
            return None # 如果失败，返回None，后续会跳过

        # 获取标签（训练数据）或文件名（测试数据）
        label = self.data.iloc[idx]['label']
        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 返回数据
        if self.has_labels:
            return image,label
        else:
            return image,self.data.iloc[idx]['filename']

# 步骤3：加载和预处理数据
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# 定义基本路径
base_path = "../spring_camp_intelligent_knights/data/"

# 检查路径是否存在
if not os.path.exists(base_path):
    print(f"错误：路径{base_path}不存在")
    exit(1)

# 定义图片预处理步骤
transform = transforms.Compose([
    transforms.Resize((224,224)), # 调整图片大小为 224 x 224
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #归一化
])

# 加载训练数据
train_dataset = ImageDataset(txt_file=os.path.join(base_path,"train.txt"),
                          image_dir=os.path.join(base_path,"train/"),
                          transform=transform,
                          has_labels=True)

# 定义一个函数来处理批次数据，过滤掉无效图片
def collate_fn_train(batch):
    images = []
    labels = []
    for item in batch:
        if item is None: # 如果图片加载失败，跳过
            continue
        image,label = item # 拆开图片和标签
        images.append(image)
        labels.append(label)
    return images,labels

# 创建训练数据加载器
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,
                              collate_fn=collate_fn_train)

# 加载测试数据（同理，同上面操作训练数据的流程）
a_dataset = ImageDataset(txt_file=os.path.join(base_path,"A.txt"),
                          image_dir=os.path.join(base_path,"A/"),
                          transform=transform,
                          has_labels=False)

def collate_fn_test(batch):
    images = []
    filenames = []
    for item in batch:
        if item is None: # 如果图片加载失败，跳过
            continue
        image,filename = item # 拆开图片和标签
        images.append(image)
        filenames.append(filename)
    return images,filenames

a_loader = DataLoader(a_dataset,batch_size=32,shuffle=False,
                              collate_fn=collate_fn_test)

# 打印数据集大小
print(f"训练数据集大小：{len(train_dataset)}张图片")
print(f"测试数据集大小：{len(a_dataset)}张图片")

# 步骤4：构建和训练模型
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# 加载预训练的 ResNet-18模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# 修改最后一层，适用5个类型（迁移学习）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,5)

# 设置设备（GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train() # 设置模型为训练模式
    running_lose = 0.0
    batch_count = 0
    for images,labels in train_loader:
        # 如果批次为空，则跳过
        if not images or not labels:
            continue
        # 将图片和标签转换为张量并移动到设备上
        images = torch.stack(images).to(device)
        labels = torch.tensor(labels).to(device)
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs,labels)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 累加损失
        running_lose += loss.item()
        batch_count += 1
    # 打印每轮的平均损失
    if batch_count > 0:
        print(f"第{epoch+1}/{num_epochs}轮，平均损失：{running_lose/batch_count:.4f}")
    else:
        print(f"第{epoch + 1}/{num_epochs}轮，没有有效批次可以训练")

# 步骤5：对测试数据进行预测
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
model.eval() # 设置模型为评估模式
predictions = []
with torch.no_grad(): # 不计算梯度
    for images,filenames in a_loader:
        if not images or not filenames:
            continue
        # 将图片转换为张量并移动到设备上
        images = torch.stack(images).to(device)
        # 预测
        outputs = model(images)
        _,predicted = torch.max(outputs,1)

        # 将数字标签转换为中文标签
        predicted_labels = []
        for p in predicted:
            number_label = p.item()
            text_label = train_dataset.reverse_label_map[number_label]
            predicted_labels.append(text_label)

        # 将文件名和预测标签配对
        for i in range(len(filenames)):
            filename = filenames[i]
            label = predicted_labels[i]
            predictions.append((filename,label))

# 步骤6：保存预测结果
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# 保存文件路径
# output_file = os.path.join(base_path,"predictions.txt")
output_file = "predictions.txt"
# 检查文件是否存在
if os.path.exists(output_file):
    logging.warning(f"文件 {output_file} 已存在，将被覆盖")
# 打开文件，使用UTF-8编码
with open(output_file,"w",encoding="utf-8",newline="\n") as f:
    for filename,label in predictions:
        # 检查文件名，确保不含路径
        if "/" in filename or "\\" in filename:
            print(f"警告：文件名{filename}包含路径，已自动修正")
            filename = os.path.basename(filename)

        # 使用Tab键分隔，强制使用LF换行符
        f.write(f"{filename}\t{label}\n")

print(f"预测完毕，结果已写入{output_file}")
