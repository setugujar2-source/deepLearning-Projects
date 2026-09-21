# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果用中文显示标签）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前使用设备: {device}')

# ---------- 超参数 ----------
img_height = 64
img_width = 64
batch_size = 32
epochs = 20
learning_rate = 0.0001

# ---------- 数据增强和预处理 ----------
# 训练集：随机翻转、旋转、缩放，并归一化
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(size=(img_height, img_width), scale=(0.8, 1.0)),
    transforms.ToTensor(),                     # 转换为 Tensor，并缩放到 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 使用 ImageNet 的均值标准差（通用做法）
                         std=[0.229, 0.224, 0.225])
])

# 验证集和测试集：只调整大小和归一化，不做随机增强
val_test_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- 加载数据集 ----------
# 假设你的数据文件夹结构：
# my_dataset/
# ├── train/  (每类一个子文件夹，如 book, cup, mouse)
# └── test/   (同上)
train_dataset = datasets.ImageFolder(root='my_dataset/train', transform=train_transforms)
# 从训练集中划分 20% 作为验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(123)
)
# 验证集需要使用验证集的数据预处理（注意：random_split 后数据仍然应用 train_transforms，
# 但验证集不想要增强，所以需要重新设置。简单起见，我们在下面为验证集重新创建数据集。）
# 更规范的做法：先用 ImageFolder 加载全部训练图片，然后为验证集重新指定 transform。
# 为了简单，这里我们直接手动从 train 目录重新读取一次并应用 val_test_transforms。
val_dataset_full = datasets.ImageFolder(root='my_dataset/train', transform=val_test_transforms)
val_dataset = torch.utils.data.Subset(val_dataset_full, range(train_size, len(val_dataset_full)))

# 测试集
test_dataset = datasets.ImageFolder(root='my_dataset/test', transform=val_test_transforms)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 获取类别名称
class_names = val_dataset_full.classes   # 直接使用 .classes
print('类别:', class_names)

# ---------- 构建 CNN 模型 ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (img_height//8) * (img_width//8), 64),  # 64*8*8
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=len(class_names)).to(device)
print(model)

# ---------- 定义损失函数和优化器 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------- 训练函数 ----------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ---------- 验证/测试函数 ----------
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ---------- 训练循环 ----------
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'Epoch {epoch}/{epochs} | '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

# ---------- 在测试集上最终评估 ----------
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f'\n测试集准确率: {test_acc:.4f}')

# ---------- 可视化训练过程 ----------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='训练准确率')
plt.plot(val_accs, label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()