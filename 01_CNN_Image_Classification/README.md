# 基于PyTorch的CNN图像分类实践

## 项目简介
使用 PyTorch 搭建卷积神经网络（CNN），对自建数据集（书本、水杯、鼠标）进行三分类识别。

## 核心技术
- **数据增强**：使用了 `RandomHorizontalFlip`、`RandomRotation`、`RandomResizedCrop` 等数据增强技术，解决小样本数据下的过拟合问题。
- **模型结构**：自定义 `SimpleCNN` 类（继承 `nn.Module`），包含三层卷积池化结构，以及全连接层和 Dropout(0.5)。
- **训练流程**：手动实现完整的 PyTorch 训练循环，包括前向传播、梯度清零、反向传播和参数更新。
- **设备适配**：自动检测并适配 GPU/CPU 设备进行训练。

## 运行结果
经过20轮训练，测试集准确率达到 **66.89% 以上**（训练曲线见代码运行生成的图表）。

## 如何运行
1. 准备数据集目录结构：
   `my_dataset/train/{book, cup, mouse}`
   `my_dataset/test/{book, cup, mouse}`
2. 安装依赖：`pip install torch torchvision matplotlib numpy`
3. 运行代码：`python cnn_image_classification.py`
