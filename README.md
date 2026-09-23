# DeepLearning-Projects

我的深度学习实践项目仓库。主要聚焦于**时序信号回归**与**图像分类**。

## 🛠️ 技术栈
Python | PyTorch | TensorFlow | NumPy | Pandas | Matplotlib | Scikit-learn

## 📁 项目列表

### 1. 基于PyTorch的CNN图像分类
- **路径**：`01_CNN_Image_Classification`
- **简介**：自建书本/水杯/鼠标三分类数据集。采用数据增强和Dropout防止过拟合，测试集准确率达 **66.89% **。

### 2. 基于一维CNN的IGBT结温预测
- **路径**：`02_IGBT_Temperature_Regression`
- **简介**：模拟电力电子器件测试场景，引入物理特征工程（Power=Vce*Ic），配合早停与模型轻量化，将测试集 **MAE降至1.21℃**。

### 3. 基于ResNet18的CIFAR-10迁移学习
- **路径**：`03_ResNet_CIFAR10_Transfer_Learning`
- **简介**：加载ImageNet预训练权重微调全连接层，采用AdamW优化器，仅5轮训练，测试集准确率达 **94.21%**。
