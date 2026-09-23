基于ResNet18的CIFAR-10迁移学习

- 加载ImageNet预训练权重，微调全连接层适配10分类。
- AdamW优化器，微调学习率3e-5。
- 仅需5轮训练，测试集准确率达 **94.21%**。
- 代码文件：[ResNet_CIFAR10_Transfer_Learning.ipynb](./ResNet_CIFAR10_Transfer_Learning.ipynb)
