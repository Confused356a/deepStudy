# Playground — 深度学习小实战

零散实验和练手代码，按学习阶段分类。

## 目录

| 目录 | 内容 | 典型实验 |
|---|---|---|
| `01-pytorch-basics/` | PyTorch 基础 | 张量操作、autograd、Dataset/DataLoader |
| `02-linear-models/` | 线性模型 | 线性回归、逻辑回归、Softmax 分类 |
| `03-mlp/` | 多层感知机 | 手写 MLP、激活函数对比、BatchNorm |
| `04-cnn/` | 卷积神经网络 | LeNet、AlexNet、VGG 小规模训练 |
| `05-rnn/` | 循环神经网络 | RNN/LSTM/GRU、序列预测、时间序列 |
| `06-transfer-learning/` | 迁移学习 | 预训练模型微调、特征提取、ResNet |
| `07-generative/` | 生成模型 | 自编码器、VAE、GAN 玩具示例 |
| `08-nlp/` | 自然语言处理 | 词嵌入、文本分类、简单 Transformer |

## 使用方式

- 每个实验建一个子目录或 `.py` 文件
- 小实验可以直接写单文件脚本
- 训练前先过拟合小 batch，确认 loss 能下降
- 遇到值得总结的发现，记到对应目录的笔记里
