# d2l — 李沐《动手学深度学习》学习记录

> 官方教材：[d2l.ai](https://d2l.ai/)  |  代码框架：PyTorch  |  学习路线：按章节推进，先理论后代码

---

## 进度总览

| 章节 | 目录 | 状态 | 关键产出 |
|------|------|:----:|----------|
| **Ch02** 预备知识 | `ch02-prelim/` | ✅ | 数据操作、预处理、线性代数、微积分、自动求导、概率 |
| **Ch03** 线性神经网络 | `ch03-linear/` | ✅ | 线性回归+Softmax 分类，从零&简洁双版本，7 节全部完成 |

| **Ch04** 多层感知机 | `ch04-mlp/` | ⬜ | 过拟合/欠拟合、权重衰减、Dropout |
| **Ch05** 深度学习计算 | `ch05-computation/` | ⬜ | 自定义层、参数管理、GPU 训练 |
| **Ch06** CNN 基础 | `ch06-cnn/` | ⬜ | LeNet（与小土堆 CNN 重叠，快速重温）|
| **Ch07** 现代 CNN | `ch07-modern-cnn/` | ⬜ | AlexNet / VGG / NiN / GoogLeNet / ResNet / DenseNet |
| **Ch08** 循环神经网络 | `ch08-rnn/` | ⬜ | RNN、BPTT、序列数据 |
| **Ch09** 现代 RNN | `ch09-modern-rnn/` | ⬜ | GRU / LSTM / 双向 RNN / 机器翻译 |
| **Ch10** 注意力机制 | `ch10-attention/` | ⬜ | 注意力 / Transformer / BERT |
| **Ch11** 优化算法 | `ch11-optim/` | ⬜ | Momentum / Adam / 学习率调度 |
| **Ch13** 计算机视觉 | `ch13-cv/` | ⬜ | 数据增广、微调、目标检测、语义分割 |
| **Ch14** NLP | `ch14-nlp/` | ⬜ | 词嵌入、Word2Vec、预训练 BERT |

> 跳过 Ch12 性能（偏工程优化，后期补）。

---

## 目录约定

```
d2l/
├── README.md           # 本文件：进度 + 实验参数汇总
├── chXX-name/
│   ├── README.md        #   本章笔记：公式推导、关键概念、实验结果
│   ├── model.py         #   网络定义（按论文/架构命名）
│   ├── train.py         #   训练脚本
│   └── notes/           #   推导草稿、踩坑记录
```

---

## 学习方法

1. 先读教材对应章节，理解动机和核心公式
2. 小规模验证（过拟合小 batch，确认 loss 下降）
3. 完整训练 + TensorBoard 记录 loss / accuracy / 对比实验
4. 每个章节写 README 总结：我学到了什么

---

## 环境

- GPU: RTX 3070 Laptop 8GB
- CUDA 12.2 + PyTorch 2.5.1+cu121
- Conda: `deepstudy1` (Python 3.11)
