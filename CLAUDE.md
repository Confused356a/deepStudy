# DeepStudy — 深度学习论文复现与思考

## 项目定位

本目录用于深度学习系统学习：经典论文复现、架构思考、实验记录。基于 PyTorch 生态。

## 目录约定

```
deepStudy/
├── d2l/              # 李沐《动手学深度学习》按章节学习
│   └── chXX-name/
│       ├── README.md    # 本章笔记、公式推导、实验结果
│       ├── model.py     # 网络定义
│       ├── train.py     # 训练脚本
│       └── notes/       # 推导草稿
├── papers/          # 按论文组织的复现代码，每个子目录一篇论文
│   └── <paper-slug>/
│       ├── README.md    # 论文要点、复现笔记
│       ├── model.py     # 模型定义
│       ├── train.py     # 训练脚本
│       └── notes/       # 推导草稿、实验日志
├── playground/      # 临时实验、玩具代码
├── utils/           # 跨论文复用的工具
└── datasets/        # 数据集（gitignore，太大不提交）
```

## 工作方式

- 每个论文复现为一个独立子目录，含完整代码和笔记
- 推导过程写到 `notes/` 下，保留思考痕迹
- 训练前先小规模验证（过拟合小 batch，确认 loss 能下降）
- 实验参数和结果写在 README 里，方便回顾对比

## 技术栈

- Python 3.10+
- PyTorch (最新稳定版)
- torchvision, torchaudio
- 绘图: matplotlib, seaborn
- 实验管理: tensorboard / wandb（按需）

## 论文复现清单 (待填充)

1.

## 编码风格

- 变量名直译论文符号（如 `logits`, `probs`, `temperature`）
- 关键公式在注释中标注论文公式编号
- 模型 forward 保持可读性，不要过度抽象
- 分模块写，不要一个文件上千行

## 学习原则

- 先理解动机和核心思想，再抠细节
- 能跑通 > 完美复现；先跑起来，再对齐精度
- 每个论文复现后写一段"我学到了什么"
