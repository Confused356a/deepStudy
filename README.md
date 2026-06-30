# DeepStudy — 深度学习论文复现与思考

研究生阶段系统学习深度学习：经典论文复现、架构思考、实验记录。基于 PyTorch 生态。

## 目录结构

```
deepStudy/
├── d2l/              # 李沐《动手学深度学习》按章节学习 🔄 进行中
│   ├── README.md        # d2l 整体进度 + 章节导航
│   ├── ch02-prelim/     # Ch02 预备知识 ✅
│   ├── ch03-linear/     # Ch03 线性神经网络 ✅
│   └── ch04-mlp/         # Ch04 多层感知机 ✅
├── playground/      # 临时实验、玩具代码
│   └── 04-cnn/          # 13 个 CNN 基础实验 ✅
├── papers/          # 按论文组织的复现代码（待开始）
├── utils/           # 跨论文复用的工具
├── datasets/        # 数据集（不提交）
├── LEARNING_LOG.md  # 📝 每日学习日志（自动维护）
└── CLAUDE.md        # 项目规范 + AI 协作指引
```

## 当前进度

```
小土堆 PyTorch 基础 (13 实验) ✅ ──→ d2l Ch02 预备知识 ✅ ──→ d2l Ch03 线性 ✅ ──→ d2l Ch04 MLP ✅
  (2026-06-23 完成)              (2026-06-24 完成)             (2026-06-28 完成)         (2026-06-30 完成)
```

| 模块 | 状态 | 说明 |
|------|:----:|------|
| PyTorch 基础 | ✅ | Conv2d → Pooling → 激活 → 全连接 → Sequential → Loss → 训练循环 → SGD → 预训练 → 保存/加载 → GPU 训练（13 个实验，`playground/04-cnn/`） |
| d2l Ch02 预备知识 | ✅ | 数据操作、预处理、线性代数、微积分、自动求导、概率统计（6 节，`d2l/ch02-prelim/`） |
| d2l Ch03 线性神经网络 | ✅ | 线性回归 + Softmax 分类，从零 & 简洁双版本对照（7 节，`d2l/ch03-linear/`） |
| d2l Ch04 MLP | ✅ | 感知机→激活函数→MLP从零/简洁→过拟合→权重衰减→Dropout→数值稳定性（8 节，`d2l/ch04-mlp/`） |
| d2l Ch05 深度学习计算 | ⬜ | 自定义层、参数管理、GPU 训练 |
| d2l Ch05+ | ⬜ | CNN / RNN / Transformer / 优化算法 / CV / NLP |
| 论文复现 | ⬜ | 待 d2l 基础章节完成后启动 |

> 📝 详细知识点、实验发现、工程踩坑 → **[LEARNING_LOG.md](LEARNING_LOG.md)**
> 📖 d2l 章节笔记、公式推导、实验结果 → **[d2l/README.md](d2l/README.md)**

## 论文清单

| 状态 | 论文 | 笔记 |
|------|------|------|
| 待开始 | — | 先完成 d2l 基础章节再启动论文复现 |

## 技术栈

- **框架**: PyTorch 2.5.1+cu121
- **GPU**: RTX 3070 Laptop 8GB
- **环境**: Conda `deepstudy1` (Python 3.11)
- **可视化**: matplotlib、TensorBoard
- **数据集**: CIFAR-10、Fashion-MNIST（缓存在 `datasets/`）

## 学习原则

- 先理解动机和核心思想，再抠细节
- 能跑通 > 完美复现；先跑起来，再对齐精度
- 从零实现 + 框架简洁实现双版本对照，理解底层
- 每个模块完成后记录"我学到了什么"
- 工程踩坑即时记录，避免重复犯错
