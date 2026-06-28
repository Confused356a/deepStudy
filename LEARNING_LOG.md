# DeepStudy 学习日志

> 深度学习系统学习——逐组件理解，从基础到论文复现。基于 PyTorch + CUDA。

*最后更新：2026-06-28 21:50*  |  自动维护，每天检查更新

---

## 当前进度总览

| 模块 | 状态 | 完成内容 |
|------|:----:|----------|
| 01-pytorch-basics | ⬜ | 待开始 |
| 02-linear-models | ⬜ | 待开始 |
| 03-mlp | ⬜ | 待开始 |
| **04-cnn** | ✅ | Conv2d → 训练/保存/加载（13个实验）|
| **d2l-Ch02** | ✅ | 预备知识 6 小节（2026-06-24 完成） |
| **d2l-Ch03** | ✅ | 线性神经网络全 7 节：线性回归+Softmax 分类 — 从零 & 简洁双版本 (2026-06-28) |
| d2l-Ch04+ | ⬜ | MLP → 现代 CNN → Transformer（待继续）|
| 05-rnn | ⬜ | 待开始 |
| 06-transfer-learning | ⬜ | 待开始 |
| 07-generative | ⬜ | 待开始 |
| 08-nlp | ⬜ | 待开始 |

---

## 04-CNN：卷积神经网络

### 学习路线

```
Conv2d → MaxPool2d → ReLU/Sigmoid → Linear → Sequential → Loss → 训练循环 → SGD优化器 → 预训练模型 → 保存/加载 → 模型独立文件 → 完整训练脚本
 01        02           03           04         05         06       07          08           09         10/11       model        12
```

### 01 — Conv2d 卷积层

- `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`
- 输出尺寸公式：**H_out = (H_in + 2P − K) / S + 1**
  - 32×32, K=3, P=0, S=1 → 30×30
- CIFAR-10 shape：`[batch, 3, 32, 32]`
- TensorBoard `add_images` 可视化特征图

### 02 — MaxPool2d 池化层

- `nn.MaxPool2d(kernel_size, ceil_mode)`
- `ceil_mode=True` 向上取整保留边缘
- 先小 tensor 验证，再上真实数据
- 核心：保留局部最强信号，减少数据量
- 坑：整数 tensor 进 MaxPool2d 报错，需 `dtype=float32`

### 03 — 激活函数 ReLU / Sigmoid

- `nn.ReLU()` — 负值置 0，缓解梯度消失
- `nn.Sigmoid()` — 压缩到 (0,1)，深层网络梯度消失风险
- 引入**非线性**：没有激活函数，多层等价于单层

### 04 — Linear 全连接层

- `nn.Linear(in_features, out_features)` — y = xW^T + b
- 图像展平：`torch.flatten(imgs, start_dim=1)`→ `[batch, 3072]`
- 3072 = 3×32×32
- Linear(3072, 10)：像素直接映射到类别

### 05 — Sequential 容器

- `nn.Sequential()` 打包多层，forward 自动顺序执行
- 3 层 CNN：Conv→Pool×3 → Flatten → Linear×2
- 尺寸链：32 → 32 → 16 → 16 → 8 → 8 → 4 → 1024 → 64 → 10
- `writer.add_graph(model, input)` — TensorBoard 可视化计算图

### 06 — 损失函数

- `nn.L1Loss` — MAE，|pred−target|
- `nn.MSELoss` — MSE，(pred−target)²
- **关键坑：** `torch.tensor([1,2,3])` 默认 int64，损失函数要求 float32
  - 解决：加 `dtype=torch.float32`

### 07 — 完整训练循环

- **CrossEntropyLoss**：内置 LogSoftmax + NLLLoss
  - outputs 传 raw logits，targets 传 Long 类型类别索引
- **训练四步曲**：`zero_grad → forward → loss → backward → step`
- `running_loss += loss.item()` — `.item()` 断开计算图
- TensorBoard `add_scalar` 记录 loss 曲线
- 2 epoch，avg loss 2.31→2.25
- **发现：** 没有 ReLU 的纯卷积+线性，loss 下降极慢

### 08 — SGD 优化器深入

- `torch.optim.SGD(params, lr=0.01)`
- `optim.zero_grad()` 清梯度 → `optim.step()` 更新 w = w − lr×grad
- 20 epoch 训练观察 loss 趋势
- 坑：`running_loss += result_loss`（保留 tensor）会堆积计算图

### 09 — 预训练模型 VGG16

- `torchvision.models.vgg16(pretrained=True)` — ImageNet 预训练权重 ~528MB
- ImageNet 1000 类 → CIFAR-10 10 类，需替换分类头
- `model.add_module(name, layer)` — 动态加层
- **关键问题：** VGG16 的 `forward()` 固定走 features→classifier，add_module 加的层不会被调用
  - 正确：`model.classifier[6] = nn.Linear(4096, 10)`

### 10 — 模型保存

- **方式一（完整模型）：** `torch.save(model, path)` — 结构+参数 pickle 序列化
- **方式二（推荐）：** `torch.save(model.state_dict(), path)` — 仅参数字典，更轻量
- `state_dict()` 是**无参方法**，返回 `OrderedDict{层名: 参数tensor}`
- 坑：`state_dict("path")` 传参报错 — 括号不能传路径
- `pretrained` 参数已废弃，改用 `weights=None` / `weights="DEFAULT"`

### 11 — 模型加载

- **方式一：** `torch.load(path, weights_only=False)` — 完整反序列化
- **方式二（推荐）：** 先建架构 → `model.load_state_dict(torch.load(path))`
- **`weights_only` 安全机制：** PyTorch 2.5+ 默认 `True`，加载完整模型必须显式设 `False`
- 方式二加载 state_dict 用默认 `weights_only=True` 即可，更安全
- 文件名坑：`vgg16_method_pth`（下划线）✗ → `vgg16_method2.pth`（点号）✓

### model — 模型独立成文件

- 模型定义和训练脚本分离：`model.py` 只放网络 + 形状测试
- `if __name__ == "__main__":` — 可直接测试，也可被 `from model import Xiaoke` 安全导入
- `torch.ones(batch, C, H, W)` 做 dummy input，秒级验证输出 shape
- 再次踩坑 `import nn`（pip 的 nn 包覆盖 torch.nn）— 必须写 `import torch.nn as nn`

### 12 — 完整训练脚本

- `from model import *` — 从独立文件导入模型，不重复定义
- 训练集/测试集分离：`train=True` / `train=False`
- `len(dataset)` 获取数据集大小
- 学习率作为变量管理：`learning_rate = 0.001`
- 10 epoch 训练循环，记录 `total_train_step`
- **GPU 训练三步：** `device = torch.device("cuda" if cuda.is_available() else "cpu")` → `model.to(device)` → `data.to(device)`
  - 损失函数也需要 `.to(device)`（CrossEntropyLoss 无所谓，但涉及可学习参数的 loss 需要）
- **坑：** 训练循环传了 `xiaoke(imgs)`，测试循环写成 `xiaoke()` 忘记传数据 → `TypeError: forward() missing 1 required positional argument: 'x'`
- **TensorBoard 可视化：** `SummaryWriter("logs/12_train")` → `add_scalar("train_loss", loss.item(), step)` 每步记录 → `add_scalar("test_loss", avg, epoch)` 每轮记录
- **逢百输出：** `if total_train_step % 100 == 0` 减少刷屏，`.item()` 取标量避免打印 tensor
- **测试 loss 平均：** `total_test_loss / len(test_dataloader)` 比直接打印累加 tensor 更可读
- **测试正确率计算（2026-06-24 新增）：**
  - `(outputs.argmax(1) == targets).sum()` — argmax 沿 dim=1 取预测类别，与标签比对的 True/False 求和得到每 batch 正确数
  - `total_accuracy / test_data_size` — 累加所有 batch 正确数除以测试集总样本数，得到整体正确率（0~1 小数）
  - `writer.add_scalar("test_accuracy", ..., total_test_step)` — 正确率同步写入 TensorBoard
  - **注意：** `total_accuracy` 是 Python int（`.sum()` 返回的是 tensor 但累加和除法会自动转换），不会像 loss 那样堆积计算图

---

## 贯穿全程的工程经验

| 经验 | 说明 |
|------|------|
| 数据集缓存优先 | 加载前先检查 `datasets/` 是否已有 |
| dtype 陷阱 | `torch.tensor` 默认 int64，损失函数要 float32 |
| import nn 陷阱 | pip 的 `nn` 包会覆盖 `torch.nn`，必须写 `import torch.nn as nn` |
| 小 tensor 先验证 | 复杂模型前先用小数据确认尺寸逻辑 |
| TensorBoard | add_images（图）、add_graph（结构）、add_scalar（曲线） |
| running_loss | 用 `.item()` 取标量累加，epoch 末取平均 |
| 预训练模型 | 替换分类头（非 add_module），冻结 backbone 可选 |
| 模型保存 | state_dict 优于完整模型；`weights_only=False` 需显式声明 |
| 模型独立文件 | `from model import` 复用，`if __name__` 测试形状 |
| state_dict() | 无参方法，不能传路径，返回 OrderedDict |
| GPU 训练三步骤 | `device` → `model.to(device)` → `imgs.to(device)` + `targets.to(device)` |

---

## 🎓 小土堆 PyTorch 系列完结总结

**完成日期：2026-06-23**

### 覆盖内容

| # | 实验文件 | 主题 |
|---|---------|------|
| 01 | `01_conv2d_cifar10.py` | Conv2d 卷积层 |
| 02 | `02_pooling_cifar10.py` | MaxPool2d 池化层 |
| 03 | `03_relu.py` | ReLU / Sigmoid 激活函数 |
| 04 | `04_linear.py` | Linear 全连接层 |
| 05 | `05_seq.py` | Sequential 容器 |
| 06 | `06_loss.py` | L1Loss / MSELoss / CrossEntropyLoss |
| 07 | `07_loss_network.py` | 完整训练循环（四步曲） |
| 08 | `08_optim.py` | SGD 优化器 |
| 09 | `09_model_pretrained.py` | 预训练模型 VGG16 |
| 10 | `10_model_save.py` | 模型保存（完整 vs state_dict） |
| 11 | `11_model_load.py` | 模型加载 + weights_only 安全机制 |
| 12 | `model.py` | 模型独立成文件 + dummy input 验证 |
| 13 | `12_train.py` | GPU 完整训练脚本 + TensorBoard |

### 核心能力

- 能用 PyTorch 从零搭建 CNN、运行训练循环、可视化 loss 曲线
- 理解 Conv2d / MaxPool2d / Linear 每层的尺寸变化
- 会加载预训练模型并替换分类头做迁移学习
- 掌握模型保存/加载的两种方式及安全性考量
- 能用 GPU 训练并排查 dtype / import / forward 参数等常见坑
- 模型独立文件 + `if __name__ == "__main__"` 的工程化习惯

### 下一步

→ **李沐《动手学深度学习》**(d2l.ai) 🔄 进行中 — 2026-06-24 启动，Ch02 预备知识已完成

---

## d2l — 李沐《动手学深度学习》

### Ch02 预备知识（2026-06-24 完成）

> 代码位置: `d2l/ch02-prelim/`，每个 `.py` 文件可直接运行

| # | 文件 | 主题 | 关键收获 |
|---|------|------|----------|
| 01 | `01_ndarray.py` | 数据操作 | 广播规则（末维对齐）、`+=` 原地操作、tensor↔numpy 共享内存 |
| 02 | `02_data_preprocessing.py` | 数据预处理 | pandas → tensor 流水线：`read_csv → fillna → get_dummies → torch.tensor` |
| 03 | `03_linear_algebra.py` | 线性代数 | `dim` 是聚合操作核心、`keepdim=True` 方便广播、`@` 才是矩阵乘 |
| 04 | `04_calculus.py` | 微积分 | 梯度下降直觉可视化、链式法则=反向传播理论基础 |
| 05 | `05_autograd.py` | ⭐ 自动求导 | **grad 累加机制** = `zero_grad()` 存在理由、`detach` vs `no_grad`、动态图优势 |
| 06 | `06_probability.py` | 概率统计 | NLL = -log P(y_true/x) → 交叉熵本质、大数定律可视化 |

### Ch02 工程踩坑

| 坑 | 表现 | 解决 |
|----|------|------|
| GBK 编码 | emoji/特殊 Unicode（✅✓²∇∂）→ `UnicodeEncodeError` | Windows 终端不支持，全改 ASCII |
| `randn_like(Long)` | X 默认 int64 → `NotImplementedError` | `X.float()` 转换 |
| 二次 `backward()` | 默认首轮释放计算图 → `RuntimeError` | `backward(retain_graph=True)` |

### Ch02 与小土堆的关系

前半部分（tensor 操作、autograd）在小土堆中已大量使用，但 d2l 讲得更系统。
真正新增的：pandas 预处理、广播规则、`keepdim`/`cumsum`、`detach` vs `no_grad`、NLL 数学推导。

### Ch03 线性神经网络（2026-06-28 完成 ✅）

> 代码位置: `d2l/ch03-linear/`，共 7 节，从零 + 简洁双版本对照

| # | 文件 | 主题 | 关键收获 |
|---|------|------|----------|
| 01 | `01_linear_regression_theory.py` | 3.1 线性回归理论 | 解析解推导、梯度下降直觉、学习率影响、batch size 选择 |
| 02 | `02_scratch_linear_regression.py` | 3.2 从零实现 (完整 7 步) | 数据生成→迭代器→初始化→模型→损失→SGD→训练循环 |
| 03 | `03_concise_linear_regression.py` | 3.3 简洁实现 | TensorDataset+DataLoader、nn.Linear、nn.MSELoss、optim.SGD |
| 04 | `04_softmax_regression.py` | 3.4 Softmax 回归理论 | softmax 函数、交叉熵直觉、温度参数、数值稳定技巧 |
| 05 | `05_fashion_mnist.py` | 3.5 图像分类数据集 | Fashion-MNIST 加载、ToTensor、DataLoader、类别分布 |
| 06 | `06_softmax_scratch.py` | 3.6 Softmax 从零实现 | 手写交叉熵、softmax 稳定版、Accumulator、混淆矩阵、错误分析 |
| 07 | `07_softmax_concise.py` | 3.7 Softmax 简洁实现 | nn.CrossEntropyLoss=LogSoftmax+NLLLoss、nn.Flatten、权重可视化 |

---

### 3.1–3.2 线性回归：理论与从零实现

**解析解 vs 梯度下降**
- 线性回归: w* = (X^T X)^(-1) X^T y — 唯一有闭式解的"神经网络"
- 深度学习全部用梯度下降 — 因为模型有非线性，没有解析解

**小批量 SGD 的 3 步**
```
with torch.no_grad():           # 参数更新不追踪
    param -= lr * param.grad / batch_size   # /batch_size 因 loss 是 sum
    param.grad.zero_()          # 清零! 否则下次 backward 累加
```

**训练循环 = 深度学习的核心**
```python
for epoch:
    for X, y in data_iter():
        l = loss(model(X, w, b), y)   # forward + loss
        l.sum().backward()             # backward
        sgd([w, b], lr, batch_size)    # update (内含 zero_grad)
```
这 4 行代码覆盖 CNN/RNN/Transformer —— 所有深度学习训练都是这个循环。

**新学到的点**
- `yield` 生成器惰性求值 — 百万级数据集的必备工程技能
- loss 除以 2 — 为了让导数常数抵消，这种"为梯度设计"的思路贯穿 DL
- `no_grad()` 块 — 框架不会自动区分"前向计算"和"参数更新"，必须显式声明
- 3 epoch 从随机初始值学到误差 10^-4 级别的参数 — 线性问题 + 线性模型 = 几乎完美拟合

---

### 3.3 线性回归简洁实现：从零 → 框架的对照

**框架一行替代手写十几行：**

| 从零 (~40 行) | 框架 (~10 行) |
|---|---|
| `w=normal(...); b=zeros(...); def linreg(X,w,b)` | `model = nn.Linear(2, 1)` |
| `def squared_loss(y_hat, y): ((y_hat-y)^2)/2` | `loss_fn = nn.MSELoss(reduction='none')` |
| `def sgd(params, lr, bs): p-=lr*p.grad/bs; p.grad.zero_()` | `optimizer.zero_grad()` + `optimizer.step()` |
| `def data_iter(): randperm+yield` | `DataLoader(TensorDataset(X,y), batch_size=bs, shuffle=True)` |

**关键差异：损失函数的常数因子**
- 从零: `((y_hat - y)^2) / 2` ← 除以 2 让梯度简洁（导数中常数抵消）
- 框架: `(y_hat - y)^2` ← 标准 MSE，不减半
- 梯度只差常数因子 2，可通过学习率吸收 → 最终收敛结果等价

**实验验证：** 从零和简洁实现在相同数据上训练 5 epoch，loss 下降曲线几乎重合，最终参数都收敛到真实值附近。

---

### 3.4 Softmax 回归：从回归到分类的桥梁

**核心公式：**

```
Softmax:  y_hat_i = exp(o_i) / sum_j(exp(o_j))

性质: ① 非负 (exp > 0)  ② 和为 1 (归一化)  ③ 保持顺序 (exp 单调)
```

**交叉熵损失 (Cross-Entropy)：**
```
CE(y_hat, y) = -log(y_hat[y])   ← 取真实类别对应概率的负对数

y_hat[y] = 1.0  → -log(1.0) = 0      完美预测, loss=0
y_hat[y] = 0.5  → -log(0.5) = 0.693  不太确定
y_hat[y] = 0.01 → -log(0.01) = 4.605 几乎错了, loss 很大
```

**⭐ 数值稳定技巧（非常重要！）：**
```
softmax(o) = softmax(o - max(o))   ← 减去每行最大值

为什么？exp(1000) → inf, exp(-1000) → 0 → log(0) = -inf
减掉 max: exp 不会溢出, 且数学上等价 (分子分母同时除以 exp(max))
```

**温度参数 T：**
```
softmax(o / T)  — T 控制"硬度"
T → 0: 趋近 one-hot (硬决策，最大概率→1)
T → ∞: 趋近均匀分布 (所有类别等概率)
T = 1: 标准 softmax
```
这是知识蒸馏、强化学习中常用的技巧。

**Softmax vs 逻辑回归：**
- 逻辑回归 = Softmax 在 k=2 时的特例: `softmax([o1, o0])_1 = sigmoid(o1 - o0)`

---

### 3.5 Fashion-MNIST：图像分类入门数据集

- 70,000 样本 (60k 训练 + 10k 测试)，10 类服装，28×28 灰度
- `torchvision.datasets.FashionMNIST(root, train=True, transform, download)`
- `transforms.ToTensor()` — PIL Image → float32 tensor [0, 1]
- `DataLoader(dataset, batch_size=256, shuffle=True)` — 批量加载
- 数据形状: `[batch, 1, 28, 28]` → Softmax 需展平为 `[batch, 784]`
- 这是最简单的"图像模型"：每个像素是独立特征，无视空间结构（CNN 会利用空间结构）
- 类别均衡：每类 ~6,000 样本，无需处理不平衡

---

### 3.6 Softmax 从零实现：在 Fashion-MNIST 上训练

**手写组件：**
```python
# 数值稳定的 softmax
def softmax(X):
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)  # 减去 max!
    return X_exp / X_exp.sum(dim=1, keepdim=True)

# 交叉熵 — 取真实类别概率的负对数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  # 高级索引

# 模型 = softmax(X @ W + b)
def net(X):
    return softmax(X.reshape(-1, 784) @ W + b)
```

**Accumulator 累加器模式：**
```python
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]
```
用于 epoch 级统计（正确数/总数），比直接用 Python 累加更干净。

**训练结果：** 20 epoch，lr=0.1，batch=256
- 测试准确率: **~83%**（随机猜测 10%，提升 73pp）
- 线性模型在图像数据上也能达到不错效果
- 每类准确率差异显著：简单类（Trouser/Bag ~90%）vs 困难类（Shirt ~65%）
- Top 混淆对：Shirt↔T-shirt/top、Shirt↔Pullover、Sneaker↔Ankle boot

**可视化产出：** Loss 曲线 + 准确率曲线 + 每类准确率柱状图 + 混淆矩阵 + 错误样本展示

---

### 3.7 Softmax 简洁实现：一行模型 + 一个损失函数

**⭐ 核心认知：`nn.CrossEntropyLoss = LogSoftmax + NLLLoss`**

```python
model = nn.Sequential(
    nn.Flatten(),         # [batch, 1, 28, 28] → [batch, 784]
    nn.Linear(784, 10)    # [batch, 784] → [batch, 10]
)
loss_fn = nn.CrossEntropyLoss()   # 内部做 LogSoftmax + NLL
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

**⚠️ 经典坑：CrossEntropyLoss 期望 raw logits（不是 softmax 后的概率！）**
```
从零:  y_hat = softmax(X @ W + b)      ← 需要手动 softmax
       l = -log(y_hat[range(n), y])     ← 然后取负对数
       l.sum().backward()

框架:  logits = model(X)                ← raw logits, 不要 softmax!
       l = loss_fn(logits, y)           ← 内部做 LogSoftmax + NLL
       l.backward()                     ← loss 已取 mean
```

**另一关键差异：**
- 从零 loss 是 sum → 需要 `l.sum().backward()`，SGD 内 `/batch_size`
- 框架 loss 是 mean → 直接 `l.backward()`，optimizer 不管 batch_size

**权重可视化：** 学到的 `W` (10×784) 可 reshape 为 `(10, 28, 28)` 灰度图 — 每个类别对应一个 28×28 的"模板"，浅色=正权重（该像素是此类），深色=负权重（该像素不是此类）。

**代码量对比：** 从零 ~120 行 vs 框架 ~60 行 — 减半！

**训练结果：** 与从零实现性能一致（~83%），训练/测试差距小 → 几乎没有过拟合。

---

### Ch03 总体收获

1. **深度学习三组件统一框架：模型 → 损失 → 优化** — 线性回归和 Softmax 分类共享同一训练循环
2. **从零 → 简洁的双版本对照** — 每个模型都写两遍：先手写理解底层，再用框架体会抽象
3. **Softmax + 交叉熵 = 分类问题标配** — 数值稳定技巧（减 max）是工业界标准做法
4. **CrossEntropyLoss 输入 raw logits** — 这是 PyTorch 新手最容易犯的错
5. **loss sum vs mean** — 从零用 sum+SGD 内除以 batch_size，框架用 mean+optim 不关心 batch_size
6. **线性模型在图像上的上限** — 784 维 → 10 类，Softmax 回归 ~83% 准确率。这是 CNN 的 baseline：后面每加一层卷积，就能突破这个天花板
7. **混淆矩阵是分类调试利器** — 能看到模型在哪些类别对上犯错，不是只看一个准确率数字

### Ch03 工程踩坑

| 坑 | 表现 | 解决 |
|----|------|------|
| CrossEntropyLoss 输入 softmax 后的概率 | loss 不降或 NaN（双重 softmax） | 传入 raw logits，框架内部做 LogSoftmax |
| 从零 loss 用 sum，框架 loss 用 mean | `l.backward()` 梯度量级不一致 | 从零: `l.sum().backward()` + SGD 内 `/batch_size`；框架: 直接用 mean |
| softmax 数值溢出 | 大数据时 `exp(1000) → inf` | 减去每行最大值: `softmax(o - max(o))` |
| 中文字体 | matplotlib 中文显示方块 | `font.sans-serif` 设置 SimHei/DejaVu Sans |
| 测试时忘记 `model.eval()` + `no_grad()` | Dropout/BatchNorm 行为不对、梯度积压 | 测试循环前加 `model.eval()` + `with torch.no_grad():` |



---

## 环境

- GPU: RTX 3070 Laptop 8GB
- CUDA: 12.2 + PyTorch 2.5.1+cu121
- Conda: `deepstudy1` (Python 3.11)
- 数据集: CIFAR-10（华为云镜像）
