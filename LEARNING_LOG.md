# DeepStudy 学习日志

> 深度学习系统学习——逐组件理解，从基础到论文复现。基于 PyTorch + CUDA。

*最后更新：2026-06-22*  |  自动维护，每天 22:17 检查更新

---

## 当前进度总览

| 模块 | 状态 | 完成内容 |
|------|:----:|----------|
| 01-pytorch-basics | ⬜ | 待开始 |
| 02-linear-models | ⬜ | 待开始 |
| 03-mlp | ⬜ | 待开始 |
| **04-cnn** | 🟢 | Conv2d → 训练/保存/加载（13个实验） |
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

## 环境

- GPU: RTX 3070 Laptop 8GB
- CUDA: 12.2 + PyTorch 2.5.1+cu121
- Conda: `deepstudy1` (Python 3.11)
- 数据集: CIFAR-10（华为云镜像）
