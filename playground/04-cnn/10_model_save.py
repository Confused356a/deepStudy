"""
PyTorch 模型保存的两种方式：

方式一：torch.save(model, path)
  - 保存整个模型对象（网络结构 + 参数），序列化为 pickle
  - 加载时直接得到模型，不需要再定义网络类
  - 缺点：文件大，依赖 exact 代码结构，换个项目可能 load 不了

方式二：torch.save(model.state_dict(), path)
  - 只保存参数字典 OrderedDict（层的名字 → 权重 tensor）
  - 加载时需先创建模型实例，再 load_state_dict 灌入参数
  - 优点：文件更小，跨代码版本兼容，业界标准做法
"""

import torchvision
import torch

vgg16 = torchvision.models.vgg16(weights=None)

#保存方式一(保存模型＋参数)
torch.save(vgg16, "vgg16_method1.pth")

#保存方式二(保存成字典模式)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")