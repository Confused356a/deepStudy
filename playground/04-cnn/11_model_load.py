"""
PyTorch 模型加载的两种方式，对应 10_model_save.py 的保存：

方式一：model = torch.load(path, weights_only=False)
  - 直接反序列化整个模型对象，拿来即用
  - 必须设 weights_only=False（PyTorch 2.5+ 安全机制，默认只允许加载纯权重）
  - 缺点：依赖保存时的 exact 代码环境

方式二：model.load_state_dict(torch.load(path))
  - 先手动创建模型架构，再把参数字典灌进去
  - torch.load 加载 state_dict 用默认 weights_only=True 即可（更安全）
  - 优点：架构可以改（比如换分类头），只加载匹配的层；业界标准
"""

import torch
import torchvision

#load方式一加载
model1 = torch.load("vgg16_method1.pth", weights_only=False)
print(model1)

#加载方式二
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)