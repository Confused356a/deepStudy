import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

# ========== 先理解 MaxPool2d 在单个 tensor 上的行为 ==========
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))  # [1, 1, 5, 5]  batch=1, channel=1, 5×5
print("输入 shape:", input.shape)
print("输入:\n", input)

class Xiaoke(nn.Module):
    def __init__(self):
        super(Xiaoke, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)
        return output

xiaoke = Xiaoke()
output = xiaoke(input)
print("输出 shape:", output.shape)
print("输出:\n", output)

# ========== 在 CIFAR-10 上可视化 ==========
dataset = torchvision.datasets.CIFAR10("datasets", train=False,
                                             transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("../logs_maxpool")
step = 0

for data in dataloader:
    imgs, target = data
    output = xiaoke(imgs)
    print("imgs:", imgs.shape, "→ output:", output.shape)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output[:, :3, :, :], step)  # 取前3通道显示
    step += 1

writer.close()

# 最大池化：保留局部最强信号，减少数据量，保留关键特征
