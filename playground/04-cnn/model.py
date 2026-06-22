import torch
import torch.nn as nn
import torchvision
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

#搭建神经网络
class Xiaoke(nn.Module):
    def __init__(self):
        super(Xiaoke, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    xiaoke = Xiaoke()
    input = torch.ones(64, 3, 32, 32)
    output = xiaoke(input)
    print(output.shape)
