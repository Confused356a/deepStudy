import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../../datasets", train=False,
                                             transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Xiaoke(nn.Module):
    def __init__(self):
        super(Xiaoke, self).__init__()
        self.linear1 = nn.Linear(3 * 32 * 32, 10)  # 3072 → 10

    def forward(self, input):
        output = self.linear1(input)
        return output


xiaoke = Xiaoke()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # 展平: (batch, 3, 32, 32) → (batch, 3072)
    output = torch.flatten(imgs, start_dim=1)
    print(output.shape)
    output = xiaoke(output)
    print(output.shape)
    break
