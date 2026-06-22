import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# ---------- 数据 ----------
dataset = torchvision.datasets.CIFAR10("../../datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Xiaoke(nn.Module):
    def __init__(self):
        super(Xiaoke, self).__init__()

        # self.conv1 = Conv2d(3, 32, 5,padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
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

loss = nn.CrossEntropyLoss()
xiaoke = Xiaoke()
#随机梯度下降优化器
optim = torch.optim.SGD(xiaoke.parameters(), lr=0.01, )

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = xiaoke(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        #print(result_loss)
        running_loss += result_loss
    print(running_loss)