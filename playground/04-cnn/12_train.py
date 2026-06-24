import torch
import torch.nn as nn
from model import *
import torchvision
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------- 数据 ----------
train_data = torchvision.datasets.CIFAR10("../../datasets", train=True,
                                       transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("../../datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

#求一下长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度: {}".format(train_data_size))
print("训练测试集长度: {}".format(test_data_size))

#用dataloader加载一下数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# ---------- 设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备: {}".format(device))

# ---------- TensorBoard ----------
writer = SummaryWriter("../../logs/12_train")

#创建网络模型
xiaoke = Xiaoke()
xiaoke = xiaoke.to(device)

#损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

#优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(xiaoke.parameters(), lr=learning_rate)

#设置训练网络的一些参数
total_train_step = 0 #记录训练次数
total_test_step = 0 #记录测试次数
epoch = 10

for i in range(epoch):
    print("---第 {} 轮训练开始---".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = xiaoke(imgs)
        loss = loss_function(outputs, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if total_train_step % 100 == 0:
            print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = xiaoke(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    avg_test_loss = total_test_loss / len(test_dataloader)

    print("整体测试集上的loss: {}".format(avg_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", avg_test_loss, i)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

writer.close()







