import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------- 数据 ----------
dataset = torchvision.datasets.CIFAR10("../../datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

# ---------- 模型 ----------
class Xiaoke(nn.Module):
    def __init__(self):
        super(Xiaoke, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model1(x)

# ---------- 训练准备 ----------
xiaoke = Xiaoke()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(xiaoke.parameters(), lr=0.01)
writer = SummaryWriter("../logs_train")

total_step = 0
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        imgs, targets = data
        outputs = xiaoke(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_step += 1

        if i % 100 == 0:
            print(f"[Epoch {epoch+1}, Batch {i:3d}] loss = {loss.item():.4f}")

        writer.add_scalar("train/loss", loss.item(), total_step)

    avg_loss = running_loss / len(dataloader)
    print(f"=== Epoch {epoch+1} 完成，avg loss = {avg_loss:.4f} ===\n")

writer.close()
print("训练完成，TensorBoard: tensorboard --logdir ../logs_train")
