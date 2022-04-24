import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from network import cnn
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("log")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# ToTensor():[0,255]->[C,H,W];Normalize: 标准化(均值+标准差);数据增强(随机翻转图片，随机调整亮度)
transform1 = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomGrayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform2 = transforms.Compose([
                                transforms.Resize([32,32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform3 = transforms.Compose([
                                transforms.Resize([32,32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])

# cifar_size: 50000+10000 32*32*3 10classes
data_train = datasets.CIFAR10('./c10data', train=True, transform=transform2, download=True)
data_test = datasets.CIFAR10('./c10data', train=False, transform=transform2, download=True)
# data_train = datasets.MNIST('./mnist', download=True, train=True, transform=transform3)
# data_test = datasets.MNIST('./mnist', download=True, train=False, transform=transform3)
subset_list = list(range(0, len(data_train),6))
data_train = torch.utils.data.Subset(data_train, subset_list)

# 定义batch, 即一次训练的样本量大小
train_batch_size = 128
test_batch_size = 128

# 对数据进行装载，利用batch _size来确认每个包的大小，用Shuffle来确认打乱数据集的顺序。
train_loader = DataLoader(data_train, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=test_batch_size, shuffle=False)

# 定义10个分类标签
classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

model=cnn.CNN().to(device)

# 设置训练次数
num_epochs = 10
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化方法
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义存储损失函数和准确率的数组
train_losses = []
train_acces = []
eval_losses = []
eval_acces = []



# 训练模型
print("start training...")
for epoch in range(num_epochs):

    # 记录训练开始时刻
    start_time = time.time()

    train_loss = 0
    train_acc = 0

    model.train()

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()   # 模型参数梯度清零
        loss.backward()    # 误差反向传递
        optimizer.step()    # 更新参数

        train_loss += loss

        _, pred = outputs.max(1)
        num_correct = (pred == labels).sum().item()
        acc = num_correct / labels.shape[0]
        train_acc += acc

    # 取平均存入
    train_losses.append(train_loss / len(train_loader))
    train_acces.append(train_acc / len(train_loader))

    writer.add_scalar('Train/Loss', train_loss/len(train_loader), epoch)
    writer.add_scalar('Train/Acc', train_acc / len(train_loader), epoch)

    # 测试集：
    eval_loss = 0
    eval_acc = 0

    # 将模型设置为测试模式
    model.eval()

    # 处理方法同上
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss

            _, pred = outputs.max(1)
            num_correct = (pred == labels).sum().item()  # 记录标签正确的个数
            acc = num_correct / labels.shape[0]
            eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))

    writer.add_scalar('Test/Loss', eval_loss / len(test_loader), epoch)
    writer.add_scalar('Test/Acc', eval_acc / len(test_loader), epoch)
    # 输出效果
    print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f},'
          'Test Loss:{:.4f},Test Acc:{:.4f}'
          .format(epoch, train_loss / len(train_loader),
                  train_acc / len(train_loader),
                  eval_loss / len(test_loader),
                  eval_acc / len(test_loader)))
    # 输出时长
    stop_time = time.time()
    print("time is:{:.4f}s".format(stop_time-start_time))
print("end training.")