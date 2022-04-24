import time
import torch
from torchvision.utils import save_image
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from network import Autoencoder
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("log")


# def add_noise(inputs, noise_factor=0.3):
#     noisy = inputs + torch.randn_like(inputs) * noise_factor
#     noisy = torch.clip(noisy, 0., 1.)
#     return noisy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ToTensor():[0,255]->[C,H,W];Normalize: 标准化(均值+标准差);数据增强(随机翻转图片，随机调整亮度)
transform1 = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomGrayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform2 = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# cifar_size: 50000+10000 32*32*3 10classes
data_train = datasets.CIFAR10('./c10data', train=True, transform=transform2, download=True)

#batch
train_batch_size = 128
train_loader = DataLoader(data_train, batch_size=train_batch_size, shuffle=True)

classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

model=Autoencoder.autoEncoder().to(device)
num_epochs = 40
# loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
print("start training...")
for epoch in range(num_epochs):

    start_time = time.time()

    train_loss = 0

    model.train()

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs = add_noise(inputs, 0.1)
        outputs,z = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()   # 模型参数梯度清零
        loss.backward()    # 误差反向传递
        optimizer.step()    # 更新参数

        train_loss += loss

    print('epoch:{},Train Loss:{:.4f},'.format(epoch, train_loss / len(train_loader), ))

    writer.add_scalar('Train/Loss', train_loss/len(train_loader), epoch)
    stop_time=time.time()
    print("time is:{:.4f}s".format(stop_time-start_time))

    model.eval()
    with torch.no_grad():
        real=inputs[:10]
        recon,z= model(real)
        save_image(recon, "images/autoencoder_recon_image/%d.png" % epoch, nrow=2, normalize=True)
torch.save(model.state_dict(), './model/autoencoder2.pth')
