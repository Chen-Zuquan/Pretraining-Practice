import time
import torch
from torchvision.utils import save_image
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from network import vae
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("log")

def loss_function(recon_x, x, mu, logvar):
    a=0.0001
    MSE_loss = nn.MSELoss()
    reconstruction_loss = MSE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)*a
    return reconstruction_loss + KL_divergence,reconstruction_loss,KL_divergence

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

model=vae.CVAE().to(device)
num_epochs = 40
# loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
print("start training...")
for epoch in range(num_epochs):

    start_time = time.time()

    train_loss = 0
    all_kl_loss=0
    all_reconstruction_loss=0
    model.train()

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs = add_noise(inputs, 0.1)
        gen_imgs, mu, logvar = model(inputs)
        loss, reconstruction_loss, KL_loss = loss_function(gen_imgs, inputs,mu, logvar)

        optimizer.zero_grad()   # 模型参数梯度清零
        loss.backward()    # 误差反向传递
        optimizer.step()    # 更新参数

        train_loss += loss
        all_kl_loss+=KL_loss
        all_reconstruction_loss+=reconstruction_loss
    print('epoch:{},Train Loss:{:.4f},reconstruction_Loss:{:.4f},KL_loss:{:.4f}'.format(epoch, train_loss / len(train_loader)
                                                                                  ,all_kl_loss / len(train_loader)
                                                                                  , all_reconstruction_loss / len(train_loader)      ))

    writer.add_scalar('Train/Loss', train_loss/len(train_loader), epoch)
    stop_time=time.time()
    print("time is:{:.4f}s".format(stop_time-start_time))

    model.eval()
    with torch.no_grad():
        real=inputs[:10]
        gen_imgs, mu, logvar= model(real)
        save_image(gen_imgs, "images/cvae_recon_image/%d.png" % epoch, nrow=5, normalize=True)
torch.save(model.state_dict(), './model/cvae.pth')
