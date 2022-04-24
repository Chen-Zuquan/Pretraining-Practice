import time
import torch
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from network import Bigan
from torch.utils.tensorboard import SummaryWriter

def D_loss(DG, DE, eps=1e-6):
    loss = torch.log(DE + eps) + torch.log(1 - DG + eps)
    return -torch.mean(loss)

def EG_loss(DG, DE, eps=1e-6):
    loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    return -torch.mean(loss)

writer=SummaryWriter("log")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform2 = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# cifar_size: 50000+10000 32*32*3 10classes
data_train = datasets.CIFAR10('./c10data', train=True, transform=transform2, download=True)

#batch
train_batch_size = 128
train_loader = DataLoader(data_train, batch_size=train_batch_size, shuffle=True)

classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

E = Bigan.Encoder().to(device)
G = Bigan.Generator().to(device)
D = Bigan.Discriminator().to(device)

num_epochs = 20
# loss function
optimizer_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()),
                                lr=0.001, betas=(0.5, 0.999), weight_decay=1e-5)
optimizer_D = torch.optim.Adam(D.parameters(),
                               lr=0.01, betas=(0.5, 0.999), weight_decay=1e-5)


train_losses = []
print("start training...")
for epoch in range(num_epochs):

    start_time = time.time()

    D_loss_acc = 0.
    EG_loss_acc = 0.
    D.train()
    E.train()
    G.train()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        # initialize z from 50-dim U[-1,1]
        z = 2 * torch.rand(images.size(0), 256) - 1
        z = z.to(device)

        # compute G(z) and E(X)
        Gz = G(z)
        EX = E(images)

        # compute D(G(z), z) and D(X, E(X))
        DG = D(Gz, z)
        DE = D(images, EX)

        # compute losses
        loss_D = D_loss(DG, DE)
        D_loss_acc += loss_D.item()

        # Discriminator training
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Encoder & Generator training
        Gz = G(z)
        EX = E(images)

        # compute D(G(z), z) and D(X, E(X))
        DG = D(Gz, z)
        DE = D(images, EX)
        loss_EG = EG_loss(DG, DE)
        EG_loss_acc += loss_EG.item()
        optimizer_EG.zero_grad()
        loss_EG.backward()
        optimizer_EG.step()
    print('epoch:{},Discriminator Loss:{:.4f},'.format(epoch, loss_D / len(train_loader), ))
    print('epoch:{},Encoder_generator Loss:{:.4f},'.format(epoch, loss_EG / len(train_loader), ))

    stop_time=time.time()
    print("time is:{:.4f}s".format(stop_time-start_time))

    n_show = 10
    D.eval()
    E.eval()
    G.eval()
    # show image
    with torch.no_grad():
        real = images[:n_show]
        z = 2 * torch.rand(n_show, 256) - 1
        z = z.to(device)
        gener = G(z)
        recon = G(E(real))
        save_image(gener,"images/gene_image/%d.png" % epoch, nrow=2,normalize=True)
        save_image(recon, "images/recon_image/%d.png" % epoch, nrow=2, normalize=True)
        save_image(real, "images/real_image/%d.png" % epoch, nrow=2, normalize=True)
torch.save(E.state_dict(), './model/bigan_encoder.pth')
