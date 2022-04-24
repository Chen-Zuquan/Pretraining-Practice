import numpy as np
from PIL import Image
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from network import Siamese
from tqdm import tqdm


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
temperature=0.5

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_set = CIFAR10Pair(root="c10data",
                            train=True,
                            transform=train_transform,
                            download=True)
batchsize=1000

train_loader = DataLoader(train_set,
                          batch_size=batchsize,
                          shuffle=True,)

model = Siamese.SimCLR().to(device)

optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.00001
)

# train
model.train()
SimCLR_loss=0
epochs=10
for epoch in range(epochs):
    start_time = time.time()
    for x1,x2,label in train_loader:
        x1=x1.to(device)
        x2=x2.to(device)
        projection1,feature1=model(x1)
        projection2,feature2=model(x2)

        # [2*B, D]
        out = torch.cat([projection1, projection2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batchsize, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batchsize, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(projection1 * projection2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        SimCLR_loss+=loss
    stop_time = time.time()
    print("time is:{:.4f}s".format(stop_time - start_time))
    print('epoch:{},Siamese Loss:{:.4f},'.format(epoch, SimCLR_loss / len(train_loader), ))

torch.save(model.state_dict(), './model/SimCLR.pth')