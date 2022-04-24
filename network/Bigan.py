import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(16),  # Batch Normalization加速神经网络的训练
            nn.AvgPool2d(kernel_size=2, stride=2),
            # [b,16,16,16]->[b,16,16,32]->[b,8,8,16]
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.linear1=nn.Linear(1024,256)
        self.relu=nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    def forward(self, X):
        x=self.encoder(X)
        x=self.flatten(x)
        return self.relu(self.linear1(x))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(3),
        )
        self.linear1 = nn.Linear(256, 1024)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    def forward(self, z):
        x = self.sigmoid(self.linear1(z))
        x = x.reshape(-1, 16, 8, 8)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(32*32*3 + 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    def forward(self, X, z):
        X=X.reshape(-1,3072)
        Xz = torch.cat([X, z], dim=1)
        return self.layers(Xz)

class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),  # Batch Normalization加速神经网络的训练
            nn.AvgPool2d(kernel_size=2, stride=2),
            # [b,16,16,16]->[b,16,16,32]->[b,8,8,16]
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.linear1 = nn.Linear(1024, 256)
        self.relu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            # 线性分类器
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.dense(x)
        return x
if __name__ == '__main__':
    import torch
    x1=torch.randn(3,3,32,32)
    z2=torch.randn(3,256)
    encoder=Encoder()
    discriminator=Discriminator()
    generator=Generator()

    z1=encoder(x1)
    x2=generator(z2)
    print(z1.size())
    out1=discriminator(x1,z1)
    out2=discriminator(x2,z2)

    print(out1.size(),out2.size())