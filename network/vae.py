import torch
import torch.nn as nn


device=torch.device("cuda:0")
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            # [(W-F+2P)/S + 1 ] * [(W-F+2P)/S + 1] * M
            # [b,32,32,3]->[b,32,32,16]->[b,16,16,16]
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),  # Batch Normalization加速神经网络的训练
            nn.AvgPool2d(kernel_size=2, stride=2),
            # [b,16,16,16]->[b,16,16,32]->[b,8,8,16]
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
        )

        self.relu = nn.ReLU()
        self.encoder_linear = nn.Linear(1024, 256)
        self.fc_mu = nn.Linear(256, 256)
        self.fc_logvar = nn.Linear(256, 256)
        self.decoder_linear = nn.Linear(256, 1024)
        self.flatten = nn.Flatten()

    def encode(self,x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.relu(self.encoder_linear(x))
        return self.fc_mu(z),self.fc_logvar(z)

    def decode(self,z):
        # h3=self.relu(self.fc3(z))
        deconv_input = self.relu(self.decoder_linear(z))
        deconv_input=deconv_input.view(-1,16,8,8)
        return self.decoder(deconv_input)

    def reparametrize(self,mu,logvar):
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()).to(device) * std + mu
        return z


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        self.encoder = nn.Sequential(
            # [(W-F+2P)/S + 1 ] * [(W-F+2P)/S + 1] * M
            # [b,32,32,3]->[b,32,32,16]->[b,16,16,16]
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),  # Batch Normalization加速神经网络的训练
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [b,16,16,16]->[b,16,16,32]->[b,8,8,16]
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.relu=nn.ReLU()
        self.encoder_linear = nn.Linear(1024, 256)
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
        x = self.relu(self.encoder_linear(x))
        x=self.dense(x)
        return x
if __name__ == '__main__':
    import torch
    img=torch.randn(3,3,32,32).to(device)
    net=CVAE().to(device)
    out,latent_variable,cat=net(img)
    print(out.size(),latent_variable.size())