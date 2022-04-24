import torch.nn as nn
import torch.nn.functional as F
class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
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
        self.linear1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        feature = self.relu(self.linear1(x))
        projection = self.projection(feature)
        return F.normalize(projection),F.normalize(feature)

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
    img=torch.randn(3,3,32,32)
    net=SimCLR()
    out,latent_variable=net(img)
    print(out.size(),latent_variable.size())