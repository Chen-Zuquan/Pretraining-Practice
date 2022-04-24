import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
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

        self.relu = nn.ReLU()
        self.encoder_linear = nn.Linear(1024, 256)
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            # 线性分类器
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

    # 前向计算
    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.relu(self.encoder_linear(x))
        return self.dense(x)

