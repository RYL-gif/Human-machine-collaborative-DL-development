import torch.nn.functional as F
from torch import nn

class CNN(nn.Module):
    #二分类，最后的输出形状为(batch,2)
    def __init__(self):
        super().__init__()
        self.featuremap1=None
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(645248, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        #x1=x.flatten()
        self.featuremap1 = x.detach().cpu()  # 提取这一层的输出
       # x = F.log_softmax(x, dim=1)

        return x
