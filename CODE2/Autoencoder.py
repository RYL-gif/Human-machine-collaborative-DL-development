from torch import nn


class AutoEncoder_CNN(nn.Module):
    def __init__(self):
        super(AutoEncoder_CNN, self).__init__()
        self.featuremap1 = None
        self.featuremap2 = None
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, 3, stride=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 1, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 1, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
            #print("first",x.shape)
            encoded = self.encoder(x)
            x=encoded
            #print("encode", x.shape)
            #x = x.view(x.size(0), -1)
            #x = F.log_softmax(x, dim=1)
            #x = torch.flatten(x)
            #print(x.shape)
            en=x.detach().cpu().numpy()
            #print(en.shape)
            decoded = self.decoder(encoded)
            x = decoded
            #print("decode", x.shape)
            #x = x.view(x.size(0), -1)
            de=x.detach().cpu().numpy()

            return encoded, decoded,en,de

