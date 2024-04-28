import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
  
        self.disc = nn.Sequential(

            nn.Conv2d(im_chan, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, image):
        # out = self.disc_layer0(image)
        # out = self.disc_layer1(out)
        # out = self.disc_layer2(out)
        # out = self.disc_layer3(out)
        # out = self.disc_layer4(out)
        return self.disc(image)

if __name__ == "__main__":
    disc = Discriminator(hidden_dim=64)
    noise = torch.rand(128, 1, 28, 28)
    print(noise.shape)
    # tmp = disc.disc_layer0(noise)
    # print(tmp.shape)
    # tmp = disc.disc_layer1(tmp)
    # print(tmp.shape)
    # tmp = disc.disc_layer2(tmp)
    # print(tmp.shape)
    # tmp = disc.disc_layer3(tmp)
    # print(tmp.shape)
    # tmp = disc.disc_layer4(tmp)
    # print(tmp.shape)
    print(disc(noise).shape)
    