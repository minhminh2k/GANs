import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.gen = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(hidden_dim, im_chan, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)
    
if __name__ == "__main__":
    z_dim = 100
    gen = Generator(z_dim=z_dim, im_chan=3)
    noise = torch.rand(128, z_dim)
    print(noise.shape)
    print(gen(noise).shape)