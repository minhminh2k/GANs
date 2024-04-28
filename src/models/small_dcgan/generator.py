from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, z_dim=64, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        def block(input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
            layers = []
            if not final_layer:
                layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride))
                layers.append(nn.BatchNorm2d(output_channels)) # only add batch norm to hidden layers
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride))
                layers.append(nn.Tanh())
            return layers

        self.generator = nn.Sequential(
            *block(z_dim, hidden_dim * 4),
            *block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            *block(hidden_dim * 2, hidden_dim),
            *block(hidden_dim, im_chan, kernel_size=4, final_layer=True)
        )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.generator(x)

if __name__ == "__main__":
    z_dim = 64
    gen = Generator(z_dim=z_dim)
    noise = torch.randn(128, z_dim, 1, 1)
    print(gen(noise).shape) # [128, 1, 28, 28]


# class Generator(nn.Module):
#     def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
#         super(Generator, self).__init__()
#         self.z_dim = z_dim
#         # Build the neural network
#         self.gen_layer0 = self.make_gen_block(z_dim, hidden_dim * 4)
#         self.gen_layer1 = self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1)
#         self.gen_layer2 = self.make_gen_block(hidden_dim * 2, hidden_dim)
#         self.gen_layer3 = self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True)

#     def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
#         # Build the neural block
#         if not final_layer:
#             return nn.Sequential(
#                 nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
#                 nn.BatchNorm2d(output_channels),
#                 nn.ReLU()
#             )
#         else: # Final Layer
#             return nn.Sequential(
#                 nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
#                 nn.Tanh()                
#             )

#     def unsqueeze_noise(self, noise):
#         return noise.view(len(noise), self.z_dim, 1, 1)

#     def forward(self, noise):
#         x = self.unsqueeze_noise(noise)
#         out = self.gen_layer0(x)
#         out = self.gen_layer1(out)
#         out = self.gen_layer2(out)
#         out = self.gen_layer3(out)
#         return out

# if __name__ == "__main__":
#     z_dim = 64
#     gen = Generator(z_dim=z_dim)
#     noise = torch.randn(128, z_dim, 1, 1)
#     print(noise.shape)
#     tmp = gen.gen_layer0(noise)
#     print(tmp.shape)
#     tmp = gen.gen_layer1(tmp)
#     print(tmp.shape)
#     tmp = gen.gen_layer2(tmp)
#     print(tmp.shape)
#     tmp = gen.gen_layer3(tmp)
#     print(tmp.shape)