from typing import Any, List

import torch
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

import torchvision.utils as vutils
from torchvision.transforms import ToPILImage

import numpy as np
import hydra

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class SmallDCGANLitModule(LightningModule):

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        lr: float,
        z_dim: int = 64,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['generator', 'discriminator'])

        # self.automatic_optimization = False
        self.generator = generator.apply(weights_init)
        self.discriminator = discriminator.apply(weights_init)

        self.gen_loss = MeanMetric()
        self.disc_loss = MeanMetric()

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        return self.generator(x)
    
    def get_noise(self, n_samples, z_dim, fixed=False):
        if fixed:
            torch.manual_seed(7749)
        return torch.randn(n_samples, z_dim, device=self.device)

    def get_disc_loss(self, real, fake):
        fake_pred = self.discriminator(fake.detach()) 
        # print("Fake pred shape", fake_pred.shape) # 128, 1
        # print("Real shape", real.shape) # 128, 3, 64, 64 / 128, 1, 28, 28
        
        real_pred = self.discriminator(real)

        # Calculate the loss
        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        disc_loss = (fake_loss + real_loss) / 2
        
        return disc_loss

    def get_gen_loss(self, fake): 
        # Prediction
        fake_pred = self.discriminator(fake)
        
        # Loss
        gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        
        return gen_loss
    
    def on_train_start(self) -> None:
        # Fixed Noise for visualize
        n_samples = 25
        self.fixed_noise = self.get_noise(n_samples=n_samples, z_dim=self.hparams.z_dim, fixed=True)

        # log sample real images for visualization
        train_loader = self.trainer.train_dataloader

        # Fashion + Mnist
        x, _ = next(iter(train_loader))
        # print(x.shape) # 128, 1, 28, 28
        # print(type(x)) # torch.tensor
        # print(_.shape) # 128 - label
        # print(type(_)) # torch.tensor
        
        # Gender
        # x = next(iter(train_loader))
        # print(x.shape) # 128, 3, 64, 64
        
        samples = x[:n_samples]
        grid = vutils.make_grid(samples, nrow=int(np.sqrt(n_samples)), normalize=True)
        self.logger.log_image(key='real images', images=[ToPILImage()(grid)])
    
    def on_train_epoch_end(self) -> None:
        fake = self.generator(self.fixed_noise)
        grid = vutils.make_grid(fake, nrow=int(np.sqrt(len(fake))), normalize=True)
        self.logger.log_image(key='fake images', images=[ToPILImage()(grid)], step=self.current_epoch)

    def training_step(self, batch, batch_idx, optimizer_idx: int):
        # 1D: Fashion + Mnist
        real = batch[0]
        
        # 3D: Gender
        # real = batch
        # print("Batch real", real.shape) # 128, 3, 64, 64

        noise = self.get_noise(z_dim=self.hparams.z_dim, n_samples=len(real))
        fake = self.generator(noise)

        if optimizer_idx == 0:
            gen_loss = self.get_gen_loss(fake)
            self.gen_loss(gen_loss)
            self.log('gen_loss', self.gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            return gen_loss
        
        elif optimizer_idx == 1:
            disc_loss = self.get_disc_loss(real, fake)
            self.disc_loss(disc_loss)
            self.log('disc_loss', self.disc_loss, on_step=False, on_epoch=True, prog_bar=True)
            return disc_loss
        
    def test_step(self, *args: Any, **kwargs: Any):
        pass

    def configure_optimizers(self):
        gen_opt = Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        disc_opt = Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        
        return [gen_opt, disc_opt], []


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "small_dcgan.yaml")
    _ = hydra.utils.instantiate(cfg)
    print(_.generator)
    print(_.discriminator)