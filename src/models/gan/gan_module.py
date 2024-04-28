from typing import Any, List

import torch
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from torchvision.transforms import ToPILImage
import torchvision.utils as vutils

import hydra
import numpy as np

class GANLitModule(LightningModule):

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
        
        self.gen = generator
        self.disc = discriminator
        
        self.gen_loss = MeanMetric()
        self.disc_loss = MeanMetric()

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        return self.gen(x) # (len, 784)
    
    def get_noise(self, n_samples, z_dim, fixed=False):
        if fixed:
            torch.manual_seed(7749)
        return torch.randn(n_samples, z_dim, device=self.gen.device) # (samples, z_dim)

    def get_disc_loss(self, real, fake):
        # Discriminator's prediction of the fake image 
        fake_pred = self.disc(fake.detach())
        
        # Discriminator's prediction of the real image and calculate the loss
        real_pred = self.disc(real)

        # Calculate the discriminator's loss 
        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        disc_loss = (fake_loss + real_loss) / 2

        return disc_loss

    def get_gen_loss(self, fake): 
        # Discriminator's prediction of the fake image
        fake_pred = self.disc(fake)
        
        # Calculate the generator's loss
        gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        
        return gen_loss
    
    def on_train_start(self) -> None:
        # cache fixed noise for visualization
        self.fixed_noise = self.get_noise(n_samples=25, z_dim=self.hparams.z_dim, fixed=True)

        # log sample real images for visualization
        train_loader = self.trainer.train_dataloader
        x, _ = next(iter(train_loader))
        samples = x[:25]
        grid = vutils.make_grid(samples, nrow=5, normalize=True)
        self.logger.log_image(key='real images', images=[ToPILImage()(grid)])

    def on_train_epoch_end(self, ) -> None:
        fake_sample_flatten = self.gen(self.fixed_noise) # (25, 28 * 28)
        # Resize
        fake_sample = fake_sample_flatten.view(-1, 1, 28, 28) # [25, 1, 28, 28]
        grid = vutils.make_grid(fake_sample, nrow=5, normalize=True) 
        
        # Logger
        self.logger.log_image(key='fake images', images=[ToPILImage()(grid)], step=self.current_epoch)
        
    def training_step(self, batch, batch_idx, optimizer_idx: int):
        x, y = batch
        
        # Flatten the batch of real images (batch size)
        real = x.view(len(x), -1)
        
        # Generate noise
        noise = self.get_noise(z_dim=self.hparams.z_dim, n_samples=len(x))
        fake = self.gen(noise)

        if optimizer_idx == 0:
            gen_loss = self.get_gen_loss(fake)
            self.gen_loss(gen_loss)
            self.log('gen_loss', gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            return gen_loss
        elif optimizer_idx == 1:
            disc_loss = self.get_disc_loss(real, fake) # real
            self.disc_loss(disc_loss)
            self.log('disc_loss', disc_loss, on_step=False, on_epoch=True, prog_bar=True)
            return disc_loss
        
        # return gen_loss, disc_loss
    
    def test_step(self, batch: Any, batch_idx: int):
        pass

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.gen.parameters(), self.hparams.lr, betas=(0.5, 0.999))
        disc_opt = torch.optim.Adam(self.disc.parameters(), self.hparams.lr, betas=(0.5, 0.999))

        return [gen_opt, disc_opt], []

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import pyrootutils

    root = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(root / "configs/model")
    
    @hydra.main(version_base="1.3", config_path=config_path, config_name="gan.yaml")
    def main(cfg: DictConfig):
        gan = hydra.utils.instantiate(cfg)
        print(gan.device)

    main()