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


class EuroSatGANLitModule(LightningModule):

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        lr: float,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['generator', 'discriminator'])

        # self.automatic_optimization = False
        self.generator = generator
        self.discriminator = discriminator

        self.gen_loss = MeanMetric()
        self.disc_loss = MeanMetric()

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.l1loss = torch.nn.L1Loss() 
        
        # self.device = "cuda" if torch.cuda.is_available() else "cpu" # Error

    def forward(self, x: torch.Tensor):
        return self.generator(x)
    

    def get_disc_loss(self, real, fake):
        fake_pred = self.discriminator(fake) 
        
        real_pred = self.discriminator(real)

        # Calculate the loss
        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        disc_loss = (fake_loss + real_loss) / 2
        
        return disc_loss

    def get_gen_loss(self, real, fake): 
        # Prediction
        fake_pred = self.discriminator(fake)
        
        # Loss
        gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        
        # Caculate L1 loss, MAE between fake image and targets
        gen_l1_loss = self.l1loss(fake, real) * 100
        
        # Sum losses
        loss_G = gen_loss + gen_l1_loss
        
        return loss_G
    
    def on_train_start(self) -> None:
        # Fixed Noise for visualize
        n_samples = 25

        # log sample real images for visualization
        train_loader = self.trainer.train_dataloader

        # EuroSAT
        x, y = next(iter(train_loader))
        
        self.samples = x[:n_samples]
        grid = vutils.make_grid(self.samples, nrow=int(np.sqrt(n_samples)), normalize=True)
        self.logger.log_image(key='Input images', images=[ToPILImage()(grid)])
        
        self.samples_y = y[:n_samples]
        grid_y = vutils.make_grid(self.samples_y, nrow=int(np.sqrt(n_samples)), normalize=True)
        self.logger.log_image(key='Target images', images=[ToPILImage()(grid_y)])
    
    def on_train_epoch_end(self) -> None:
        n_samples = 25
        
        # log sample real images for visualization
        train_loader = self.trainer.train_dataloader

        # EuroSAT
        x, y = next(iter(train_loader))
        
        samples = self.samples.to('cuda')
        
        fake = self.generator(samples)
        fake = fake.detach()
        grid = vutils.make_grid(fake, nrow=int(np.sqrt(len(fake))), normalize=True)
        self.logger.log_image(key='Output images', images=[ToPILImage()(grid)], step=self.current_epoch)

    def training_step(self, batch, batch_idx, optimizer_idx: int):
        inputs, true_images = batch
        
        fake = self.generator(inputs)

        if optimizer_idx == 0:
            gen_loss = self.get_gen_loss(true_images, fake)
            self.gen_loss(gen_loss)
            self.log('gen_loss', self.gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            return gen_loss
        
        elif optimizer_idx == 1:
            disc_loss = self.get_disc_loss(true_images, fake)
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