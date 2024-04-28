from typing import Any, List

import torch
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils.gan_utils import get_noise, get_one_hot_labels, combine_vectors

import hydra
import wandb

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class CGAN(LightningModule):

    def __init__(
        self,
        gen: torch.nn.Module,
        disc: torch.nn.Module,
        lr: float,
        z_dim: int = 10,
        n_classes: int = 10,
        device='cpu'
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['gen', 'disc'])

        self.gen = gen
        self.disc = disc

        self.gen_loss = MeanMetric()
        self.disc_loss = MeanMetric()

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        return self.gen(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Flatten the batch of real images from the dataset
        if len(batch) == 2:
            real, labels = batch
        else:
            real = batch 
        one_hot_labels = get_one_hot_labels(labels, self.hparams.n_classes) #[N, C]
        image_one_hot_labels = one_hot_labels[:, :, None, None] #[N, C, 1, 1]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, real.shape[-2], real.shape[-1]) #[N, C, H, W]

        if (optimizer_idx == 0): # get discriminator loss
            noise = get_noise(len(real), self.hparams.z_dim, device=self.device)
            noise_and_labels = combine_vectors(noise, one_hot_labels)

            fake = self.gen(noise_and_labels)

            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)

            fake_pred = self.disc(fake_image_and_labels.detach())
            real_pred = self.disc(real_image_and_labels)

            fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
            real_loss = self.criterion(real_pred, torch.ones_like(real_pred))

            disc_loss = (fake_loss + real_loss) / 2
            self.disc_loss(disc_loss)
            self.log("disc_loss", self.disc_loss, on_step=False, on_epoch=True, prog_bar=True)
            return disc_loss
        else: # get generator loss
            noise = get_noise(len(real), self.hparams.z_dim, device=self.device)
            noise_and_labels = combine_vectors(noise, one_hot_labels)
            
            fake = self.gen(noise_and_labels)

            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)

            # Get the discriminator's prediction of the fake image.
            fake_pred = self.disc(fake_image_and_labels)
            # Calculate the generator's loss.
            gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
            self.gen_loss(gen_loss)
            self.log("gen_loss", self.gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            return gen_loss


    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.disc.parameters(), self.hparams.lr, betas=(0.5, 0.9999))
        opt_g = torch.optim.Adam(self.gen.parameters(), self.hparams.lr, betas=(0.5, 0.9999))

        return [{"optimizer": opt_d}, {"optimizer": opt_g}]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "cgan.yaml")
    _ = hydra.utils.instantiate(cfg)
    print(_.gen)