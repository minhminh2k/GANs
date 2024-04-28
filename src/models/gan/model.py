from typing import Any, List

import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

import hydra

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

class GAN(LightningModule):

    def __init__(
        self,
        gen: torch.nn.Module,
        disc: torch.nn.Module,
        lr: float,
        z_dim: int = 10,
        device='cpu'
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['gen', 'disc'])
        self.automatic_optimization = False
        self.gen = gen
        self.disc = disc

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        return self.gen(x)

    def get_disc_loss(self, real, num_images):
        # Create noise vectors and generate a batch (num_images) of fake images. 
        noise = get_noise(num_images, self.hparams.z_dim, device=self.device)
        fake = self.gen(noise)

        # Get the discriminator's prediction of the fake image 
        # and calculate the loss.
        fake_pred = self.disc(fake.detach())
        
        # Get the discriminator's prediction of the real image and calculate the loss.
        real_pred = self.disc(real)

        # Calculate the discriminator's loss by averaging the real 
        # and fake loss and set it to disc_loss.
        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        disc_loss = (fake_loss + real_loss) / 2

        return disc_loss

    def get_gen_loss(self, num_images): 
        # Create noise vectors and generate a batch of fake images. 
        noise = get_noise(num_images, self.hparams.z_dim, device=self.device)
        fake = self.gen(noise)

        # Get the discriminator's prediction of the fake image.
        fake_pred = self.disc(fake)
        # Calculate the generator's loss.
        gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        return gen_loss

    def training_step(self, batch, batch_idx):
        # Flatten the batch of real images from the dataset
        opt_g, opt_d = self.optimizers()
        real = batch[0].view(len(batch[0]), -1)

        gen_loss = self.get_gen_loss(len(real))
        self.log("gen_loss", gen_loss, on_step=False, on_epoch=True, prog_bar=True)

        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        opt_g.step()

        disc_loss = self.get_disc_loss(real, len(real))
        self.log("disc_loss", disc_loss, on_step=False, on_epoch=True, prog_bar=True)

        opt_d.zero_grad()
        self.manual_backward(disc_loss)
        opt_d.step()
        
        # return gen_loss, disc_loss

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters(), self.hparams.lr, betas=(0.5, 0.9999))
        opt_d = torch.optim.Adam(self.disc.parameters(), self.hparams.lr, betas=(0.5, 0.9999))

        return [{"optimizer": opt_g}, {"optimizer": opt_d}]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gan.yaml")
    _ = hydra.utils.instantiate(cfg)