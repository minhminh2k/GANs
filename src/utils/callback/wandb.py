import torch
import wandb
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import pytorch_lightning as pl
from typing import Any
import cv2
from src.utils.gan_utils import get_noise

class WandbCallback(Callback):
    def __init__(self, z_dim:int = 64, batch_size=128, im_chan:int = 1, img_size:int = 28, use_fixed_noise:bool = True):
        self.size = [im_chan, img_size, img_size]
        # self.size = (1, 28, 28)
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.step = 500
        self.last_batch = None
        self.use_fixed_noise = use_fixed_noise
        self.fake_noise = get_noise(self.batch_size, self.z_dim, device = 'cuda')

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if (not self.use_fixed_noise):
            self.fake_noise = get_noise(self.batch_size, self.z_dim, device = 'cuda')
        # get the image from your model or dataloader
        # with trainer.model.eval():
        fake = trainer.model(self.fake_noise)
        fake = fake.detach()
        
        image_unflat = fake.detach().cpu().view(-1, * self.size)
        image_grid = make_grid(image_unflat[:25], nrow=5)

        real_unflat = self.last_batch.view(-1, * self.size)
        real_grid = make_grid(real_unflat[:25], nrow=5)

        # logging using WandB
        wandb_logger = trainer.logger 
        wandb_logger.log_image(key='Samples', images=[wandb.Image(image_grid), wandb.Image(real_grid)], caption=["fake", "real"])
        

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        pass