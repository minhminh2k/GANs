import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from models.dcgan.model import DCGAN

from src import utils
import numpy as np

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def inference(cfg: DictConfig):
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    IMG_MEAN = [0.5, 0.5, 0.5]
    IMG_STD = [0.5, 0.5, 0.5]

    def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
    
    gen = model.generator
    gen.eval()
    noise = torch.randn(1, cfg.model.z_dim)
    
    pred = gen(noise)
    
    print(pred.shape) # 1, 1, 28, 28 or 1, 3, 64, 64
    print(pred.max())
    print(pred.min())
    print()
    pred = pred.detach()  # (1, 3, img_size, img_size)
    # pred = torch.sigmoid(pred)
    pred = denormalize(pred)
    print(pred.shape)
    print(pred.max())
    print(pred.min())
    # pred = pred.squeeze(0)
    # pred = pred.permute(1, 2, 0)
    # pred = pred.cpu().numpy() # .astype(np.uint8)
    
    # pred = pred * 255
    # pred = pred.astype(np.uint8)
    
    # print(pred.max())
    # print(pred.min())
    # print(pred.sum())
    # plt.imsave("./abc.jpg", pred)
    # plt.show()
    
if __name__ == "__main__":
    inference()