import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from models.dcgan.model import DCGAN
from src.models.gan_euro.generator import Generator
from src.models.gan_euro.gan_module import EuroSatGANLitModule

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
    # noise = torch.randn(1, cfg.model.z_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = EuroSatGANLitModule.load_from_checkpoint(
    #     "checkpoints/epoch_199.ckpt",
    #     # map_location=torch.device(device),
    # )
    # gen = model.generator
    # gen.eval()
    '''
    for X, y in subset_loader:
    fig, axes = plt.subplots(5, 3, figsize=(9, 9))

    for i in range(5):
        axes[i, 0].imshow(np.transpose(X.numpy()[i], (1, 2, 0)))
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.transpose(y.numpy()[i], (1, 2, 0)))
        axes[i, 1].set_title("Target")
        axes[i, 1].axis('off')

        generated_image = generator(X[i].unsqueeze(0)).detach().numpy()[0]
        axes[i, 2].imshow(np.transpose(generated_image, (1, 2, 0)))
        axes[i, 2].set_title("Generated")
        axes[i, 2].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.savefig('Test.jpg')
    plt.show()
    '''

    
if __name__ == "__main__":
    inference()