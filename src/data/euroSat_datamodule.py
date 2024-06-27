import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Tuple, Optional

import glob
import os
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from pytorch_lightning import LightningDataModule
from torch import Tensor
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import albumentations as A
import torch

from src.data.components.euroSat import EuroSatDataset

class EuroSatDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        img_size: int = 64,
        batch_size: int = 64,
        num_workers: int = 0,
        transform: Tensor = None,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = transform
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        # self.setup()
                
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = EuroSatDataset(
                data_dir=self.hparams.data_dir,
                img_size=self.hparams.img_size,
                transforms=self.hparams.transform,
            )
            data_len = len(dataset)
            train_len = int(data_len * self.hparams.train_val_test_split[0])
            val_len = int(data_len * self.hparams.train_val_test_split[1])
            test_len = data_len - train_len - val_len

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )
            
            print(len(self.data_train), len(self.data_val), len(self.data_test))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "euro.yaml")
    cfg.data_dir = str(root / "data")
    dataset = EuroSatDataset()
    print(dataset.__len__())    

    _ = hydra.utils.instantiate(cfg)
    features, labels = next(iter(_.train_dataloader()))
    print(features.shape)
    print(labels.shape)
    
    