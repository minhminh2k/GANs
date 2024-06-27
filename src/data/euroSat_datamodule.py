from typing import Tuple
import glob
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch import Tensor
import cv2
from PIL import Image
import opendatasets as od
import os

class GenderDataset(Dataset):

    dataset_dir = 'gender'
    dataset_url = 'https://www.kaggle.com/datasets/yasserhessein/gender-dataset'

    def __init__(
        self, 
        root: str = "data/", 
        transforms: Tensor = None
    ):
        
        super().__init__()

        self.dataset_dir = osp.join(root, "gender-dataset/Dataset")
        img_dir = [
            f"{self.dataset_dir}/Test/Female/*.jpg",
            f"{self.dataset_dir}/Test/Male/*.jpg",
            f"{self.dataset_dir}/Validation/Female/*.jpg",
            f"{self.dataset_dir}/Validation/Male/*.jpg",
            f"{self.dataset_dir}/Train/Female/*.jpg",
            f"{self.dataset_dir}/Train/Male/*.jpg"
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(
            img_dir[1]) + glob.glob(img_dir[2]) + glob.glob(
                img_dir[3]) + glob.glob(img_dir[4]) + glob.glob(img_dir[5])
        
        self.transforms = transforms
        
        self.prepare_data()

    def prepare_data(self) -> None:
        if os.path.exists(self.dataset_dir):
            print("Data is downdloaded")
            return
        od.download(self.dataset_url)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)

        return image * 2.0 - 1.0

class GenderDataModule(LightningDataModule):

    dataset_dir = 'gender'
    dataset_url = 'https://www.kaggle.com/datasets/yasserhessein/gender-dataset'

    def __init__(
        self,
        data_dir: str = "data/",
        img_dims: int = 3,
        img_size: int = 64,
        batch_size: int = 128,
        num_workers: int = 0,
        transform: Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = transform

    def train_dataloader(self):
        return DataLoader(
            dataset=GenderDataset(transforms=self.transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True
        )

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "gender.yaml")
    cfg.data_dir = str(root / "data")
    dataset = GenderDataset()
    print(dataset.__len__())
    # print(dataset.__getitem__(0))
    

    _ = hydra.utils.instantiate(cfg)
    features = next(iter(_.train_dataloader()))
    print(features.shape)
    # print(len(labels))