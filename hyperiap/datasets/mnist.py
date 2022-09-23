import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from typing import Optional


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        dataset = MNIST(
            self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.mnist_test = MNIST(
            self.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
