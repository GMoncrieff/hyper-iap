
import numpy as np
from torch.utils.data import Dataset
import torch


class TimeseriesDataset(Dataset):
    """Example time series dataset."""

    def __init__(self, train=True, transform=None):
        """
        Args:
            train (bool): Train or test data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.train = train

        if self.train:
            self.filename = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv"
        else:
            self.filename = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv"

        # in future, just download data here
        self.x, self.y = self.get_data()

        assert len(self.x) == len(self.y), "feature and label length differ"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        # in future load item from disk here
        # memory efficient method
        features, target = self.x[index], self.y[index]

        if self.transform is not None:
            features = self.transform(features)
            
        return features, target

    def get_data(self):
        data = np.loadtxt(self.filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        #x = x.reshape((x.shape[0], x.shape[1], 1))
        #rep 27 times to simulate added dim
        x = np.repeat(x[:, :, np.newaxis], 27, axis=2)
        #x = x.transpose(0,2,1)
        y = y.astype(int)
        y[y == -1] = 0

        return torch.tensor(x).float(), torch.tensor(y)
