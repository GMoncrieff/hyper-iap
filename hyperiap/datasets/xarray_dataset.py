# from xbatcher.loaders.torch import MapDataset
from typing import Any, Callable, Optional, Tuple

import torch

PREDICTOR_VAR = "reflectance"
LABEL_VAR = "label"


class MapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_generator,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        PyTorch Dataset adapter for Xbatcher
        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        transform : callable, optional
            A function/transform that takes in an array and returns a transformed version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        """
        self.X_generator = X_generator
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.__getitem__ currently requires a single integer key"
                )

        # TODO: figure out the dataset -> array workflow
        # currently hardcoding a variable name
        x_batch = self.X_generator[idx][PREDICTOR_VAR].load().torch.to_tensor()
        y_batch = (
            self.X_generator[idx][LABEL_VAR].load().torch.to_tensor().type(torch.int64)
        )

        if self.transform:
            x_batch = self.transform(x_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        return x_batch, y_batch
