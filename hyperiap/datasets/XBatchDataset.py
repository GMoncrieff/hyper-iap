import torch


class XBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        X_generator,
        y_generator,
    ) -> None:
        """
        PyTorch Dataset adapter for Xbatcher
        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        y_generator : xbatcher.BatchGenerator
        """

        self.X_generator = X_generator
        self.y_generator = y_generator

    def __iter__(self):

        for xb, yb in zip(self.X_generator, self.y_generator):
            xb = xb.to_array().squeeze(dim="variable")
            yb = yb.to_array().squeeze(dim="variable")
            # val = (torch.tensor(data=xb.data), torch.tensor(data=yb.data))
            # yield {'x':val[0], 'y':val[1]}
            # val = (torch.tensor(data=xb.data), torch.tensor(data=yb.data))
            yield {"x": torch.tensor(data=xb.data), "y": torch.tensor(data=yb.data)}
