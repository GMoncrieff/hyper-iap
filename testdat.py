from hyperiap.datasets.timeseries_module import TimeSeriesDataModule
from hyperiap.datasets.timeseries import TimeseriesDataset
from torchvision import transforms

def cli_main():
    dataset = TimeseriesDataset(
        train=True
    )
    bat, y = next(iter(dataset))
    print(bat)
    
if __name__ == "__main__":
    cli_main()