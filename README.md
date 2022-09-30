
---

<div align="center">    
 
# Detection of invasive plants using hyperspectral imagery    
[![build](https://github.com/GMoncrieff/hyper-iap/actions/workflows/ci-testing.yml/badge.svg)](https://github.com/GMoncrieff/hyper-iap/actions/workflows/ci-testing.yml)
[![codecov](https://codecov.io/gh/GMoncrieff/hyper-iap/branch/main/graph/badge.svg?token=AJQEB1CXPZ)](https://codecov.io/gh/GMoncrieff/hyper-iap)
</div>
 
## Description   
This package is intented for running deep learning classifiers on hyperspectral data for mapping invasive alien plants. Models are pretrained on noisy land cover label data and then fine-tuned using point labels. The package is designed to be suffiently general that any geospatial classification problem with point labels or raster can be handled.

**Currently this package is under heavy development**

### Development roadmap
Current and planned data sources, models and module features are indicated below

#### Data sources
- [x] Timeseries
- [ ] Sentinel 2
- [ ] Hyperspectral
#### Models
- [x] [TempCNN](https://www.mdpi.com/2072-4292/11/5/523)
- [x] [ViT](https://arxiv.org/abs/2010.11929)
- [x] [SpectralFormer](https://ieeexplore.ieee.org/document/9627165)

#### Module features

- [x] Logging with W&B
- [ ] Hyperparameter tuning with W&B sweeps

## Getting started
First, install dependencies   
```bash
# clone project   
git clone https://github.com/GMoncrieff/hyper-iap

# install project   
cd hyper-iap  
pip install -r requirements.txt
 ```   
 Next, train classifiers using the command line.   
 ```bash
python run_classifier.py --model_class=tempcnn.TEMPCNN --data_class=timeseries_module.TimeSeriesDataModule    
```
For a full list of command line options run
 ```bash
python run_classifier.py --help
```

## Imports
You can also import individual modules and incorporate them into python workflows
```python
from pytorch_lightning import Trainer, seed_everything
from hyperiap.models.tempcnn import TEMPCNN

from hyperiap.models.baseclassifier import BaseClassifier
from hyperiap.datasets.timeseries_module import TimeSeriesDataModule

#setup data module
ts = TimeSeriesDataModule()
#setup model
model = BaseClassifier(TEMPCNN(data_config=ts.config()))
#setup trainer
trainer = Trainer(max_epochs=10)
#Train!
trainer.fit(model, datamodule=ts)
```

## Acknowledgements

The module builds on contributions and implementations from :

* [TempCNN](https://github.com/charlotte-pel/igarss2019-dl4sits) - [Pelletier et al., 2019](https://www.mdpi.com/2072-4292/11/5/523)
* [BreizhCrops](https://github.com/dl4sits/BreizhCrops) - [Rußwurm et al., 2019](https://arxiv.org/abs/1905.11893)
* [ViT](https://github.com/google-research/vision_transformer) - [ Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)
* [SpectralFormer](https://github.com/charlotte-pel/igarss2019-dl4sits) - [Hong et al., 2021](https://ieeexplore.ieee.org/document/9627165)
* [Full Stack Deep Learning](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022)
* [xbatcher](https://github.com/xarray-contrib/xbatcher)
* [zen3geo](https://github.com/weiji14/zen3geo)
  

The alien plant label data originates from : 
* [Holden et al., 2021](https://www.sciencedirect.com/science/article/abs/pii/S2352938520306236)  

With the land cover labels used from pre-training from
* [The South African Department of Forestry, Fisheries and the Environment ](https://egis.environment.gov.za/sa_national_land_cover_datasets)