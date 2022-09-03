import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
#from torchvision.datasets.mnist import MNIST
from torchvision import transforms

#from hyperiap.models.mlp import MLP
from hyperiap.models.cnn1d import simpleCNN, TempCNN
from hyperiap.models.baseclassifier import BaseClassifier
import numpy as np

def cli_main():
    meh = np.random.random(500)
    meh = np.expand_dims(meh, axis=1)
    model = BaseClassifier(simpleCNN()) 
    
    model2 = TempCNN()
    
    print(model2)
    print(model)
    
if __name__ == "__main__":
    cli_main()
