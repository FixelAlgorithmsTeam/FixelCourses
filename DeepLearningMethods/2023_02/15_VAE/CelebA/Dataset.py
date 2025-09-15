# %%
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from torch.utils.data     import DataLoader, random_split
from torchvision.datasets import ImageFolder

# %%
def SetRange(mX, bForward=True):
    if bForward == True:
        return 2 * mX - 1   #-- [0, 1] --> [-1, 1]
    else:
        return (mX + 1) / 2 #-- [0, 1] --> [-1, 1]
# %%                                
# dirPath = r'C:\Users\oryai\Workarea\Data\CelebA\img_align_celeba\\'
def GetData(dirPath):
    oTransform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop          (148),
                                    transforms.Resize              (64),
                                    transforms.ToTensor            (),
                                    transforms.Lambda              (SetRange)
                                    ])

    oDataset = ImageFolder(dirPath, transform=oTransform)
    N        = len(oDataset)
    nTrain   = int(0.9 * N)
    # nTrain   = 8
    nTest    = N - nTrain

    oTrainSet, oTestSet = random_split(oDataset, [nTrain, nTest], generator=torch.Generator().manual_seed(42))
    
    return oTrainSet, oTestSet
# %%
def GetDataLoaders(dirPath, batchSize):
    oTrainSet, oTestSet = GetData(dirPath)

    oTrainDL = DataLoader(oTrainSet, shuffle=True,  batch_size=1*batchSize, num_workers=2, persistent_workers=True)
    oTestDL  = DataLoader(oTestSet,  shuffle=False, batch_size=2*batchSize, num_workers=2, persistent_workers=True)

    return oTrainDL, oTestDL