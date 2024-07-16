
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import default_collate
from torchvision.datasets.vision import VisionDataset  
from torchvision.transforms.functional import pil_to_tensor, to_tensor

import skimage as ski
from PIL import Image

# Miscellaneous
import os
import pathlib

from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

def ExtFileName( fullFileName: str ) -> str:
    # Extracts the file name without the extension

    fileName, fileExt = os.path.splitext(fullFileName)

    return fileName

class ImageSegmentationDataset(VisionDataset):
    def __init__(self, datasetPath: str, imgTransform: Callable = None, maskTransform: Callable = None, maskDType = np.int64, lImgFormats: List = ['jpg', 'jpeg', 'png']):
        super().__init__(datasetPath, transforms = None, transform = imgTransform, target_transform = maskTransform)
        
        self.maskDType = maskDType
        self.imgFolderPath = os.path.join(self.root, 'Images')
        self.annFolderPath  = os.path.join(self.root, 'Annotations')

        lAnnFileName = [ExtFileName(itmName) for itmName in os.listdir(self.annFolderPath)]
        lImgFileName = [ExtFileName(itmName) for itmName in os.listdir(self.imgFolderPath)]

        sMatchFileName =  set(lAnnFileName) & set(lImgFileName)

        _lAnn = [] #<! Annotations
        _lImg = [] #<! Images

        lFiles = os.listdir(self.imgFolderPath)
        lFiles.sort() #<! Make order deterministic

        for fileItm in lFiles:
            fileName, fileExt = os.path.splitext(fileItm)
            if ((fileExt[1:] in lImgFormats) and (fileName in sMatchFileName)):
                _lImg.append(os.path.join(self.imgFolderPath, fileItm))
                _lAnn.append(os.path.join(self.annFolderPath, fileName + '.png'))

        self._lAnn = _lAnn
        self._lImg = _lImg

    def __len__(self) -> int:
        return len(self._lImg)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        # On PyTorch 2.3.1 (Checked only on Windows) `imread()` caused issues
        # mI = ski.io.imread(self._lImg[idx])
        # mM = ski.io.imread(self._lAnn[idx])

        mI = np.array(Image.open(self._lImg[idx]).convert("RGB")) #<! Guarantees 3 channels
        mM = np.array(Image.open(self._lAnn[idx]).convert("L")) #<! Guarantees 1 channels
        # mM = mM - 1
        
        # mI = mI.astype(np.float32) / 255.0
        # mM = mM.astype(np.int64)

        # if (np.ndim(mI) < 3):
        #     # Grayscale Image
        #     print('2D')
        #     mI = mI[:, :, None]

        # if (np.size(mI, 2) == 1):
        #     # Grayscale Image
        #     print('Gray')
        #     mI = np.tile(mI, (1, 1, 3))
        
        # if (np.size(mI, 2) == 4):
        #     # RGBA Image
        #     print('!!!!!RGBA!!!!!')
        #     mI = mI[:, :, :3]
        
        if self.transform is not None:
            mI = self.transform(mI)
        
        if self.target_transform is not None:
            mM = self.target_transform(mM[:, :, None]) #<! Transforms expect HxWxC
        
        # print(f'Image Index: {idx}')
        # print(f'Image shape: {mI.shape}')
        # print(f'Target shape: {mM.shape}')

        return mI, mM


def GenTrainTesIdx(numSamples: int, trainSize: float = 0.8, seedNum: int = 123):
    
    vAllIdx         = np.arange(numSamples)
    numTainsSamples = int(trainSize * numSamples)
    
    rng = np.random.default_rng(seedNum) #<! Stable Random Number Generator
    
    vTrainIdx   = rng.choice(numSamples, numTainsSamples, replace = False) 
    vTestIdx    = np.setdiff1d(vAllIdx, vTrainIdx)

    vTrainIdx   = np.sort(vTrainIdx)
    vTestIdx    = np.sort(vTestIdx)

    return vTrainIdx, vTestIdx

class AdjustMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        
        return torch.squeeze(x) - 1


class ToTensor( nn.Module ):
    def __init__( self ) -> None:
        super(ToTensor, self).__init__()
            
    def forward( self: Self, *args ) -> Tuple[torch.Tensor]:
        """
        Converts input to Tensor.  
        """
		
        return tuple(torch.tensor(itm) for itm in args)


def GenDataLoaders( dsTrain: Dataset, dsVal: Dataset, batchSize: int, *, numWorkers: int = 0, CollateFn: Callable = default_collate, dropLast: bool = True, PersWork: bool = False ) -> Tuple[DataLoader, DataLoader]:

    if numWorkers == 0: 
        PersWork = False

    dlTrain = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = 1 * batchSize, num_workers = numWorkers, collate_fn = CollateFn, drop_last = dropLast, persistent_workers = PersWork)
    dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = numWorkers, persistent_workers = PersWork)

    return dlTrain, dlVal

