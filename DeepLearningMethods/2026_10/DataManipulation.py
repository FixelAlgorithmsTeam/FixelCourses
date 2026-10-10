
# Python STD
from enum import auto, Enum, unique
# import math
import shutil

# Data
import numpy as np
# import pandas as pd
# import scipy as sp

# Machine Learning

# Image Processing / Computer Vision
import skimage as ski

# Optimization

# Auxiliary

# Visualization

# Miscellaneous
import gdown
import gzip
import os
import urllib.request

# Typing
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from numpy.typing import NDArray

# See https://docs.python.org/3/library/enum.html
@unique
class DiffMode(Enum):
    # Type of data in the CSV
    BACKWARD    = auto()
    CENTRAL     = auto()
    FORWARD     = auto()
    COMPLEX     = auto()

# Constants
L_ARCHIVE_EXT = ['.zip', '.tar.bz2', '.bz2', '.tbz2', '.tar.gz', '.gz', '.tgz', '.tar', '.tar.xz', '.xz', '.txz']

def DownloadGDriveZip( fileId: str, lFileCont: List[str] ) -> None:

    for fileName in lFileCont:
        if os.path.isfile(fileName):
            os.remove(fileName)
    
    fileNameExt = gdown.download(id = fileId)
    fileName, fileExt = os.path.splitext(fileNameExt)
    if fileExt in L_ARCHIVE_EXT:
        # Might not work with `tar` files (Might require unpacking twice)
        shutil.unpack_archive(fileNameExt)
        os.remove(fileNameExt)

def DownloadDecompressGzip( fileUrl: str, fileName: str ) -> None:
    # Based on https://stackoverflow.com/a/61195974

    # Read the file inside the .gz archive located at url
    with urllib.request.urlopen(fileUrl) as urlResponse:
        with gzip.GzipFile(fileobj = urlResponse) as uncompressedData:
            fileContent = uncompressedData.read()
        # write to file in binary mode 'wb'
        with open(fileName, 'wb') as hFile:
            hFile.write(fileContent)    
    
    return

def ParseEnvFile( fileName: str = '.env', *, filePath: str = '.', keyValSep: str = '=' ) -> Dict[str, str]:
    # Read the file and parse it into a dictionary
    dEnv = {}
    with open(os.path.join(filePath, fileName), 'r') as hF:
        for line in hF:
            if line.startswith('#') or not line.strip():
                continue
            key, value = line.strip().split(keyValSep, 1)
            dEnv[key]  = value.strip()
    return dEnv

def DownloadProgress(blockNum, blockSize, totalSize):
    # https://stackoverflow.com/a/74314365

    bytesDownloaded   = blockNum * blockSize
    relativeProgress  = blockNum * blockSize / totalSize
    bytesDownloadedKb = bytesDownloaded // 1024
    totalSizeKb       = totalSize // 1024

    print(f'Downloaded: {relativeProgress:0.2%} of the file ({bytesDownloadedKb} [Kilo Byte] / {totalSizeKb} [Kilo Byte])', end = '\r')

def DownloadUrl( fileUrl: str, fileName: str ) -> str:
    # See improvement in https://stackoverflow.com/a/63832993
    # The `fileName` can be a full path.
    
    if not os.path.exists(fileName):
        urllib.request.urlretrieve(fileUrl, fileName, DownloadProgress)

    print(f'\nDownloaded file: {fileName}')

    return fileName

def DownloadKaggleDataset( userName: str, datasetName: str, fileName: str ) -> None:
    # Downloads the Kaggle Dataset using `curl` like command
    # The `userName` and `datasetName` are in the form 'userName/datasetName': 
    # `https://www.kaggle.com/datasets/girish17019/mobile-phone-defect-segmentation-dataset` -> `girish17019`, `mobile-phone-defect-segmentation-dataset`

    # Converts: `curl -L -o <fileName> https://www.kaggle.com/api/v1/datasets/download/<userName>/<datasetName>` into Python

    kaggleUrl = f'https://www.kaggle.com/api/v1/datasets/download/{userName}/{datasetName}'

    DownloadUrl(kaggleUrl, fileName)

    return

def ConvertMnistDataDf( imgFilePath: str, labelFilePath: str ) -> Tuple[NDArray, NDArray]:
    numPx = 28 * 28
    # Merge of https://pjreddie.com/projects/mnist-in-csv/ and https://github.com/keras-team/keras/blob/master/keras/datasets/fashion_mnist.py
    f = open(imgFilePath, "rb")
    l = open(labelFilePath, "rb")

    lCol = [f'Px {ii:04}' for ii in range (numPx)]
    lCol.append('Label')

    vY = np.frombuffer(l.read(), np.uint8, offset = 8)
    mX = np.frombuffer(f.read(), np.uint8, offset = 16)
    # mX = np.reshape(mX, (numPx, len(vY))).T
    mX = np.reshape(mX, (len(vY), numPx))

    f.close()
    l.close()

    return mX, vY

