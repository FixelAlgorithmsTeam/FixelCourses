
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

def DownloadDecompressGzip( fileUrl: str, fileName: str) -> None:
    # Based on https://stackoverflow.com/a/61195974

    # Read the file inside the .gz archive located at url
    with urllib.request.urlopen(fileUrl) as response:
        with gzip.GzipFile(fileobj = response) as uncompressed:
            file_content = uncompressed.read()
        # write to file in binary mode 'wb'
        with open(fileName, 'wb') as f:
            f.write(file_content)
            f.close()
        return

def DownloadUrl( fileUrl: str, fileName: str ) -> str:
    
    if not os.path.exists(fileName):
        urllib.request.urlretrieve(fileUrl, fileName)

    return fileName

def ConvertMnistDataDf( imgFilePath: str, labelFilePath: str ) -> Tuple[np.ndarray, np.ndarray]:
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