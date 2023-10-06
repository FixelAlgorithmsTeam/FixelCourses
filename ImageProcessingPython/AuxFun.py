
# Python STD
# import enum
import math

# Data
import numpy as np
import pandas as pd
import scipy as sp

# Models

# Image Processing / Computer Vision
import skimage as ski

# Visualization
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

# Miscellaneous
from enum import auto, Enum, unique
# Typing
from typing import Dict, List, Tuple

# See https://docs.python.org/3/library/enum.html
@unique
class DataType(Enum):
    # Type of data in the CSV
    TEST_DATA       = auto()
    TRAIN_DATA      = auto()
    VALIDATION_DATA = auto()

def MatBlockView(mI: np.ndarray, tuBlockShape: Tuple[int, int]) -> np.ndarray:
    """
    Generates a view of block of shape `blockShape` of the input 2D NumPy array.
    Input:
      - mI           : Numpy 2D array.
      - tuBlockShape : A tuple of the block shape.
    Output:
      - tBlockView   : Tensor of blocks on its 3rd axis.
    Remarks:
      - It assumed the shape of the input array `mI` is an integer multiplication
        of the block size.
      - No verification of compatibility of shapes is done.
    """
    # Pay attention to integer division
    # Tuple addition means concatenation of the Tuples
    tuShape   = (mI.shape[0] // tuBlockShape[0], mI.shape[1] // tuBlockShape[1]) + tuBlockShape
    tuStrides = (tuBlockShape[0] * mI.strides[0], tuBlockShape[1] * mI.strides[1]) + mI.strides
    
    return np.lib.stride_tricks.as_strided(mI, shape = tuShape, strides = tuStrides)