
# Python STD
# import enum
import math

# Data
import numpy as np
import pandas as pd
import scipy as sp

# Machine Learning

# Image Processing / Computer Vision

# Optimization

# Auxiliary

# Visualization
import distinctipy
import matplotlib.colors
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

# Miscellaneous
from enum import auto, Enum, unique

# Typing
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

# Course Packages


# See https://docs.python.org/3/library/enum.html
@unique
class DiffMode(Enum):
    # Type of data in the CSV
    BACKWARD    = auto()
    CENTRAL     = auto()
    FORWARD     = auto()
    COMPLEX     = auto()

# Constants
FIG_SIZE_DEF    = (8, 8)
ELM_SIZE_DEF    = 50
CLASS_COLOR     = ('b', 'r')
EDGE_COLOR      = 'k'
MARKER_SIZE_DEF = 10
LINE_WIDTH_DEF  = 2


def Plot2DFun( vX: np.ndarray, vY: np.ndarray, mZ: np.ndarray, /, *, hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF ) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(figsize = figSize, subplot_kw = {'projection': '3d'})
    else:
        hF = hA.get_figure()

    mX, mY = np.meshgrid(vX, vY)
    
    hSurf = hA.plot_surface(mX, mY, mZ, cmap = 'viridis')

    hA.set_title('')
    hA.set_xlabel('x')
    hA.set_ylabel('y')
    hA.set_zlabel('f(x, y)')

    # hF.colorbar(hSurf, shrink = 0.5, aspect = 5)
    hF.colorbar(hSurf, fraction = 0.05)

    return hA
    

