
# Python STD
from enum import auto, Enum, unique
import math

# Data
import numpy as np
# import pandas as pd
import scipy as sp

# Machine Learning

# Deep Learning
import torch
import torch.nn            as nn
import torch.nn.functional as F

# Image Processing / Computer Vision

# Optimization

# Auxiliary

# Visualization
import matplotlib.pyplot as plt

# Miscellaneous

# Course Packages
from DeepLearningBlocks import NNMode


# Typing
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union


# Auxiliary Functions

def ResetModelWeights( oModel: nn.Module ) -> None:
    # https://discuss.pytorch.org/t/19180

    for layer in oModel.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif hasattr(layer, 'children'):
            for child in layer.children():
                ResetModelWeights(child)