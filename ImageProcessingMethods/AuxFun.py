
# Python STD
# import enum
import math

# Data
import numpy as np
import pandas as pd
import scipy as sp

from numba import njit

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

def ImageGradient(mI: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the image gradient vector.
    Input:
      - mI  : Numpy 2D array.
    Output:
      - mIx : Horizontal image gradient.
      - mIy : Vertical image gradient.
    Remarks:
      - Matches MATLAB's `gradient()`.
    """
    
    mIx = np.empty_like(mI)
    mIy = np.empty_like(mI)

    # Horizontal
    mIx[:, 1:-1] = (mI[:, 2:] - mI[:, :-2]) / 2
    mIx[:, 0]    = mI[:, 1] - mI[:, 0]
    mIx[:, -1]   = mI[:, -1] - mI[:, -2]

    # Vertical
    mIy[1:-1, :] = (mI[2:, :] - mI[:-2, :]) / 2
    mIy[0, :]    = mI[1, :] - mI[0, :]
    mIy[-1, :]   = mI[-1, :] - mI[-2, :]
    
    return mIx, mIy

# @njit
def _BuilBinaryMrfGraphWeights( mL0: np.ndarray, mL1: np.ndarray, tC: np.ndarray ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the sparse graph weights matrix for the binary MRF energy function using (vI, vJ, vV) format.
    
    Parameters:
    ----------
    L0 : np.ndarray
        2D array of unary costs for assigning 0 to each pixel (shape: N x M).
        
    L1 : np.ndarray
        2D array of unary costs for assigning 1 to each pixel (shape: N x M).
        
    C : np.ndarray
        4D array representing the pairwise costs between pixels.
        Shape: (N, M, dN, dM), where (dN, dM) defines the neighborhood structure.
        For instance, if you are considering 4-neighborhood:
        - dN = [-1, 0, 1, 0]
        - dM = [0, 1, 0, -1]
        
    Returns:
    -------
    vI : list
        List of row indices for non-zero entries in the sparse matrix.
    
    vJ : list
        List of column indices for non-zero entries in the sparse matrix.
    
    vV : list
        List of values for the non-zero entries in the sparse matrix.
    Remarks:
     - The matrix is not required to be symmetric.
     - The graph has 2 directed terminals:
        - Source to pixels.
        - Pixels to sink.
    """
    
    numRows, numCols  = mL0.shape  #<! Image dimensions
    numPixels         = numRows * numCols
    numNeighbors      = tC.shape[2]
    estNumNnz         = 2 * numPixels + numPixels * numNeighbors
    
    # Initialize lists to store row indices (vI), column indices (vJ), and values (vV)
    vI = np.empty(estNumNnz)
    vJ = np.empty(estNumNnz)
    vV = np.empty(estNumNnz)
    
    # Source and sink indices (the last two nodes in the graph)
    srcIdx = numPixels     #<! Source terminal
    snkIdx = numPixels + 1 #<! Sink terminal

    elmIdx = -1 #<! The non zero element index
    pxIdx  = -1 #<! The pixel index

    # Add pairwise weights (N Links)
    # Going from one above, clockwise
    if (numNeighbors == 4):
        vShiftI = [-1, 0, 1, 0]  #<! Vertical neighbors (up, down)
        vShiftJ = [0, 1, 0, -1]  #<! Horizontal neighbors (left, right)
    elif (numNeighbors == 8):
        vShiftI = [-1, -1, -1,  0,  1,  1,  1,  0]  #<! Vertical neighbors (up, down)
        vShiftJ = [ 0,  1,  1,  1,  0, -1, -1, -1]   #<! Horizontal neighbors (left, right)
    else:
        raise ValueError(f'The number of neighbors in `tC` does not match either 4 or 8')
    
    # Add unary weights (T Links)
    for ii in range(numRows):
        for jj in range(numCols):
            pxIdx  += 1
            elmIdx += 1
            
            # Source link (Assign 1)
            vI[elmIdx] = srcIdx  #<! Source to pixel (i,j)
            vJ[elmIdx] = pxIdx
            vV[elmIdx] = (mL1[ii, jj])

            elmIdx += 1
            
            # Sink link (Assign 0)
            vI[elmIdx] = pxIdx  #<! Pixel (i,j) to sink
            vJ[elmIdx] = snkIdx
            vV[elmIdx] = (mL1[ii, jj])
    
    

    pxIdx  = -1 #<! The pixel index
    
    for ii in range(numRows):
        for jj in range(numCols):
            pxIdx += 1
            elmIdx += 1
            
            # Check neighbors in 4-connected neighborhood
            for kk in range(numNeighbors):
                ni = ii + vShiftI[kk]
                nj = jj + vShiftJ[kk]
                val = tC[ii, jj, kk]
                
                if 0 <= ni < numRows and 0 <= nj < numCols:
                    nIdx = ni * numCols + nj #<! Neighbor index
                    
                    # Add the pairwise cost for this edge
                    vI[elmIdx] = pxIdx  # Pixel (i,j) to its neighbor
                    vJ[elmIdx] = nIdx
                    vV[elmIdx] = val
    
    numNnz = elmIdx + 1
    
    return vI[:numNnz], vJ[:numNnz], vV[:numNnz]

def BuilBinaryMrfGraphWeights( mL0: np.ndarray, mL1: np.ndarray, tC: np.ndarray ) -> sp.sparse.csr_matrix:
    """
    Build the sparse graph weights matrix for the binary MRF energy function using (vI, vJ, vV) format.
    
    Parameters:
    ----------
    L0 : np.ndarray
        2D array of unary costs for assigning 0 to each pixel (shape: N x M).
        
    L1 : np.ndarray
        2D array of unary costs for assigning 1 to each pixel (shape: N x M).
        
    C : np.ndarray
        4D array representing the pairwise costs between pixels.
        Shape: (N, M, dN, dM), where (dN, dM) defines the neighborhood structure.
        For instance, if you are considering 4-neighborhood:
        - dN = [-1, 0, 1, 0]
        - dM = [0, 1, 0, -1]
        
    Returns:
    -------
    vI : list
        List of row indices for non-zero entries in the sparse matrix.
    
    vJ : list
        List of column indices for non-zero entries in the sparse matrix.
    
    vV : list
        List of values for the non-zero entries in the sparse matrix.
    Remarks:
     - The matrix is not required to be symmetric.
     - The graph has 2 directed terminals:
        - Source to pixels.
        - Pixels to sink.
    """

    numRows, numCols  = mL0.shape  #<! Image dimensions
    numPixels         = numRows * numCols
    
    vI, vJ, vV = _BuilBinaryMrfGraphWeights(mL0, mL1, tC)
    mW = sp.sparse.csr_matrix((vV, (vI, vJ)), shape=(numPixels + 2, numPixels + 2))

    return mW
