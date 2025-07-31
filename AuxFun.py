# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning

# Miscellaneous
import os
from platform import python_version
import random
import re
import shutil

# Visualization
import matplotlib.pyplot as plt

# Set of Function to Manage the Project

def ShiftIndexedFilenames(inDir: str, outDir: str, startIndex: int, valShift: int) -> None:
    """
    Copies or renames files with a 4-digit numeric prefix, adding `valShift`
    to the prefix starting at `startIndex`.

    Parameters
    ----------
    inDir : Directory Path
        Directory containing files with names like '0001SomeName.txt'.
    outDir : Directory Path
        Target directory to write files to (can be the same as `inDir` for in place).
    startIndex : int
        1-based index: from this file onward (sorted), the shift is applied.
    valShift : int
        Value to add to the numeric prefix of selected files.

    Returns
    -------
    None
    """
    
    os.makedirs(outDir, exist_ok = True)

    rePattern = re.compile(r"^(\d{4})(.*)$")
    lFiles = sorted([
        f for f in os.listdir(inDir)
        if os.path.isfile(os.path.join(inDir, f)) and rePattern.match(f)
    ])

    # Reverse order for in place renaming to avoid collisions
    lFiles = list(reversed(lFiles))

    for ii, fileName in enumerate(lFiles, start = 1):
        match = rePattern.match(fileName)
        if not match:
            continue

        fileIdxStr, fileNameSuffix = match.groups()
        fileIdx = int(fileIdxStr)

        # Shift only from `startIndex` and onward
        if fileIdx >= startIndex:
            newIndex = fileIdx + valShift
        else:
            newIndex = fileIdx

        fileNameNew = f"{newIndex:04d}{fileNameSuffix}"

        srcPath = os.path.join(inDir, fileName)
        dstPath = os.path.join(outDir, fileNameNew)

        if srcPath != dstPath:
            shutil.copy2(srcPath, dstPath)
