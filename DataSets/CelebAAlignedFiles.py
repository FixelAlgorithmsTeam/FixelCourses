# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # CelebA Aligned Dataset
# Arranges a dataset for training based on the CelebA Aligned dataset.
# The dataset is available at: [CelebA Aligned Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 09/08/2026 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning

# Miscellaneous
import os
import random
import shutil
import time
import zipfile

# Visualization
# import matplotlib.pyplot as plt


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

PROJECT_NAME     = 'FixelCourses'
DATA_FOLDER_NAME = 'DataSets'
BASE_FOLDER_PATH = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)


# %% Local Packages


# %% Auxiliary Functions

# %% Parameters

datasetName       = 'CelebAAligned'
zipFileName       = 'img_align_celeba.zip'
annoFileName      = 'identity_CelebA.txt'
outputDatasetName = 'CelebAAlignedFiles'
classFileName     = 'Classes.txt'
progressInterval  = 1_000


# %% Create Image Folder

datasetFolderPath = os.path.join(DATA_FOLDER_PATH, datasetName)
zipFilePath       = os.path.join(datasetFolderPath, zipFileName)
annoFilePath      = os.path.join(datasetFolderPath, annoFileName)
outputFolderPath  = os.path.join(datasetFolderPath, outputDatasetName)
classFilePath     = os.path.join(outputFolderPath, classFileName)
tempClassFilePath = f'{classFilePath}.tmp'

if os.path.isdir(outputFolderPath):
    shutil.rmtree(outputFolderPath)
os.makedirs(outputFolderPath, exist_ok = True)

with open(annoFilePath, 'r', encoding = 'utf-8') as annoFile:
    totalNumSamples = sum(1 for line in annoFile if line.strip())

numSamples = 0
startTime = time.perf_counter()
with (
    open(annoFilePath, 'r', encoding = 'utf-8') as annoFile,
    zipfile.ZipFile(zipFilePath, 'r') as zipFile,
    open(tempClassFilePath, 'w', encoding = 'utf-8', newline = '\n') as classFile,
):
    for lineIdx, line in enumerate(annoFile, start = 1):
        lineParts = line.split()
        if len(lineParts) != 2:
            raise ValueError(f'Invalid annotation at line {lineIdx}: {line.rstrip()}')

        fileName, identity = lineParts
        expectedFileName = f'{lineIdx:06d}.jpg'
        if fileName != expectedFileName:
            raise ValueError(
                f'Expected {expectedFileName} at line {lineIdx}, found {fileName}'
            )

        label = int(identity) - 1
        imageData = zipFile.read(f'img_align_celeba/{fileName}')

        with open(os.path.join(outputFolderPath, fileName), 'wb') as imageFile:
            imageFile.write(imageData)
        classFile.write(f'{label}\n')
        numSamples += 1

        if (numSamples % progressInterval == 0) or (numSamples == totalNumSamples):
            elapsedTime = time.perf_counter() - startTime
            samplesPerSecond = numSamples / elapsedTime
            remainingTime = (totalNumSamples - numSamples) / samplesPerSecond
            progress = 100 * numSamples / totalNumSamples
            print(
                f'Progress: {numSamples:,}/{totalNumSamples:,} ({progress:5.1f}%) | '
                f'{samplesPerSecond:,.1f} samples/s | ETA: {remainingTime / 60:.1f} min',
                flush = True,
            )

os.replace(tempClassFilePath, classFilePath)

print(f'Created {numSamples} images and {classFilePath}')

# %%
