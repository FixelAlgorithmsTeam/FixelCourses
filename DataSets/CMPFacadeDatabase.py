# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # CMP Facade Database
# Arranges a dataset for training based on the CMP Facade Database.
# The dataset is available at: [CMP Facade Database](https://cmp.felk.cvut.cz/~tylecr1/facade).
# The script assumes both the Base and Extended versions of the dataset are downloaded and unzipped in the `DataSets\CMPFacadeDatabase` folder.
# It renames the images into format of `00001.jpg`, `00002.jpg`, ..., 0001.png, 0002.png, ...
# It deletes the `xml` files.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 04/09/2026 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
# import scipy as sp
# import pandas as pd

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning

# Miscellaneous
import os
from platform import python_version
import random
import requests
import shutil

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

def DownloadFileUrl( fileUrl: str, outFile: str ) -> None:
    
    r = requests.get(fileUrl, allow_redirects = True)
    open(outFile, 'wb').write(r.content)


# %% Parameters

datasetName     = 'CMPFacadeDatabase'


# %% Download


# %% Delete the XML Files



# %% Rename the Files



# %%
