# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plot Figures - Plot Clustering Score
# Plot simple cases clustering to score the number of clusters with different methods.
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# Remarks:
# - A
# 
# To Do & Ideas:
# 1. B
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 0.1.000 | 27/02/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# Image Processing

# Miscellaneous
import os
from platform import python_version, system
import random
# import warnings

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Typing
from typing import Callable, List, Tuple, Union

# %% Configuration

# %matplotlib inline

# warnings.filterwarnings('ignore')

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme
# sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
# sns.color_palette('tab10')
sns.set_theme(style = "ticks", context = "talk")
plt.style.use("dark_background")  # inverts colors to dark theme

figIdx = 0

# %% Constants


# %% Courses Packages


# %% Auxiliary Functions


# %% Parameters

# Data
numSamples  = 1_000
numClusters = 5
dataDim     = 2
lClusterStd = [ii / 3 for ii in range(1, numClusters + 1)]

# Model
lNumClusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Visualization
numGridPts = 500


# %% Loading / Generating Data

mX, vY = make_blobs(
    n_samples    = numSamples, 
    centers      = numClusters, 
    n_features   = dataDim, 
    random_state = 15
)

figIdx += 1

hF, hA = plt.subplots(figsize = (10, 6))
sns.scatterplot(x = mX[:, 0], y = mX[:, 1], hue = vY,
                palette = 'tab10', ax = hA)
hA.set_title('Cluster Data')
hA.set_xlabel('$x_1$')
hA.set_ylabel('$x_2$');

hF.savefig(f'Figure{figIdx:04d}.svg', transparent = True)


# %% Analyze Data

# dClusterModels = {
#     'Agglomerative': AgglomerativeClustering, 
#     'KMeans'       : KMeans, 
#     'Spectral'     : SpectralClustering
# }

# dClusterMetricsAdjusted = {
#     'CompletenessScore' : completeness_score,
#     'FowlkesMallows'    : fowlkes_mallows_score,
#     'Homogeneity'       : homogeneity_score
#     'MutualInfo'        : adjusted_mutual_info_score,
#     'RandScore'         : adjusted_rand_score,
# }

# dClusterMetrics = {
#     'CalinskiHarabasz' : calinski_harabasz_score,
#     'DaviesBouldin'    : davies_bouldin_score,
#     'Silhouette'       : silhouette_score
# }

# for numClusters in lNumClusters:
#     for modelName, modelClass in dClusterModels.items():
#         oClusModel = modelClass(n_clusters = numClusters)
#         vYHat      = oClusModel.fit_predict(mX)


dClusterMetricsAdjusted = {
    'MutualInfo'       : adjusted_mutual_info_score,
    'RandScore'        : adjusted_rand_score,
}

dClusterMetrics = {
    'CalinskiHarabasz' : calinski_harabasz_score,
    'DaviesBouldin'    : davies_bouldin_score,
    'Silhouette'       : silhouette_score
}

dfScores = pd.DataFrame(index = lNumClusters, columns = list(dClusterMetrics.keys()) + list(dClusterMetricsAdjusted.keys()))

for numClusters in lNumClusters:
    oKMeans = KMeans(n_clusters = numClusters)
    vYHat   = oKMeans.fit_predict(mX)
    
    for metricName, metricFunc in dClusterMetricsAdjusted.items():
        dfScores.loc[numClusters, metricName] = metricFunc(vY, vYHat)
    for metricName, metricFunc in dClusterMetrics.items():
        dfScores.loc[numClusters, metricName] = metricFunc(mX, vYHat)

# Scale the `'CalinskiHarabasz'` score into [0, 1] range
dfScores['CalinskiHarabasz'] = MinMaxScaler().fit_transform(dfScores[['CalinskiHarabasz']])


# %% Plot Results

figIdx += 1

hF, hA = plt.subplots(figsize = (10, 6))
sns.lineplot(data = dfScores, 
             ax = hA)
hA.set_title('KMeans - Clustering Score')
hA.set_xlabel('Number of Clusters')
hA.set_ylabel('Score');

hF.savefig(f'Figure{figIdx:04d}.svg', transparent = True)


# %%
