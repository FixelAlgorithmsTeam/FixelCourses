# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plots - Classifier Calibration
# Based on SciKit Learn Calibration tutorial.
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
# | 0.1.000 | 19/02/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# PyTorch

# Image Processing

# Miscellaneous
# from collections import OrderedDict
import os
from platform import python_version, system
import random
# import warnings

# Visualization
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

# %% Configuration

# %matplotlib inline

# warnings.filterwarnings('ignore')

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# sns.set_theme() #>! Apply SeaBorn theme
plt.style.use('dark_background')  # inverts colors to dark theme

figIdx = 0

# %% Constants


# %% Project Packages


# %% Auxiliary Functions

class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        df           = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        
        return proba


# %% Parameters




# %% Loading Data

X, y = make_classification(
    n_samples     = 100_000, 
    n_features    = 20, 
    n_informative = 2, 
    n_redundant   = 2, 
    random_state  = 42
)

train_samples = 1000  # Samples used for training the models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle   = False,
    test_size = 100_000 - train_samples,
)


# %% Display Data



# %% Analysis

oLrCls = LogisticRegressionCV(
    Cs       = np.logspace(-6, 6, 101), 
    cv       = 10, 
    scoring  = "neg_log_loss", 
    max_iter = 1_000
)
oGnbCls       = GaussianNB()
oSvmCls       = NaivelyCalibratedLinearSVC(C = 1.0)
oRndForestCls = RandomForestClassifier(random_state = 42)

lCls = [
    (oLrCls, "Logistic Regression"),
    (oGnbCls, "Naive Bayes"),
    (oSvmCls, "SVC"),
    (oRndForestCls, "Random forest"),
]


# %% Plots
figIdx = 1

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
for i, (clf, name) in enumerate(lCls):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins = 10,
        name   = name,
        ax     = ax_calibration_curve,
        color  = colors(i),
        marker = markers[i],
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration Plots")
ax_calibration_curve.set_xlabel("Mean Predicted Probability (Positive Class)")
ax_calibration_curve.set_ylabel("Fractional of Positives")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(lCls):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range = (0, 1),
        bins  = 10,
        label = name,
        color = colors(i),
    )
    ax.set(title=name, xlabel="Mean Predicted Probability", ylabel="Count")

plt.tight_layout()
plt.show()

fig.savefig(f'Figure{figIdx:04d}.svg', transparent = True)

# %%

oLrCls        = LogisticRegression(C = 1.0)
oSvmCls       = NaivelyCalibratedLinearSVC(max_iter = 10_000)
oSvmClsCalIso = CalibratedClassifierCV(oSvmCls, cv = 2, method = "isotonic")
oSvmClsCalSig = CalibratedClassifierCV(oSvmCls, cv = 2, method = "sigmoid")

lCls = [
    (oLrCls, "Logistic Regression"),
    (oSvmCls, "SVC"),
    (oSvmClsCalIso, "Calibrated SVC (Isotonic)"),
    (oSvmClsCalSig, "Calibrated SVC (Sigmoid)"),
]

figIdx += 1

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(lCls):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins = 10,
        name   = name,
        ax     = ax_calibration_curve,
        color  = colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration Plots (SVC)")
ax_calibration_curve.set_xlabel("Mean Predicted Probability (Positive Class)")
ax_calibration_curve.set_ylabel("Fractional of Positives")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(lCls):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range = (0, 1),
        bins  = 10,
        label = name,
        color = colors(i),
    )
    ax.set(title=name, xlabel = "Mean Predicted Probability", ylabel = "Count")

plt.tight_layout()
plt.show()

fig.savefig(f'Figure{figIdx:04d}.svg', transparent = True)
