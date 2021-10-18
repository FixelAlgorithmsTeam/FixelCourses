# %%
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

import torch

# %%
def Plot(vTrainLoss, vValLoss, vTrainMetric, vValMetric, vLR, epoch, fig=None):
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    else:
        axes      = fig.axes
        for ax in axes:
            ax.cla()

    axes[0].set_title('Loss')
    axes[0].plot     (vTrainLoss,   label='Train')
    axes[0].plot     (vValLoss,     label='Test' )

    axes[1].set_title('Metric')
    axes[1].plot     (vTrainMetric, label='Train') 
    axes[1].plot     (vValMetric,   label='Test' ) 

    axes[2].set_title('Learning rate')
    axes[2].plot     (vLR,          label='Learnign rate')
    for ax in axes:
        ax.legend()
        ax.grid  ()

    fig.canvas.draw()
    plt.pause(1e-2)

    return fig