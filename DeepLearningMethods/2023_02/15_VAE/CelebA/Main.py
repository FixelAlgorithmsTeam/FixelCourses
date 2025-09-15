# %%
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

import torch
import torch.optim       as optim

from torch.optim.lr_scheduler import OneCycleLR

# %%
from Dataset  import GetDataLoaders
from Model    import VAE
from Loss     import VAELoss, R2Score
from Training import TrainModel

# %%
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
def Main():
    dirPath           = r'C:\Users\oryai\Workarea\Data\CelebA\img_align_celeba\\'
    batchSize         = 256
    oTrainDL, oTestDL = GetDataLoaders(dirPath, batchSize)

    β          = .4  
    Loss       = VAELoss(β)
    Metric     = R2Score()

    nEpochs    = 50
    nIter      = nEpochs * len(oTrainDL)
    
    oVAE       = VAE       ().to(DEVICE)
    oVAE.load_state_dict(torch.load('./CelebA_VAE.pt'))
    oOptim     = optim.Adam(oVAE.parameters(), betas=(0.9, 0.99))
    oScheduler = OneCycleLR(oOptim, max_lr=1e-3, total_steps=nIter)
    lHistory   = TrainModel(oVAE, oTrainDL, oTestDL, Loss, Metric, nEpochs, oOptim, oScheduler, sModelName='CelebA_VAE')
    # lHistory   = TrainModel(oVAE, oTrainDL, oTrainDL, Loss, Metric, nEpochs, oOptim, oScheduler, sModelName='CelebA_VAE')

    plt.show()

if __name__ == '__main__':
    Main()