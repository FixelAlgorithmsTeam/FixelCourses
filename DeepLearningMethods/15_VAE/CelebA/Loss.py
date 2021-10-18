# %%
import numpy    as np

import torch
import torch.nn as nn

# %%
from sklearn.metrics import r2_score

class R2Score():
    def __call__(self, vHatY, vY):
        with torch.no_grad():
            vY    = vY   .detach().cpu().view(-1)
            vHatY = vHatY.detach().cpu().view(-1)
        return r2_score(vY, vHatY)

# %%
class VAELoss():
    def __init__(self, β):
        self.β = β
    
    def __call__(self, mHatX, mμ, mLogVar, mX):
        MSE = nn.MSELoss(reduction='sum')(mHatX, mX)
        KLD  = 0.5 * torch.sum(mμ**2 + mLogVar.exp() - 1 - mLogVar)
        N    = mX.shape[0]
    
        return (MSE + self.β * KLD) / N

