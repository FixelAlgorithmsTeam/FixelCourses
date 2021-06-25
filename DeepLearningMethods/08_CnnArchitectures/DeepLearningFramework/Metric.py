import torch
import numpy as np

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
#-- Classification accuracy:
def Accuracy(mScore, vY):
    vHatY = mScore.detach().argmax(dim=1)
    return (vHatY == vY).float().mean().item()

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
from sklearn.metrics import r2_score

def R2Score(vHatY, vY):
    vY    = vY   .detach().cpu().view(-1)
    vHatY = vHatY.detach().cpu().view(-1)
    return r2_score(vY, vHatY)
