import torch
import numpy as np

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
#-- Classification accuracy:
class Accuracy:
    def __init__(self, ignoreIndex=None):
        self.ignoreIndex = ignoreIndex

    def __call__(self, mScore, vY):
        vHatY = mScore.detach().argmax(dim=1)
        if self.ignoreIndex is None:
            return (vHatY == vY).float().mean().item()
        else:
            return (vHatY == vY).float()[vY != self.ignoreIndex].mean().item()

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
from sklearn.metrics import r2_score

def R2Score(vHatY, vY):
    vY    = vY   .detach().cpu().view(-1)
    vHatY = vHatY.detach().cpu().view(-1)
    return r2_score(vY, vHatY)
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def BinaryAcuuracy(vZ, vY):
    vZ = vZ.detach().cpu()
    vY = vY.detach().cpu()
    return torch.mean( ((vZ > 0) == vY).float() )