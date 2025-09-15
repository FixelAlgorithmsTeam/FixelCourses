import torch
import torch.nn as nn

#-- https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothingLoss(nn.Module):
    def __init__(self, nClass, ε=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.nClass = nClass
        self.ε      = ε

    def forward(self, mScore, vY):
        ε      = self.ε
        nClass = self.nClass
        mScore = mScore.log_softmax(dim=-1)

        with torch.no_grad():
            mSmoothY         = torch.zeros_like(mScore)
            mSmoothY.fill_   (ε / (nClass - 1))
            mSmoothY.scatter_(1, vY.data.unsqueeze(1), 1-ε)

        return torch.mean( torch.sum(-mSmoothY * mScore, dim=-1) )