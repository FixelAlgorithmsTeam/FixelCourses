import torch.nn as nn

#-- Tensor to vector (reshape):
class Squeeze(nn.Module):
    def forward(self, mX):
        return mX[:,0]

#-- Tensor to vector (reshape):
class Reshape(nn.Module):
    def __init__(self, vShape):
        super(Reshape, self).__init__()
        self.vShape = vShape

    def forward(self, mX):
        return mX.view(mX.shape[0], *self.vShape)