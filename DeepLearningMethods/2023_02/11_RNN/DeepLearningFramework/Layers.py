import torch.nn as nn

#-- Tensor to vector (reshape):
class Squeeze(nn.Module):
    def forward(self, mX):
        return mX[:,0]