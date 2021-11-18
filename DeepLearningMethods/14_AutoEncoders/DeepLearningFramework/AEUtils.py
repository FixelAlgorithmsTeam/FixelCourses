import torch
from torch.utils.data import Dataset

class NNISTDataset(Dataset):
    def __init__(self, oDataset, σ):
        self.oDataset = oDataset
        self.σ        = σ

    def __len__(self):
        return len(self.oDataset)

    def __getitem__(self, idx):
        mX, _ = self.oDataset[idx]
        mXN   = mX + self.σ * torch.randn_like(mX)

        return mXN, mX
