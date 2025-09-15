import torch
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def Line2Tensor(oVocab, line):
    lLine = ['<SOS>'] + line.split() + ['<EOS>']
    return torch.tensor(oVocab(lLine))
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
class LangDataset(torch.utils.data.Dataset):
    def __init__(self, lSrc, lTrg, oSrcVocab, oTrgVocab):
        self.lSrc      = lSrc
        self.lTrg      = lTrg
        self.oSrcVocab = oSrcVocab
        self.oTrgVocab = oTrgVocab
        
    def __len__(self):
        return len(self.lSrc)
    
    def __getitem__(self, idx):
        sSrc = self.lSrc[idx]
        sTrg = self.lTrg[idx]
        
        vSrc = Line2Tensor(self.oSrcVocab, sSrc)
        vTrg = Line2Tensor(self.oTrgVocab, sTrg)

        return vSrc, vTrg
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
from torch.nn.utils.rnn import pad_sequence

def LangCollate(lBatch):
    lSrc, lTrg = zip(*lBatch)
    mTrg       = pad_sequence(lTrg, padding_value=3, batch_first=True)
    
    return (lSrc, mTrg)
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#