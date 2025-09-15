import numpy as np

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Dataset:
    def __init__(self, mX, vY, batchSize, bDropLast=False):
        self.mX        = mX
        self.vY        = vY
        self.N         = len(vY)
        self.batchSize = min(batchSize, self.N)
        self.nBatches  = np.ceil(self.N / self.batchSize).astype(np.int32)

    def __len__(self):
        return self.nBatches

    #-- Loop over mini-batches:
    def __iter__(self):
        vIdx = np.random.permutation(self.N)

        for ii in range(self.nBatches):
            startIdx  = ii * self.batchSize
            vBatchIdx = vIdx[startIdx : startIdx + self.batchSize]
            mBatchX   = self.mX[:,vBatchIdx]
            vBatchY   = self.vY[vBatchIdx]

            yield mBatchX, vBatchY
