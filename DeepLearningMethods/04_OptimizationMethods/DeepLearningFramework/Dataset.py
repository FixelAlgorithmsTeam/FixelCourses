import numpy             as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Dataset:
    def __init__(self, mX, vY, batchSize):
        self.mX             = mX
        self.vY             = vY
        self.N              = len(vY)
        self.batchSize      = min(batchSize, self.N)
        self.numMiniBatches = self.N // self.batchSize

    def __len__(self):
        return self.numMiniBatches

    def __iter__(self):
        self.vIdx = np.random.permutation(self.N)
        self.ii   = 0

        return self

    def __next__(self):
        if self.ii < self.numMiniBatches:
            startIdx  = self.ii * self.batchSize
            vBatchIdx = self.vIdx[startIdx : startIdx + self.batchSize]
            mBatchX   = self.mX[:,vBatchIdx]
            vBatchY   = self.vY[vBatchIdx]
            self.ii  += 1

            return mBatchX, vBatchY
        else:
            raise StopIteration
