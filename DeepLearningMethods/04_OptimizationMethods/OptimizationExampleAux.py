import numpy             as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class LinearLayer:
    def __init__(self, dIn, dOut):
        mW = np.random.randn(dOut, dIn) * np.sqrt(2 / dIn)
        vB = np.zeros(dOut)

        self.dParams = {'mW' : mW, 'vB': vB}
        self.dGrads  = {}

    def Forward(self, mX):
        mW      = self.dParams['mW']
        vB      = self.dParams['vB']
        self.mX = mX                   #-- store for Backward
        mZ      = mW @ mX + vB[:,None]

        return mZ

    def Backward(self, mDz):
        mW  = self.dParams['mW']

        vDb = mDz.sum(axis=1)
        mDw = mDz @ self.mX.T
        mDx = mW.T @ mDz

        self.dGrads['vB'] = vDb
        self.dGrads['mW'] = mDw

        return mDx

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class ReLULayer:
    def __init__(self):
        self.dGrads = {}

    def Forward(self, mX):
        self.mX = mX                 #-- store for Backward
        mZ      = np.maximum(mX, 0)

        return mZ

    def Backward(self, mDz):
        mX    = self.mX
        mMask = (mX > 0).astype(float)

        mDx   = mDz * mMask

        return mDx

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class SequentialModel:
    def __init__(self, lLayers):
        self.lLayers = lLayers

    def Forward(self, mX):
        for oLayer in self.lLayers:
            mX = oLayer.Forward(mX)
        return mX

    def Backward(self, mDz):
        for oLayer in reversed(self.lLayers):
            mDz = oLayer.Backward(mDz)


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def CrossEntropyLoss(vY, mZ):
    '''
    Returns both the loss and the gradient w.r.t the input (mZ)
    '''
    mHatY  = np.exp(mZ)
    mHatY /= np.sum(mHatY, axis=0)
    N      = len(vY)
    loss   = -np.log(mHatY[vY,range(N)]).mean()

    mDz               = mHatY
    mDz[vY,range(N)] -= 1
    mDz              /= N

    return loss, mDz

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def Accuracy(mHatY, vY):
    vHatY    = np.argmax(mHatY, axis=0)
    accuracy = (vHatY == vY).astype(float).mean()
    return accuracy


class Dataset:
    def __init__(self, mX, vY, batchSize):
        self.mX             = mX
        self.vY             = vY
        self.N              = len(vY)
        self.batchSize      = min(batchSize, self.N)
        self.numMiniBatches = self.N // self.batchSize
            
    def __len__(self):
        return self.numMiniBatches
    
    #-- Return mini-batches:
    def __iter__(self):
        vIdx = np.random.permutation(self.N)
    
        for ii in range(self.numMiniBatches):
            startIdx  = ii * self.batchSize
            vBatchIdx = vIdx[startIdx : startIdx + self.batchSize]
            mBatchX   = self.mX[:,vBatchIdx]
            vBatchY   = self.vY[vBatchIdx]

            yield mBatchX, vBatchY

class Plot:
    def __init__(self, sTitle, sLabel, sXlabel, sColor, vData=[]):
        self.sTitle  = sTitle
        self.sLabel  = sLabel
        self.sXlabel = sXlabel
        self.sColor  = sColor
        self.vData   = vData

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Recorder:
    def __init__(self, lPlots, figsize=(12,4)):
        self.lTitles = np.unique([oPlot.sTitle for oPlot in lPlots])
        self.N       = len(self.lTitles)
        self.fig, _  = plt.subplots(1, self.N, figsize=(12, 4))
        self.dAxes   = {}
        ii           = 0
        for oPlot in lPlots:
            ax = self.dAxes.get(oPlot.sTitle, None)
            if ax == None:
                ax                       = self.fig.axes[ii]
                ii                      += 1
                self.dAxes[oPlot.sTitle] = ax

            ax.set_title(oPlot.sTitle)
            ax.set_xlabel(oPlot.sXlabel)
            ax.plot(oPlot.vData, c=oPlot.sColor, label=oPlot.sLabel)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

    def Append(self, sTitle, sLabel, vData):
        ax = self.dAxes[sTitle]
        for oLine in ax.lines:
            if oLine.get_label() == sLabel:
                vYdata = np.append(oLine.get_ydata(), vData)
                N      = len(vYdata)
                oLine.set_data(list(range(N)), vYdata)
        lYlim = ax.axis()[2:4]
        if N > 1:
            ax.axis(xmin=0, xmax=N, ymin=np.minimum(np.min(vData), lYlim[0]), ymax=np.maximum(np.max(vData), lYlim[1]))
        else:
            ax.axis(xmin=0, xmax=N, ymin=np.min(vData), ymax=np.max(vData)+1e-10)
            
    def Get(self, sTitle, sLabel):
        ax = self.dAxes[sTitle]
        for oLine in ax.lines:
            if oLine.get_label() == sLabel:
                return oLine.get_ydata()

    def Draw(self):
        self.fig.canvas.draw()
        plt.pause(1e-10)