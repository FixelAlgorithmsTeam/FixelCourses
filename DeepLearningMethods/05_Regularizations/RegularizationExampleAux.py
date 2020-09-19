import numpy             as np
import matplotlib.pyplot as plt
import time

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

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
#-- Basic gradient descent update rule:
class Sgd:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def Step(self, mW, mDw, dState={}):
        mW -= self.lr * mDw
        return mW, dState


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Sgdm:
    def __init__(self, lr=1e-3, beta=0.9):
        self.lr   = lr
        self.beta = beta

    def Step(self, mW, mDw, dState={}):
        mV            = dState.get('mV', np.zeros(mW.shape))
        mV            = self.beta * mV - self.lr * mDw
        mW           += mV
        dState['mV']  = mV

        return mW, dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps

    def Step(self, mW, mDw, dState={}):
        mV            = dState.get('mV', np.zeros(mW.shape))
        mS            = dState.get('mS', np.zeros(mW.shape))
        ii            = dState.get('ii', 0) + 1

        mV            = self.beta1 * mV + (1 - self.beta1) * mDw
        mS            = self.beta2 * mS + (1 - self.beta2) * mDw * mDw

        mTildeV       = mV / (1 - self.beta1**ii)
        mTildeS       = mS / (1 - self.beta2**ii)

        mW           -= self.lr * mTildeV / (np.sqrt(mTildeS) + self.eps)
        dState['mV']  = mV
        dState['mS']  = mS
        dState['ii']  = ii

        return mW, dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Optimizer:
    def __init__(self, oUpdateRule):
        self.oUpdateRule = oUpdateRule #-- SGD, SGDM, ADAM
        self.dStates     = {}

    def Step(self, oModel, lr=None):

        if lr is not None:
            self.oUpdateRule.lr = lr

        for ii, oLayer in enumerate(oModel.lLayers):
            for sParam in oLayer.dGrads:
                #-- Get param, gradient and history:
                mW          = oLayer.dParams[sParam]
                mDw         = oLayer.dGrads[sParam]
                sLayerParam = str(ii) + sParam
                dState      = self.dStates.get(sLayerParam, {})

                #-- Apply step:
                mW, dState                = self.oUpdateRule.Step(mW, mDw, dState)

                #-- Set param and hisoty:
                oLayer.dParams[sParam]    = mW
                self.dStates[sLayerParam] = dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainLoop(oModel, oTrainData, oOptim):

    epochLoss = 0
    epochAcc  = 0
    #-- Iterate over the mini-batches:
    for ii, (mX, vY) in enumerate(oTrainData):
        #-- Forward:
        mZ        = oModel.Forward(mX)
        loss, mDz = CrossEntropyLoss(vY, mZ)

        #-- Backward:
        oModel.Backward(mDz)

        #-- Update parameters
        oOptim.Step(oModel)

        epochLoss += loss
        epochAcc  += Accuracy(mZ, vY)
        print(f'Iteration: {ii:3d}: loss = {loss:.6f}\r', end='')

    epochLoss /= ii + 1
    epochAcc  /= ii + 1

    return epochLoss, epochAcc

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def ValidationLoop(oModel, oValData):

    epochLoss = 0
    epochAcc  = 0
    #-- Iterate over the mini-batches:
    for ii, (mX, vY) in enumerate(oValData):
        #-- Forward:
        mZ      = oModel.Forward(mX)
        loss, _ = CrossEntropyLoss(vY, mZ)

        epochLoss += loss
        epochAcc  += Accuracy(mZ, vY)

    epochLoss /= ii + 1
    epochAcc  /= ii + 1

    return epochLoss, epochAcc

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainModel(oModel, oTrainData, oValData, numEpochs, oOptim):

    oRecorder = Recorder([
        Plot('Loss',       'train', 'epoch', 'b'),
        Plot('Loss',       'val',   'epoch', 'r'),
        Plot('Accuracy',   'train', 'epoch', 'b'),
        Plot('Accuracy',   'val',   'epoch', 'r'),
    ])

    for epoch in range(numEpochs):
        startTime           = time.time()
        #-- Train:
        trainLoss, trainAcc = TrainLoop(oModel, oTrainData, oOptim)
        #-- Validation:
        valLoss,   valAcc   = ValidationLoop(oModel, oValData)
        endTime             = time.time()

        #-- Display:
        oRecorder.Append('Loss',     'train', trainLoss),
        oRecorder.Append('Loss',     'val',   valLoss),
        oRecorder.Append('Accuracy', 'train', trainAcc),
        oRecorder.Append('Accuracy', 'val',   valAcc),
        oRecorder.Draw()

        print('Epoch '              f'{epoch:3d}:',     end='')
        print(' | Train loss: '     f'{trainLoss:.5f}', end='')
        print(' | Val loss: '       f'{valLoss:.5f}',   end='')
        print(' | Train Accuracy: ' f'{trainAcc:2.4f}', end='')
        print(' | Val Accuracy: '   f'{valAcc:2.4f}',   end='')
        print(' | epoch time: '     f'{(endTime-startTime):3.3f} |')


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def LearningRateSweep(oModel, oTrainData, oOptim, vLearningRate):

    #-- Record mini-batches loss:
    oRecorder = Recorder([
        Plot('Batch loss', 'train', 'learning rate', 'b'),
    ])

    numIter = len(vLearningRate)
    ii      = 0
    while ii < numIter:
        #-- Iterate over the mini-batches:
        for mX, vY in oTrainData:
            if ii >= numIter:
                break

            #-- Forward:
            mZ        = oModel.Forward(mX)
            loss, mDz = CrossEntropyLoss(vY, mZ)

            #-- Backward:
            oModel.Backward(mDz)

            #-- Update parameters (with new learning rate)
            oOptim.Step(oModel, vLearningRate[ii])

            oRecorder.Append('Batch loss', 'train', loss)
            oRecorder.Draw()

            ii += 1

    #-- Display:
    ax = oRecorder.dAxes['Batch loss']
    ax.lines[0].set_xdata(vLearningRate)
    ax.axis(xmin=vLearningRate[0], xmax=vLearningRate[-1])
    ax.set_xscale('log')
    oRecorder.Draw()