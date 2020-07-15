import numpy             as np
import matplotlib.pyplot as plt

import time


from DeepLearningFramework.Layers        import LinearLayer, ReLULayer
from DeepLearningFramework.Models        import SequentialModel
from DeepLearningFramework.LossFunctions import CrossEntropyLoss
from DeepLearningFramework.Dataset       import Dataset
from DeepLearningFramework.Auxiliary     import Accuracy, Plot, Recorder
from DeepLearningFramework.Optimizer     import Optimizer, Adam

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