import numpy as np
import time

from DeepLearningFramework.Metric import CrossEntropyLoss, Accuracy

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def Epoch(oModel, oData, oOptim=None, bTrain=True):

    epochLoss = 0
    epochAcc  = 0
    count     = 0
    #-- Iterate over the mini-batches:
    for ii, (mX, vY) in enumerate(oData):

        Nb     = len(vY)
        count += Nb

        #-- Forward:
        mZ        = oModel.Forward(mX)
        loss, mDz = CrossEntropyLoss(vY, mZ)

        if bTrain == True:
            oModel.Backward(mDz) #-- backward
            oOptim.Step(oModel)  #-- update parameters

        epochLoss += Nb * loss
        epochAcc  += Nb * Accuracy(mZ, vY)
        print(f'\rIteration: {ii:3d}: loss = {loss:2.6f}', end='')

    print('', end='\r')
    epochLoss /= count
    epochAcc  /= count

    return epochLoss, epochAcc

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainModel(oModel, oTrainData, oValData, nEpochs, oOptim):

    vTrainLoss = np.full(nEpochs, np.nan)
    vTrainAcc  = np.full(nEpochs, np.nan)
    vValLoss   = np.full(nEpochs, np.nan)
    vValAcc    = np.full(nEpochs, np.nan)
    for epoch in range(nEpochs):
        startTime           = time.time()
        trainLoss, trainAcc = Epoch(oModel, oTrainData, oOptim, bTrain=True )
        valLoss,   valAcc   = Epoch(oModel, oValData,           bTrain=False)
        epochTime           = time.time() - startTime

        #-- Display:
        print('Epoch '              f'{epoch    :03d}:',  end='')
        print(' | Train loss: '     f'{trainLoss:.5f}' ,  end='')
        print(' | Val loss: '       f'{valLoss  :.5f}' ,  end='')
        print(' | Train Accuracy: ' f'{trainAcc :2.4f}',  end='')
        print(' | Val Accuracy: '   f'{valAcc   :2.4f}',  end='')
        print(' | epoch time: '     f'{epochTime:3.3f} |'       )

        vTrainLoss[epoch] = trainLoss
        vTrainAcc [epoch] = trainAcc
        vValLoss  [epoch] = valLoss
        vValAcc   [epoch] = valAcc

    return vTrainLoss, vTrainAcc, vValLoss, vValAcc

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
