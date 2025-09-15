import numpy   as np
import _pickle as pickle
import time

from DeepLearningFramework.Metric import CrossEntropyLoss, Accuracy, MSE, R2

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def Epoch(oModel, oData, Loss, Metric, oOptim=None, bTrain=True):

    epochLoss   = 0
    epochMetric = 0
    count       = 0
    #-- Iterate over the mini-batches:
    for ii, (mX, vY) in enumerate(oData):

        Nb     = len(vY)
        count += Nb

        #-- Forward:
        mZ        = oModel.Forward(mX)
        loss, mDz = Loss(vY, mZ)

        if bTrain == True:
            oModel.Backward(mDz) #-- backward
            oOptim.Step(oModel)  #-- update parameters

        epochLoss   += Nb * loss
        epochMetric += Nb * Metric(vY, mZ)
        print(f'\rIteration: {ii:3d}: loss = {loss:2.6f}', end='')

    print('', end='\r')
    epochLoss   /= count
    epochMetric /= count

    return epochLoss, epochMetric

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainClassficationModel(oModel, oTrainData, oValData, nEpochs, oOptim):

    Loss   = CrossEntropyLoss
    Metric = Accuracy

    vTrainLoss = np.full(nEpochs, np.nan)
    vTrainAcc  = np.full(nEpochs, np.nan)
    vValLoss   = np.full(nEpochs, np.nan)
    vValAcc    = np.full(nEpochs, np.nan)
    bestAcc    = 0

    for epoch in range(nEpochs):
        startTime = time.time()
        oModel.train        = True
        trainLoss, trainAcc = Epoch(oModel, oTrainData, Loss, Metric, oOptim, bTrain=True )
        oModel.train        = False
        valLoss,   valAcc   = Epoch(oModel, oValData,   Loss, Metric,         bTrain=False)
        epochTime = time.time() - startTime

        #-- Display:
        print('Epoch '              f'{epoch    :03d}:',   end='')
        print(' | Train loss: '     f'{trainLoss:.5f}' ,   end='')
        print(' | Val loss: '       f'{valLoss  :.5f}' ,   end='')
        print(' | Train Accuracy: ' f'{trainAcc :2.4f}',   end='')
        print(' | Val Accuracy: '   f'{valAcc   :2.4f}',   end='')
        print(' | epoch time: '     f'{epochTime:3.3f} |', end='')

        vTrainLoss[epoch] = trainLoss
        vTrainAcc [epoch] = trainAcc
        vValLoss  [epoch] = valLoss
        vValAcc   [epoch] = valAcc

        #-- Save best model (early stopping):
        if valAcc > bestAcc:
            bestAcc = valAcc
            print(' <-- Checkpoint!', end='')
            with open('BestModel.pkl', 'wb') as oFile:
                pickle.dump(oModel, oFile)
        print('')

    #-- Load best model (early stopping):
    with open('BestModel.pkl', 'rb') as oFile:
        oModel = pickle.load(oFile)

    return oModel, (vTrainLoss, vTrainAcc, vValLoss, vValAcc)
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainRegressionModel(oModel, oTrainData, oValData, nEpochs, oOptim):

    Loss   = MSE
    Metric = R2

    vTrainLoss = np.full(nEpochs, np.nan)
    vTrainR2   = np.full(nEpochs, np.nan)
    vValLoss   = np.full(nEpochs, np.nan)
    vValR2     = np.full(nEpochs, np.nan)
    bestR2     = 0

    for epoch in range(nEpochs):
        startTime = time.time()
        oModel.train       = True
        trainLoss, trainR2 = Epoch(oModel, oTrainData, Loss, Metric, oOptim, bTrain=True )
        oModel.train       = False
        valLoss,   valR2   = Epoch(oModel, oValData,   Loss, Metric,         bTrain=False)
        epochTime = time.time() - startTime

        #-- Display:
        print('Epoch '          f'{epoch    :03d}:',   end='')
        print(' | Train loss: ' f'{trainLoss:.5f}' ,   end='')
        print(' | Val loss: '   f'{valLoss  :.5f}' ,   end='')
        print(' | Train R2: '   f'{trainR2 :+2.4f}',   end='')
        print(' | Val R2: '     f'{valR2   :+2.4f}',   end='')
        print(' | epoch time: ' f'{epochTime:3.3f} |', end='')

        vTrainLoss[epoch] = trainLoss
        vTrainR2  [epoch] = trainR2
        vValLoss  [epoch] = valLoss
        vValR2    [epoch] = valR2

        #-- Save best model (early stopping):
        if valR2 > bestR2:
            bestR2 = valR2
            print(' <-- Checkpoint!', end='')
            try:
                with open('BestModel.pkl', 'wb') as oFile:
                    pickle.dump(oModel, oFile)
            except:
                pass
        print('')

    #-- Load best model (early stopping):
    with open('BestModel.pkl', 'rb') as oFile:
        oModel = pickle.load(oFile)

    return oModel, (vTrainLoss, vTrainR2, vValLoss, vValR2)
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#