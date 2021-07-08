import numpy as np
import time
import torch

from DeepLearningFramework.Metric import Accuracy, R2Score

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def Epoch(oModel, oDataDL, Loss, Metric, oOptim=None, oScheduler=None, bTrain=True):

    epochLoss   = 0
    epochMetric = 0
    count       = 0
    nIter       = len(oDataDL)
    vLR         = np.full(nIter, np.nan)
    DEVICE      = next(oModel.parameters()).device #-- CPU\GPU


    oModel.train(bTrain) #-- train or test

    #-- Iterate over the mini-batches:
    for ii, (mX, vY) in enumerate(oDataDL):
        #-- Move to device (CPU\GPU):
        mX = mX.to(DEVICE)
        vY = vY.to(DEVICE)

        #-- Forward:
        if bTrain == True:
            #-- Store computational graph:
            mZ   = oModel(mX)
            loss = Loss(mZ, vY)
        else:
            with torch.no_grad():
                #-- Do not store computational graph:
                mZ   = oModel(mX)
                loss = Loss(mZ, vY)

        #-- Backward:
        if bTrain == True:
            oOptim.zero_grad() #-- set gradients to zeros
            loss.backward()    #-- backward
            oOptim.step()      #-- update parameters
            if oScheduler is not None:
                vLR[ii] = oScheduler.get_last_lr()[0]
                oScheduler.step() #-- update learning rate

        Nb           = vY.shape[0]
        count       += Nb
        epochLoss   += Nb * loss.item()
        epochMetric += Nb * Metric(mZ, vY)
        print(f'\r{"Train" if bTrain else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')

    print('', end='\r')
    epochLoss   /= count
    epochMetric /= count

    return epochLoss, epochMetric, vLR

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainClassficationModel(oModel, oTrainData, oValData, Loss, Metric, nEpochs, oOptim, oScheduler=None):

    vTrainLoss = np.full(nEpochs, np.nan)
    vTrainAcc  = np.full(nEpochs, np.nan)
    vValLoss   = np.full(nEpochs, np.nan)
    vValAcc    = np.full(nEpochs, np.nan)
    vLR        = np.full(0,       np.nan)
    bestAcc    = 0

    for epoch in range(nEpochs):
        startTime                 = time.time()
        trainLoss, trainAcc, vLRi = Epoch(oModel, oTrainData, Loss, Metric, oOptim, oScheduler, bTrain=True ) #-- train
        valLoss,   valAcc,   _    = Epoch(oModel, oValData,   Loss, Metric,                     bTrain=False) #-- validate
        epochTime                 = time.time() - startTime

        #-- Display:
        print('Epoch '            f'{epoch    :03d}:',   end='')
        print(' | Train loss: '   f'{trainLoss:.5f}' ,   end='')
        print(' | Val loss: '     f'{valLoss  :.5f}' ,   end='')
        print(' | Train Metric: ' f'{trainAcc :2.4f}',   end='')
        print(' | Val Metric: '   f'{valAcc   :2.4f}',   end='')
        print(' | epoch time: '   f'{epochTime:6.3f} |', end='')

        vTrainLoss[epoch] = trainLoss
        vTrainAcc [epoch] = trainAcc
        vValLoss  [epoch] = valLoss
        vValAcc   [epoch] = valAcc
        vLR               = np.concatenate([vLR, vLRi])

        #-- Save best model (early stopping):
        if bestAcc < valAcc:
            bestAcc = valAcc
            try:
                torch.save(oModel.state_dict(), 'BestModel.pt')
            except:
                pass
            print(' <-- Checkpoint!')
        else:
            print('')

    #-- Load best model (early stopping):
    oModel.load_state_dict(torch.load('BestModel.pt'))

    return vTrainLoss, vTrainAcc, vValLoss, vValAcc, vLR

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainRegressionModel(oModel, oTrainData, oValData, Loss, nEpochs, oOptim, oScheduler=None):

    vTrainLoss = np.full(nEpochs, np.nan)
    vTrainR2   = np.full(nEpochs, np.nan)
    vValLoss   = np.full(nEpochs, np.nan)
    vValR2     = np.full(nEpochs, np.nan)
    vLR        = np.full(0,       np.nan)
    bestR2     = -np.inf

    for epoch in range(nEpochs):
        startTime                = time.time()
        trainLoss, trainR2, vLRi = Epoch(oModel, oTrainData, Loss, R2Score, oOptim, oScheduler, bTrain=True ) #-- train
        valLoss,   valR2,   _    = Epoch(oModel, oValData,   Loss, R2Score,                     bTrain=False) #-- validate
        epochTime                = time.time() - startTime

        #-- Display:
        print('Epoch '          f'{epoch    :03d}:',   end='')
        print(' | Train loss: ' f'{trainLoss:.5f}' ,   end='')
        print(' | Val loss: '   f'{valLoss  :.5f}' ,   end='')
        print(' | Train R2: '   f'{trainR2 :+2.4f}',   end='')
        print(' | Val R2: '     f'{valR2   :+2.4f}',   end='')
        print(' | epoch time: ' f'{epochTime:6.3f} |', end='')

        vTrainLoss[epoch] = trainLoss
        vTrainR2  [epoch] = trainR2
        vValLoss  [epoch] = valLoss
        vValR2    [epoch] = valR2
        vLR               = np.concatenate([vLR, vLRi])

        #-- Save best model (early stopping):
        if bestR2 < valR2:
            bestR2 = valR2
            try:
                torch.save(oModel.state_dict(), 'BestModel.pt')
            except:
                pass
            print(' <-- Checkpoint!')
        else:
            print('')

    #-- Load best model (early stopping):
    oModel.load_state_dict(torch.load('BestModel.pt'))

    return vTrainLoss, vTrainR2, vValLoss, vValR2, vLR
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#