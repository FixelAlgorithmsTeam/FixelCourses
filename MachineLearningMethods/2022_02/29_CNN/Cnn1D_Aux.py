import numpy             as np
import matplotlib.pyplot as plt
import torch
import time


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
from sklearn.metrics import r2_score

def R2Score(vHatY, vY):
    return r2_score(vY.detach().cpu(), vHatY.detach().cpu())

import time

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainLoop(oModel, oTrainDL, LossFunc, oOptim):

    epochLoss = 0
    epochR2   = 0
    count     = 0                                #-- number of samples
    device    = next(oModel.parameters()).device #-- CPU\GPU
    oModel.train(True)
        
    #-- Iterate over the mini-batches:
    for ii, (mX, vY) in enumerate(oTrainDL):
        #-- Move to device (CPU\GPU):
        mX = mX.to(device)
        vY = vY.to(device)
        
        #-- Set gradients to zeros:
        oOptim.zero_grad()
        
        #-- Forward:
        vHatY = oModel(mX).squeeze()
        loss  = LossFunc(vHatY, vY)

        #-- Backward:
        loss.backward()

        #-- Parameters update:
        oOptim.step()
        
        print(f'\rIteration: {ii:3d}: loss = {loss:.6f}', end='')
        #-- Accumulate loss:
        Nb         = mX.shape[0]
        epochLoss += Nb * loss.item()
        epochR2   += Nb * R2Score(vHatY, vY)
        count     += Nb

    print('', end='\r')
    epochLoss /= count
    epochR2   /= count
   
    return epochLoss, epochR2

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def ValidationLoop(oModel, oValDL, LossFunc):

    if oValDL is None:
        return 0, 0
    
    epochLoss = 0
    epochR2   = 0
    count     = 0                                #-- number of samples
    device    = next(oModel.parameters()).device #-- CPU\GPU
    
    #-- Iterate over the mini-batches:
    oModel.train(False)
    with torch.no_grad():
        for ii, (mX, vY) in enumerate(oValDL):
            #-- Move to device (CPU\GPU):
            mX = mX.to(device)
            vY = vY.to(device)
            
            #-- Forward:
            vHatY = oModel(mX).squeeze()
            loss  = LossFunc(vHatY, vY)

            Nb         = mX.shape[0]
            epochLoss += Nb * loss.item()
            epochR2   += Nb * R2Score(vHatY, vY)
            count     += Nb

    epochLoss /= count
    epochR2   /= count

    return epochLoss, epochR2

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainModel(oModel, oTrainDL, oValDL, LossFunc, numEpochs, oOptim):

    oRecorder = Recorder([
        Plot('Loss',     'train', 'epoch', 'b'),
        Plot('Loss',     'val',   'epoch', 'r'),
        Plot('R2 score', 'train', 'epoch', 'b'),
        Plot('R2 score', 'val',   'epoch', 'r'),
    ])

    bestR2 = 0
    for epoch in range(numEpochs):
        
        startTime          = time.time()
        trainLoss, trainR2 = TrainLoop     (oModel, oTrainDL, LossFunc, oOptim) #-- train
        valLoss,   valR2   = ValidationLoop(oModel, oValDL,   LossFunc)         #-- validation
        
        #-- Display:
        oRecorder.Append('Loss',     'train', trainLoss),
        oRecorder.Append('Loss',     'val',   valLoss),
        oRecorder.Append('R2 score', 'train', trainR2),
        oRecorder.Append('R2 score', 'val',   valR2),
        oRecorder.Draw()

        endTime = time.time()
        print('Epoch '              f'{epoch:3d}:',                  end='')
        print(' | Train loss: '     f'{trainLoss:.5f}',              end='')
        print(' | Val loss: '       f'{valLoss:.5f}',                end='')
        print(' | Train R2 score: ' f'{trainR2:2.4f}',               end='')
        print(' | Val R2 score: '   f'{valR2:2.4f}',                 end='')
        print(' | epoch time: '     f'{(endTime-startTime):3.3f} |', end='')
        
        #-- Save best model:
        if valR2 > bestR2:
            bestR2 = valR2
            try:
                torch.save(oModel.state_dict(), 'BestModelParameters.pt')
            except:
                pass
            print(' <-- Checkpoint!')
        else:
            print()

    oModel.load_state_dict(torch.load('BestModelParameters.pt'))