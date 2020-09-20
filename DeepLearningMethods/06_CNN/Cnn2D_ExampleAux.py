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
def Accuracy(mHatY, vY):
    vHatY = mHatY.argmax(dim=1)
    return (vHatY == vY).float().mean().item()


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
import time

def TrainLoop(oModel, oTrainDL, LossFunc, oOptim):

    epochLoss = 0
    epochAcc  = 0
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
        mHatY = oModel(mX)
        loss  = LossFunc(mHatY, vY)

        #-- Backward:
        loss.backward()

        #-- Parameters update:
        oOptim.step()

        print(f'Iteration: {ii:3d}: loss = {loss:.6f}\r', end='')
        #-- Accumulate loss:
        Nb         = mX.shape[0]
        epochLoss += Nb * loss.item()
        epochAcc  += Nb * Accuracy(mHatY, vY)
        count     += Nb

    epochLoss /= count
    epochAcc  /= count

    return epochLoss, epochAcc

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def ValidationLoop(oModel, oValDL, LossFunc):

    epochLoss = 0
    epochAcc  = 0
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
            mHatY = oModel(mX)
            loss  = LossFunc(mHatY, vY)

            Nb         = len(vY)
            epochLoss += Nb * loss.item()
            epochAcc  += Nb * Accuracy(mHatY, vY)
            count     += Nb

    epochLoss /= count
    epochAcc  /= count

    return epochLoss, epochAcc

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def TrainModel(oModel, oTrainDL, oValDL, LossFunc, numEpochs, oOptim):

    oRecorder = Recorder([
        Plot('Loss',     'train', 'epoch', 'b'),
        Plot('Loss',     'val',   'epoch', 'r'),
        Plot('Accuracy', 'train', 'epoch', 'b'),
        Plot('Accuracy', 'val',   'epoch', 'r'),
    ])

    bestAcc = 0
    for epoch in range(numEpochs):
        startTime           = time.time()
        trainLoss, trainAcc = TrainLoop(oModel, oTrainDL, LossFunc, oOptim) #-- train
        valLoss,   valAcc   = ValidationLoop(oModel, oValDL, LossFunc)      #-- validation

        #-- Display:
        oRecorder.Append('Loss',     'train', trainLoss),
        oRecorder.Append('Loss',     'val',   valLoss),
        oRecorder.Append('Accuracy', 'train', trainAcc),
        oRecorder.Append('Accuracy', 'val',   valAcc),
        oRecorder.Draw()

        endTime = time.time()
        print('Epoch '              f'{epoch:3d}:',                  end='')
        print(' | Train loss: '     f'{trainLoss:.5f}',              end='')
        print(' | Val loss: '       f'{valLoss:.5f}',                end='')
        print(' | Train Accuracy: ' f'{trainAcc:2.4f}',              end='')
        print(' | Val Accuracy: '   f'{valAcc:2.4f}',                end='')
        print(' | epoch time: '     f'{(endTime-startTime):3.3f} |', end='')
        
        #-- Save best model:
        if valAcc > bestAcc:
            bestAcc   = valAcc
            try:
                torch.save(oModel.state_dict(), 'BestModelParameters.pt')
            except:
                pass
            print(' <-- Checkpoint!')
        else:
            print()

    oModel.load_state_dict(torch.load('BestModelParameters.pt'))

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def LearningRateSweep(oModel, oTrainDL, LossFunc, oOptim, vLearningRate):
   
    #-- Record mini-batches loss:
    oRecorder = Recorder([Plot('Batch loss', 'train', 'learning rate', 'b')])
    device    = next(oModel.parameters()).device #-- CPU\GPU

    numIter = len(vLearningRate)
    ii      = 0
    while ii < numIter:
        #-- Iterate over the mini-batches:
        for mX, vY in oTrainDL:
            if ii >= numIter:
                break

            #-- Move to device (CPU\GPU):
            mX = mX.to(device)
            vY = vY.to(device)
                        
            #-- Set gradients to zeros:
            oOptim.zero_grad()
                
            #-- Forward:
            mHatY = oModel(mX)
            loss  = LossFunc(mHatY, vY)

            #-- Backward:
            loss.backward()

            #-- Update parameters (with new learning rate)
            oOptim.param_groups[0]['lr'] = vLearningRate[ii]
            oOptim.step()

            oRecorder.Append('Batch loss', 'train', loss.item())
            oRecorder.Draw()        
        
            ii += 1
            
    #-- Display:
    ax = oRecorder.dAxes['Batch loss']
    ax.lines[0].set_xdata(vLearningRate)
    ax.axis(xmin=vLearningRate[0], xmax=vLearningRate[-1])
    ax.set_xscale('log')
    oRecorder.Draw()

    return oRecorder