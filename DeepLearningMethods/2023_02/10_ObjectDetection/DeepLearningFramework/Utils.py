import numpy             as np
import matplotlib.pyplot as plt
import torch

from matplotlib.patches import Rectangle
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def PlotImage(mX, mBBox=None, vLabels=None, lColors=None):
    if lColors is None:
        cmap    = plt.get_cmap('tab20b')
        lColors = [cmap(ii) for ii in np.linspace(0, 1, len(vLabels))]
    
    mI      = mX.permute(1,2,0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mI, extent=[0, 1, 1, 0])
    # ax.axis(False)
    
    if mBBox is None:
        return fig
    
    for vBBox in mBBox:
        # p, cIdx, xLeft, yDown, xRight, yUp = vBBox
        cIdx, xCenter, yCenter, W, H = vBBox
        xLeft = xCenter - W / 2
        yUp   = yCenter - H / 2
        
        cIdx  = int(cIdx)
        # W     = xRight - xLeft
        # H     = yUp    - yDown
        
        color = lColors[cIdx]
        oBbox = Rectangle((xLeft, yUp), W, H, linewidth=2, edgecolor=color, facecolor='none')
        
        ax.add_patch(oBbox)
        ax.text(xLeft, yUp, s=vLabels[cIdx], color='w', verticalalignment='bottom', bbox={'color':color}, fontdict={'size':16})
        
#         ax.plot(x, y, 's', mew=5, ms=10, color='w')
#         ax.plot(x, y, 'x', mew=5, ms=10, color=color)
    
    return fig
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def CreateImage(imageSize, vX, vY, vW, vH, vIdx):
    mColor   = torch.eye(3)
    nObjects = vX.shape[0]
    nColors  = len(mColor)

    v       = torch.linspace(0, 1, imageSize)
    YY, XX  = torch.meshgrid(v, v)
    
    vSize = [3, *XX.shape]
    mI    = torch.zeros(vSize)
    mBBox = torch.empty(nObjects, 6)
    for ii in range(nObjects):
        #-- Ellipse:
        vC   = mColor[vIdx[ii]]
        mE   = ((XX - vX[ii]) / vW[ii]) ** 2 + ((YY - vY[ii]) / vH[ii])**2 <= 1
        mI  += mE[None,:,:] * vC[:,None,None] / 2
        
        #-- BBox:
        xLeft   = np.maximum(0, vX[ii] - vW[ii])
        xRight  = np.minimum(1, vX[ii] + vW[ii])
        yUp     = np.maximum(0, vY[ii] - vH[ii])
        yDown   = np.minimum(1, vY[ii] + vH[ii])
        xCenter = (xLeft + xRight) / 2
        yCenter = (yDown + yUp   ) / 2
        W       = xRight - xLeft
        H       = yDown  - yUp
        lBBox   = [xCenter, yCenter, W, H]
        
        mBBox[ii,:] = torch.tensor([1, vIdx[ii], *lBBox])
    
    mI                      = torch.clamp(mI, 0, 1)
    mI[:,mI.max(0)[0] == 0] = 1
    return mI, mBBox
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#      
def RandImage(imageSize, nObjects):
    vX, vY = np.random.rand   (2, nObjects)
    vW, vH = np.random.rand   (2, nObjects) / 2
    vIdx   = np.random.randint(3, size=(nObjects,))
    
    mI, mBBox = CreateImage(imageSize, vX, vY, vW, vH, vIdx)
    
    return mI, mBBox
    
import numpy             as np
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def PlotHistory(lHistory):

    vTrainLoss, vTrainAcc, vValLoss, vValAcc, vLR = lHistory

    _, vAx = plt.subplots(1, 3, figsize=(22, 5))

    vAx[0].plot      (vTrainLoss, lw=2, label='Train'      f'={vTrainLoss.min():.4f}')
    vAx[0].plot      (vValLoss,   lw=2, label='Validation' f'={vValLoss  .min():.4f}')
    vAx[0].set_title ('Loss')
    vAx[0].set_xlabel('epoch')
    vAx[0].set_ylim  (bottom=0)
    vAx[0].grid      ()
    vAx[0].legend    ()

    vAx[1].plot      (vTrainAcc, lw=2, label='Train'      f'={vTrainAcc.max():.4f}')
    vAx[1].plot      (vValAcc,   lw=2, label='Validation' f'={vValAcc  .max():.4f}')
    vAx[1].set_title ('Metric')
    vAx[1].set_xlabel('epoch')
    vAx[1].set_ylim  (top=1)
    vAx[1].grid      ()
    vAx[1].legend    ()

    vAx[2].plot      (vLR, lw=2)
    vAx[2].set_title ('Learning rate')
    vAx[2].set_xlabel('iteration')
    vAx[2].grid      ()
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#