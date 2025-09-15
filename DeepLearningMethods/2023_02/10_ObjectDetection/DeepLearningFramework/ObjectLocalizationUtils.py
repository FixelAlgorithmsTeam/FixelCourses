import numpy             as np
import matplotlib.pyplot as plt
import torch

from matplotlib.patches import Rectangle
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def CreateImage(imageSize, nObjects):
    mColor  = torch.eye(3)
    nColors = len(mColor)

    v       = torch.linspace(0, 1, imageSize)
    XX, YY  = torch.meshgrid(v, v, indexing='xy')
    
    vSize = [3, *XX.shape]
    mI    = torch.zeros(vSize)
    mBBox = torch.empty(nObjects, 5)
    for ii in range(nObjects):
        #-- Ellipse:
        cIdx = np.random.randint(nColors)
        vC   = mColor[cIdx]
        x    = np.random.rand()
        y    = np.random.rand()
        W    = np.random.rand() / 2
        H    = np.random.rand() / 2
        mE   = ((XX - x) / W) ** 2 + ((YY - y) / H)**2 <= 1
        mI  += mE[None,:,:] * vC[:,None,None] / 2
        
        #-- BBox:
        xLeft   = np.maximum(0, x - W)
        xRight  = np.minimum(1, x + W)
        yUp     = np.maximum(0, y - H)
        yDown   = np.minimum(1, y + H)
        xCenter = (xLeft + xRight) / 2
        yCenter = (yDown + yUp   ) / 2
        W       = xRight - xLeft
        H       = yDown  - yUp
        lBBox   = [xCenter, yCenter, W, H]
        
        mBBox[ii,:] = torch.tensor([cIdx, *lBBox])
    
    mI                      = torch.clamp(mI, 0, 1)
    mI[:,mI.max(0)[0] == 0] = 1
    return mI, mBBox
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
class ObjectLocalizationDataset(torch.utils.data.Dataset):
    def __init__(self, mX, mBBox):
        self.mX    = mX
        self.mBBox = mBBox

    def __len__(self):
        return self.mX.shape[0]

    def __getitem__(self, idx):
        mXi = self.mX   [idx]
        mBi = self.mBBox[idx]
        return mXi, mBi
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def Area(mBBox):
    #-- mBBox.shape = (N, 4)
    #-- mBBox[ii,:] = [xLeft, yUp, xRight, yDown]
    
    vXLeft, vYUp, vXRight, vYDown = mBBox.T
    return (vXRight - vXLeft).clamp(0) * (vYDown - vYUp).clamp(0)
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
def CenterToCorner(mBBox):
    #-- mBBox.shape = (N, 4)
    #-- mBBox[ii,:] = [xCenter, yCenter, width, height]
    vX, vY, vW, vH = mBBox.T
    vXLeft         = vX - vW / 2
    vXRight        = vX + vW / 2
    vYUp           = vY - vH / 2
    vYDown         = vY + vH / 2
    mBBox          = torch.stack([vXLeft, vYUp, vXRight, vYDown], dim=-1)

    return mBBox
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
def ComputeIoU(mBBox1, mBBox2):
    #-- mBBox1.shape = (N, 4)
    #-- mBBox2.shape = (N, 4)
    #-- mBBox1[ii,:] = [xCenter, yCenter, width, height]
    #-- mBBox2[ii,:] = [xCenter, yCenter, width, height]
    mBBox1 = CenterToCorner(mBBox1)
    mBBox2 = CenterToCorner(mBBox2)

    mXLeft1, vYUp1, vXRight1, vYDown1  = mBBox1.T
    vXLeft2, vYUp2, vXRight2, vYDown2  = mBBox2.T
    
    vXLeft  = torch.maximum(mXLeft1,  vXLeft2 )
    vXRight = torch.minimum(vXRight1, vXRight2)
    vYUp    = torch.maximum(vYUp1,    vYUp2   )
    vYDown  = torch.minimum(vYDown1,  vYDown2 )

    mBBoxI  = torch.stack([vXLeft, vYUp, vXRight, vYDown], dim=1)
   
    vArea1 = Area(mBBox1)
    vArea2 = Area(mBBox2)
    vAreaI = Area(mBBoxI)
    
    vIoU   = vAreaI / (vArea1 + vArea2 - vAreaI) #-- vIoU.shape = (N,)
    
    return vIoU
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def Output2Target(mOutput):
    #-- mOutput.shape = (N, 7)
    #-- mOutput[ii,:] = [R, G, B, x, y, W, H]
    mClass = torch.argmax(mOutput[:,:3], dim=1, keepdims=True) #-- mClass.shape = (N, 1)
    mBBox  =              mOutput[:,3:]                        #-- mBBox .shape = (N, 4)
    mBBox  = torch.cat   ([mClass, mBBox], dim=1)              #-- mBBox .shape = (N, 5)
    return mBBox
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def Metric(mOutput, mBBox):
    #-- mOutput.shape = (N, 7)
    #-- mBBox  .shape = (N, 5)
    #-- mOutput[ii,:] = [R, G, B, x, y, W, H]
    #-- mBBox  [ii,:] = [cIdx,    x, y, W, H]
    
    mBBoxPred  = Output2Target(mOutput) #-- mBBoxPred[ii,:] = [cIdx, x, y, W, H]
    vLabelPred = mBBoxPred[:,0]
    vLabel     = mBBox    [:,0]
    mBBoxPred  = mBBoxPred[:,1:]
    mBBox      = mBBox    [:,1:]
    vIoU       = ComputeIoU(mBBoxPred, mBBox)
    vCorrect   = (vLabelPred == vLabel).type(torch.float32)

    acc        = vCorrect.mean()            .item()
    IoU        = vIoU    .mean()            .item()
    metric     = torch.inner(vIoU, vCorrect).item() / mBBox.shape[0]
    
    return metric, acc, IoU
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def PlotPredictions(mX, vBBox, vBBoxPred):

    IoU = ComputeIoU(vBBox[None,1:], vBBoxPred[None,1:])[0]

    def PlotBBox(ax, vBBox, color, sText=None, ls='-'):
        cIdx, xCenter, yCenter, W, H = vBBox
        xLeft = xCenter - W / 2
        yUp   = yCenter - H / 2
        oBbox = Rectangle((xLeft, yUp), W, H, linewidth=2, ls=ls, edgecolor=color, facecolor='none')
        ax.add_patch(oBbox)
        ax.text(xLeft, yUp, s=sText, color='w', verticalalignment='bottom', bbox={'color':color}, fontdict={'size':14})

        return ax

    mI      = mX.permute(1,2,0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mI, extent=[0, 1, 1, 0])
    # ax.axis(False)
    
    color = 'g' if vBBox[0] == vBBoxPred[0] else 'r'
    ax    = PlotBBox(ax, vBBox,     'g',   ls='--')
    ax    = PlotBBox(ax, vBBoxPred, color, f'IoU = {IoU}')
    
    return fig
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#