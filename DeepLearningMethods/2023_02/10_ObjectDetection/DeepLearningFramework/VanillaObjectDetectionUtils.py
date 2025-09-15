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
    mBBox = torch.empty(nObjects, 6)
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
        
        mBBox[ii,:] = torch.tensor([1, cIdx, *lBBox])
    
    mI                      = torch.clamp(mI, 0, 1)
    mI[:,mI.max(0)[0] == 0] = 1
    return mI, mBBox
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def PlotImage(mX, mBBox=None, vLabels=None, lColors=None, gridSize=5):
    if lColors is None:
        cmap    = plt.get_cmap('tab20b')
        lColors = [cmap(ii) for ii in np.linspace(0, 1, len(vLabels))]
    
    mI      = mX.permute(1,2,0).numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow    (mI, extent=[0, 1, 1, 0], origin='upper')
    ax.axis      ([0, 1, 1, 0])
    ax.set_xticks(np.arange(0, 1, 1 / gridSize))
    ax.set_yticks(np.arange(0, 1, 1 / gridSize))
    ax.grid      (True)    
    
    if mBBox is None:
        return fig
    
    for vBBox in mBBox:
        p, cIdx, xCenter, yCenter, W, H = vBBox
           
        xLeft = xCenter - W / 2
        yUp   = yCenter - H / 2
        cIdx  = int(cIdx)
        
        color = lColors[cIdx]
        oBbox = Rectangle((xLeft, yUp), W, H, linewidth=2, edgecolor=color, facecolor='none')
        
        ax.add_patch(oBbox)
        ax.text(xLeft, yUp, s=vLabels[cIdx], color='w', verticalalignment='bottom', bbox={'color':color}, fontdict={'size':16})
        
        ax.plot(xCenter, yCenter, 's', mew=5, ms=10, color='w')
        ax.plot(xCenter, yCenter, 'x', mew=5, ms=10, color=color)
    
    return fig
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def BBox2Target(mBBox, gridSize):
    '''
    * Input:
        - mBBox .shape = (D, 6)
        - mBBox [ii,:] = [prob | cIdx | x, y, W, H]
    * Output:
        - mTarget.shape    = (6, S, S)
        - mTarget[:,ii,jj] = [prob | x, y, W, H | cIdx]
    '''
    D        = mBBox.shape[0]
    S        = gridSize
    mProb    = torch.zeros(1, S, S)
    mLabel   = torch.zeros(1, S, S)
    mBBoxOut = torch.zeros(4, S, S)
    
    if D > 0:
        vP, vIdx        = mBBox[:,:2].T
        mXYWH           = mBBox[:,2:] * gridSize
        vX, vY, vW ,vH  = mXYWH.T
        vCx             = vX.floor().long() #-- cell x index
        vCy             = vY.floor().long() #-- cell y index
        vX             -= vCx               #-- cell x
        vY             -= vCy               #-- cell y

        mProb   [0,vCy,vCx] = vP
        mLabel  [0,vCy,vCx] = vIdx
        mBBoxOut[:,vCy,vCx] = torch.stack([vX, vY, vW, vH])
    
    mTarget = torch.cat([mProb, mBBoxOut, mLabel])
        
    return mTarget
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
class VanillaObjectDataset(torch.utils.data.Dataset):
    def __init__(self, mX, lBBox, gridSize):
        #-- mX       .shape = (N, C, W, H)
        #-- len(lBBox)      =  N
        #-- lBBox[ii].shape = (Di, 6)
        self.mX       = mX
        self.lBBox    = lBBox
        self.gridSize = gridSize
        
    def __len__(self):
        return self.mX.shape[0]
    
    def __getitem__(self, idx):
        mXi     = self.mX   [idx]
        mBBox   = self.lBBox[idx]
        mTarget = BBox2Target(mBBox, self.gridSize)
        return mXi, mBBox, mTarget
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def VanillaObjectCollate(lBatch):
    
    lX, lBBox, lTarget = zip(*lBatch)
    mX                 = torch.stack(lX)
    mTarget            = torch.stack(lTarget)
        
    return mX, lBBox, mTarget
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#