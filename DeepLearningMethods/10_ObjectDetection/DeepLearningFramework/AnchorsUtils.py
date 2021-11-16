import numpy             as np
import matplotlib.pyplot as plt
import torch
import torch.nn          as nn

from matplotlib.patches import Rectangle
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def CreateImage(imageSize, vX, vY, vW, vH, vIdx):
    mColor   = torch.eye(3)
    nObjects = vX.shape[0]
    nColors  = len(mColor)

    v       = torch.linspace(0, 1, imageSize)
    XX, YY  = torch.meshgrid(v, v, indexing='xy')
    
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
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def RandImage(imageSize, nObjects):
    vX, vY = np.random.rand   (2, nObjects)
    vW, vH = np.random.rand   (2, nObjects) / 4 + 0.1
    vIdx   = np.random.randint(3, size=(nObjects,))
    
    mI, mBBox = CreateImage(imageSize, vX, vY, vW, vH, vIdx)
    
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
import torchvision.ops as ops

def ComputeAnchorIoU(mXYWH, mAnchors):
    #-- mXYWH   .shape = (D, 4)
    #-- mAnchors.shape = (B, 4)
        
    D         = mXYWH.shape[0]
    B         = mAnchors.shape[0]
    m00WH     = torch.cat([torch.zeros(D, 2), mXYWH[:,2:]], dim=1) #-- (D, 4)
    m00Anchor = torch.cat([torch.zeros(B, 2), mAnchors   ], dim=1) #-- (B, 4)
    mB1       = ops.box_convert(m00WH,     'xywh', 'xyxy')
    mB2       = ops.box_convert(m00Anchor, 'xywh', 'xyxy')
    mIoU      = ops.box_iou(mB1, mB2)                              #-- (D, B)
    
    return mIoU
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def BBox2Target(mBBox, mAnchors, gridSize):
    '''
    * Input:
        - mBBox   .shape = (D, 6)
        - mAnchors.shape = (B, 2)
        - mBBox   [ii,:] = [prob | cIdx | x, y, W, H]
        - mAnchors[ii,:] = [W, H]
    * Output:
        - mTarget.shape       = (B, 6, S, S)
        - mTarget[ii,:,jj,kk] = [prob | x, y, log(W), log(H) | cIdx]
    '''
    B       = mAnchors.shape[0]
    D       = mBBox   .shape[0]
    mTarget = torch   .zeros(B, 6, gridSize, gridSize)
    
    if D == 0:
        return mTarget

    #-- Compute IoU between objectes and anchors:
    mXYWH = mBBox[:,2:]
    mIoU  = ComputeAnchorIoU(mXYWH, mAnchors) #-- (D, B)
    vIdx  = torch.argmax(mIoU, dim=1)
    
    #-- Set target for each anchor:
    for bb in range(B):
        mProb  = torch.zeros(1, gridSize, gridSize)
        mLabel = torch.zeros(1, gridSize, gridSize)
        mXYwh  = torch.zeros(4, gridSize, gridSize) #-- w = log(W), h = log(H)

        vIdxB           = vIdx == bb        #-- Objects that correspond to anchor bb
        vP, vLabel      = mBBox[vIdxB,:2].T
        mXYWH           = mBBox[vIdxB,2:] * gridSize
        vX, vY, vW ,vH  = mXYWH.T
        vCx             = vX.floor().long() #-- cell x index
        vCy             = vY.floor().long() #-- cell y index
        vX             -= vCx               #-- cell x
        vY             -= vCy               #-- cell y

        mProb  [0,vCy,vCx] = vP
        mLabel [0,vCy,vCx] = vLabel
        mXYwh  [:,vCy,vCx] = torch.stack([vX, vY, torch.log(vW), torch.log(vH)])
        mTarget[bb]        = torch.cat  ([mProb, mXYwh, mLabel],               )

    return mTarget
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
class AnchorObjectDataset(torch.utils.data.Dataset):
    def __init__(self, mX, lBBox, mAnchors, gridSize):
        #-- mX       .shape = (N, C, W, H)
        #-- len(lBBox)      =  N
        #-- lBBox[ii].shape = (Di, 6)
        #-- mAnchors .shape = (B, 2)
        self.mX       = mX
        self.lBBox    = lBBox
        self.mAnchors = mAnchors
        self.gridSize = gridSize

    def __len__(self):
        return self.mX.shape[0]

    def __getitem__(self, idx):
        mXi     = self.mX   [idx]
        mBBox   = self.lBBox[idx]
        mTarget = BBox2Target(mBBox, self.mAnchors, self.gridSize)
        return mXi, mBBox, mTarget
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def AnchorObjectCollate(lBatch):

    lX, lBBox, lTarget = zip(*lBatch)
    mX                 = torch.stack(lX)
    mTarget            = torch.stack(lTarget)

    return mX, lBBox, mTarget
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def Loss(mOutput, mTarget):
    #-- mOutput.shape       = (N, B, 8, 5, 5)
    #-- mTarget.shape       = (N, B, 6, 5, 5)
    #-- mOutput[ii,bb,:,jj,kk] = [p | x, y, W, H | R, G, B]
    #-- mTarget[ii,bb,:,jj,kk] = [p | x, y, W, H | cIdx]
    mProb      = mTarget[:,:,0,:,:]
    mMask      = mProb == 1
    
    mBBox      = mTarget[:,:,1:5,:,:].permute(0,1,3,4,2)[mMask] #-- mBBox .shape = (T, 4)
    vLabel     = mTarget[:,:,5  ,:,:].long()            [mMask] #-- vLabel.shape = (T)
    
    mProbPred  = mOutput[:,:,0,  :,:] 
    mBBoxPred  = mOutput[:,:,1:5,:,:].permute(0,1,3,4,2)[mMask]  #-- mBBoxPred .shape = (T, 4)
    mLabelPred = mOutput[:,:,5:, :,:].permute(0,1,3,4,2)[mMask]  #-- mLabelPred.shape = (T, L)
    
    BCE = nn.BCEWithLogitsLoss()(mProbPred,  mProb)
    MSE = nn.MSELoss          ()(mBBoxPred,  mBBox)
    CE  = nn.CrossEntropyLoss ()(mLabelPred, vLabel)
    
    loss = BCE + MSE + CE
    
    return loss, BCE.item(), CE.item(), MSE.item()
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def Output2Target(mOutput):
    #-- mOutput.shape          = (N, B, 5+L, S, S)
    #-- mOutput[ii,bb,:,jj,kk] = [p | x, y, log(W), log(H) | C1, C2, C3, ...]
    
    mProb  = torch.sigmoid(mOutput[:,:,[0],:,:])
    mBBox  =               mOutput[:,:,1:5,:,:]
    mLabel = torch.argmax (mOutput[:,:,5:, :,:], dim=2, keepdims=True)
    
    mTarget = torch.cat([mProb, mBBox, mLabel], dim=2)
    return mTarget
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def Target2BBox(mTarget, gridSize):
    '''
    * Input:
        - mTarget.shape          = (N, B, 6, S, S)
        - mTarget[ii,bb,:,jj,kk] = [p | x, y, log(W), log(H) | cIdx]
    * Output:
        - mBBox.shape    = (N, B*S*S, 6)
        - mBBox[ii,jj,:] = [p | cIdx | x, y, W, H]
    '''
    v       = torch.arange(gridSize, device=mTarget.device)
    XX, YY  = torch.meshgrid(v, v, indexing='xy')
    
    mProb  = mTarget[:,:,[0],:,:]
    mXYWH  = mTarget[:,:,1:5,:,:].clone()
    mLabel = mTarget[:,:,[5],:,:]
    
    mXYWH[:,:,0,  :,:] += XX[None,None,:,:]
    mXYWH[:,:,1,  :,:] += YY[None,None,:,:]
    mXYWH[:,:,2:4,:,:].exp_()
    mXYWH              /= gridSize
    
    B     = mTarget.shape[1]
    mBBox = torch.cat([mProb, mLabel, mXYWH], dim=2)                 #-- mBBox.shape = (N, B, 6, S, S)
    mBBox = mBBox.permute(0,1,3,4,2).reshape(-1, B * gridSize**2, 6) #-- mBBox.shape = (N, B*S*S, 6)
        
    return mBBox
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
def NMS(mBBox, IoUThreshold=0.5, pThreshold=0.2):
    #-- mBBox.shape    = (N, D, 6)
    #-- mBBox[ii,jj,:] = [prob | cIdx | x, y, W, H]
    mBBoxCPU = mBBox.cpu() #-- GPU is slower
    N        = mBBox.shape[0]
    lBBox    = [torch.empty(0, 6)] * N
    for ii in range(N):
        vIdx = mBBoxCPU[ii,:,0] > pThreshold
        if torch.any(vIdx):
            mBBc      = mBBoxCPU[ii,vIdx]
            vIdx      = ops.batched_nms(ops.box_convert(mBBc[:,2:], 'cxcywh', 'xyxy'), mBBc[:,0], mBBc[:,1], IoUThreshold)
            lBBox[ii] = mBBc[vIdx]

    return lBBox
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#