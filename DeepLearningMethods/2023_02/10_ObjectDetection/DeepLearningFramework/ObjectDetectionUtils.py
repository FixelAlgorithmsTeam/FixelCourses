import numpy             as np
import matplotlib.pyplot as plt
import torch
import torch.nn          as nn
import torchvision.ops   as ops

DEVICE = 'cuda'

GRID_SIZE = 5
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def ComputeIoU(mXYWH, mAnchors):
    #-- mXYWH   .shape = (D, 4)
    #-- mAnchors.shape = (B, 4)
        
    D         = mXYWH   .shape[0]
    B         = mAnchors.shape[0]
    m00WH     = torch.cat([torch.zeros(D, 2), mXYWH[:,2:]], dim=1) #-- (D, 4)
    m00Anchor = torch.cat([torch.zeros(B, 2), mAnchors   ], dim=1) #-- (B, 4)
    mB1       = ops.box_convert(m00WH,     'xywh', 'xyxy')
    mB2       = ops.box_convert(m00Anchor, 'xywh', 'xyxy')
    mIoU      = ops.box_iou(mB1, mB2)                              #-- (D, B)
    
    return mIoU
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def BBox2Target(mBBox, mAnchors):
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
    mTarget = torch   .zeros(B, 6, GRID_SIZE, GRID_SIZE)
    
    if D == 0:
        return mTarget

    #-- Compute IoU between objectes and anchors:
    mXYWH = mBBox[:,2:]
    mIoU  = ComputeIoU(mXYWH, mAnchors) #-- (D, B)
    vIdx  = torch.argmax(mIoU, dim=1)

    #-- Set target for each anchor:
    for bb in range(B):
        mProb  = torch.zeros(1, GRID_SIZE, GRID_SIZE)
        mLabel = torch.zeros(1, GRID_SIZE, GRID_SIZE)
        mXYwh  = torch.zeros(4, GRID_SIZE, GRID_SIZE) #-- w = log(W), h = log(H)

        vIdxB           = vIdx == bb        #-- Objects that correspond to anchor bb
        vP, vLabel      = mBBox[vIdxB,:2].T
        mXYWH           = mBBox[vIdxB,2:] * GRID_SIZE
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
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def Loss(mOutput, mTarget):
    #-- mOutput.shape       = (N, B, 8, 5, 5)
    #-- mTarget.shape       = (N, B, 6, 5, 5)
    #-- mOutput[ii,bb,:,jj,kk] = [p | x, y, W, H | R, G, B]
    #-- mTarget[ii,bb,:,jj,kk] = [p | x, y, W, H | cIdx]

    mProb      = mTarget[:,:,0,:,:]
    mProbPred  = mOutput[:,:,0,:,:] 
    mMask      = mProb == 1

    BCELoss    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2), reduction='none')
    BCE        = BCELoss(mProbPred,  mProb)[mProb != -1].mean()
    if torch.any(mMask):
        mBBox      = mTarget[:,:,1:5,:,:].permute(0,1,3,4,2)[mMask] #-- mBBox .shape = (T, 4)
        vLabel     = mTarget[:,:,5  ,:,:].long()            [mMask] #-- vLabel.shape = (T)
        
        mBBoxPred  = mOutput[:,:,1:5,:,:].permute(0,1,3,4,2)[mMask]  #-- mBBoxPred .shape = (T, 4)
        mLabelPred = mOutput[:,:,5:, :,:].permute(0,1,3,4,2)[mMask]  #-- mLabelPred.shape = (T, L)
        
        MSE = nn.MSELoss          ()(mBBoxPred,  mBBox)
        CE  = nn.CrossEntropyLoss ()(mLabelPred, vLabel)
        
        loss = BCE + MSE + CE
        return loss, BCE.item(), CE.item(), MSE.item()
    else:
        loss = BCE
        return loss, BCE.item(), 0, 0
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def Output2Target(mOutput):
    #-- mOutput.shape          = (N, B, 8, 5, 5)
    #-- mOutput[ii,bb,:,jj,kk] = [p | x, y, log(W), log(H) | R, G, B]
    
    mProb  = torch.sigmoid(mOutput[:,:,[0],:,:])
    mBBox  =               mOutput[:,:,1:5,:,:]
    mLabel = torch.argmax (mOutput[:,:,5:, :,:], dim=2, keepdims=True)
    
    mTarget = torch.cat([mProb, mBBox, mLabel], dim=2)
    return mTarget
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def Target2BBox(mTarget):
    '''
    * Input:
        - mTarget.shape          = (N, B, 6, S, S)
        - mTarget[ii,bb,:,jj,kk] = [p | x, y, log(W), log(H) | cIdx]
    * Output:
        - mBBox.shape    = (N, B*S*S, 6)
        - mBBox[ii,jj,:] = [p | cIdx | x, y, W, H]
    '''
    v       = torch.arange(GRID_SIZE, device=mTarget.device)
    XX, YY  = torch.meshgrid(v, v, indexing='xy')
    
    mProb  = mTarget[:,:,[0],:,:]
    mXYWH  = mTarget[:,:,1:5,:,:].clone()
    mLabel = mTarget[:,:,[5],:,:]
    
    mXYWH[:,:,0,  :,:] += XX[None,None,:,:]
    mXYWH[:,:,1,  :,:] += YY[None,None,:,:]
    mXYWH[:,:,2:4,:,:].exp_()
    mXYWH              /= GRID_SIZE
    
    B     = mTarget.shape[1]
    mBBox = torch.cat([mProb, mLabel, mXYWH], dim=2)                  #-- mBBox.shape = (N, B, 6, 5, 5)
    mBBox = mBBox.permute(0,1,3,4,2).reshape(-1, B * GRID_SIZE**2, 6) #-- mBBox.shape = (N, B*25, 6)
        
    return mBBox
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def NMS(mBBox, IoUThreshold=0.5, pThreshold=0.1):
    #-- mBBox.shape    = (N, D, 6)
    #-- mBBox[ii,jj,:] = [prob | cIdx | x, y, W, H]
    mBBoxCPU = mBBox.cpu() #-- GPU is slower
    N        = mBBox.shape[0]
    lBBox    = [torch.empty(0, 6)] * N
    for ii in range(N):
        vIdx  = mBBoxCPU[ii,:,0] > pThreshold
        if torch.any(vIdx):
            mBBc      = mBBoxCPU[ii,vIdx]
            vIdx      = ops.batched_nms(ops.box_convert(mBBc[:,2:], 'cxcywh', 'xyxy'), mBBc[:,0], mBBc[:,1], IoUThreshold)
            lBBox[ii] = mBBc[vIdx]

    return lBBox
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def CatWithIndex(lX):
    vLen         = torch.tensor([mX.shape[0] for mX in lX])
    vPos         = torch.cumsum(vLen, dim=0)
    L            = vPos[-1] #-- vLen.sum()
    vDelta       = torch.zeros(L + 1)
    vDelta[vPos] = 1
    vIdxZero     = torch.where(vLen == 0)
    for idx in vIdxZero:
        vDelta[vPos[idx]] += 1

    vIdx = torch.cumsum(vDelta[:-1], dim=0)
    mX   = torch.cat   (lX)

    return mX, vIdx
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def ComputeTable(lNMS, lBBox, nLabels):
    #-- len(lNMS)      = N1
    #-- len(lBBox)     = N2
    #-- lNMS[ii].shape = (Di, 6)
    #-- lNMS[ii][jj,:] = [prob | cIdx | x, y, W, H]
    #-- Same for lBBox
    
    mBBox, vImageIdx  = CatWithIndex(lBBox) #-- (N2, 6)
    mNMS,  vImageIdx2 = CatWithIndex(lNMS ) #-- (N1, 6)

    mBBox      = mBBox     .to(DEVICE)
    mNMS       = mNMS      .to(DEVICE)
    vImageIdx  = vImageIdx .to(DEVICE)
    vImageIdx2 = vImageIdx2.to(DEVICE)
    
    v           = torch.arange(nLabels, device=DEVICE)
    vLabelCount = (mBBox[:,[1]] == v[None,:]).sum(0)
    
    if mNMS.shape[0] == 0:
        return torch.empty(0, 3, device=DEVICE), vLabelCount

    mIoU = ops.box_iou(                                  #-- mIoU.shape = (N1, N2)
        ops.box_convert(mNMS [:,2:], 'cxcywh', 'xyxy'),
        ops.box_convert(mBBox[:,2:], 'cxcywh', 'xyxy')
    )
    
    mImageMask  = vImageIdx2[:,None] == vImageIdx[None,:]
    mLabelMask  = mNMS[:,[1]]        == mBBox[:,[1]].T
    mIoU       *= mImageMask & mLabelMask & (mIoU > 0.5) #-- (N1, N2)

    vMax, vIdx          = torch.max  (mIoU, dim=0)
    vTP                 = torch.zeros(mNMS.shape[0], device=DEVICE) #-- N1
    vTP[vIdx[vMax > 0]] = 1
    mTable              = torch.cat([mNMS[:,:2], vTP[:,None]], dim=1) #-- (N1, 3)
        
    return mTable, vLabelCount
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
def ComputeMAP(mTable, vLabelCount):
    vIdx    = torch.argsort(mTable[:,0], descending=True)
    mTable  = mTable[vIdx]
    nLabels = vLabelCount.shape[0]
        
    lAP = []
    for cc in range(nLabels):
        if vLabelCount[cc] == 0:
            continue

        vIdx  = mTable[:,1] == cc
        vTPc  = mTable[vIdx,2]
        nBBox = vTPc.shape[0]

        vPrecision     = torch.ones (nBBox + 1, device=DEVICE)
        vRecall        = torch.zeros(nBBox + 1, device=DEVICE)
        vCumsumTP      = torch.cumsum(vTPc, dim=0)

        vPrecision[1:] = vCumsumTP / torch.arange(1, nBBox+1, device=DEVICE)
        vRecall   [1:] = vCumsumTP / vLabelCount[cc]
        AP             = torch.trapz(vPrecision, vRecall)
        lAP += [AP]
        
    mAP = torch.tensor(lAP).mean().item()
    
    return mAP
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
    
















