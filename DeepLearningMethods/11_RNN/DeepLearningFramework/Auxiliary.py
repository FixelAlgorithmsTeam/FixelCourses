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
def RandBox(W, H, λ):

    xCenter = np.random.randint(W)
    yCenter = np.random.randint(H)

    ratio = np.sqrt (1 - λ)
    w     = np.int32(W * ratio)
    h     = np.int32(H * ratio)

    xLow  = np.maximum(xCenter - w//2, 0)
    yLow  = np.maximum(yCenter - h//2, 0)
    xHigh = np.minimum(xCenter + w//2, W)
    yHigh = np.minimum(yCenter + h//2, H)

    return xLow, yLow, xHigh, yHigh
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#