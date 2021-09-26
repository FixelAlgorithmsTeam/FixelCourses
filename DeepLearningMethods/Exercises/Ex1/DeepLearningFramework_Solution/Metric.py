import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def CrossEntropyLoss(vY, mZ):
    '''
    Returns both the loss and the gradient w.r.t the input (mZ)
    '''
    N      = len(vY)
    mHatY  = np.exp(mZ)
    mHatY /= np.sum(mHatY, axis=0)
    loss   = -np.log(mHatY[vY,range(N)]).mean()

    mDz               = mHatY
    mDz[vY,range(N)] -= 1
    mDz              /= N

    return loss, mDz
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def MSE(vY, vHatY):
    '''
    Returns both the loss and the gradient w.r.t the input (vHatY)
    '''
    N    = len(vY)
    loss = np.linalg.norm(vHatY - vY)**2 / N
    vDy  = 2 * (vHatY - vY) / N

    return loss, vDy
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
#-- Classification accuracy:
def Accuracy(mScore, vY):
    vHatY = mScore.detach().argmax(dim=1)
    return (vHatY == vY).float().mean().item()
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
#-- Regression R²:
def R2(vY, vHatY):
    return r2_score(vY.reshape(-1), vHatY.reshape(-1))
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#