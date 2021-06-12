import numpy as np

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Optimizer:
    def __init__(self, oUpdateRule):
        self.oUpdateRule = oUpdateRule #-- SGD, SGDM, ADAM
        self.dStates     = {}

    def Step(self, oModel, lr=None):
        if lr is not None:
            self.oUpdateRule.lr = lr

        for ii, oLayer in enumerate(oModel.lLayers):
            for sParamKey in oLayer.dGrads:
                #-- Get param, gradient and history:
                mP       = oLayer.dParams[sParamKey]
                mDp      = oLayer.dGrads [sParamKey]
                sParamID = f'{ii}_{sParamKey}'
                dState   = self.dStates.get(sParamID, {})

                #-- Apply step:
                mP, dState = self.oUpdateRule.Step(mP, mDp, dState)

                #-- Set param and hisoty:
                oLayer.dParams[sParamKey] = mP
                self.dStates  [sParamID ] = dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
#-- Basic gradient descent update rule:
class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def Step(self, mP, mDp, dState={}):
        mP -= self.lr * mDp
        return mP, dState


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class SGDM:
    def __init__(self, lr=1e-3, beta=0.9):
        self.lr   = lr
        self.beta = beta

    def Step(self, mP, mDp, dState={}):
        mV            = dState.get('mV', np.zeros(mP.shape))
        mV            = self.beta * mV - self.lr * mDp
        mP           += mV
        dState['mV']  = mV

        return mP, dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps

    def Step(self, mP, mDp, dState={}):
        mV            = dState.get('mV', np.zeros(mP.shape))
        mS            = dState.get('mS', np.zeros(mP.shape))
        ii            = dState.get('ii', 0) + 1

        mV            = self.beta1 * mV + (1 - self.beta1) * mDp
        mS            = self.beta2 * mS + (1 - self.beta2) * mDp * mDp

        mTildeV       = mV / (1 - self.beta1**ii)
        mTildeS       = mS / (1 - self.beta2**ii)

        mP           -= self.lr * mTildeV / (np.sqrt(mTildeS) + self.eps)
        dState['mV']  = mV
        dState['mS']  = mS
        dState['ii']  = ii

        return mP, dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class AdamW:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, wd=0):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.wd    = wd #-- weight decay

    def Step(self, mW, mDw, dState={}):
        mV            = dState.get('mV', np.zeros(mW.shape))
        mS            = dState.get('mS', np.zeros(mW.shape))
        ii            = dState.get('ii', 0) + 1

        mV            = self.beta1 * mV + (1 - self.beta1) * mDw
        mS            = self.beta2 * mS + (1 - self.beta2) * mDw * mDw

        mTildeV       = mV / (1 - self.beta1**ii)
        mTildeS       = mS / (1 - self.beta2**ii)

        mW           -= self.lr * mTildeV / (np.sqrt(mTildeS) + self.eps) + self.wd * mW
        dState['mV']  = mV
        dState['mS']  = mS
        dState['ii']  = ii

        return mW, dState