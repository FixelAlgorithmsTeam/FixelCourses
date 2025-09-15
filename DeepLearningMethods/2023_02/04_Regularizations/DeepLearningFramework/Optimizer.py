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
    def __init__(self, lr=1e-3, β=0.9):
        self.lr = lr
        self.β  = β

    def Step(self, mP, mDp, dState={}):
        mV            = dState.get('mV', np.zeros(mP.shape))
        mV            = self.β * mV - self.lr * mDp
        mP           += mV
        dState['mV']  = mV

        return mP, dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Adam:
    def __init__(self, lr=1e-3, β1=0.9, β2=0.99, ϵ=1e-8):
        self.lr    = lr
        self.β1 = β1
        self.β2 = β2
        self.ϵ  = ϵ

    def Step(self, mP, mDp, dState={}):
        mV            = dState.get('mV', np.zeros(mP.shape))
        mS            = dState.get('mS', np.zeros(mP.shape))
        ii            = dState.get('ii', 0) + 1

        mV            = self.β1 * mV + (1 - self.β1) * mDp
        mS            = self.β2 * mS + (1 - self.β2) * mDp * mDp

        mTildeV       = mV / (1 - self.β1**ii)
        mTildeS       = mS / (1 - self.β2**ii)

        mP           -= self.lr * mTildeV / (np.sqrt(mTildeS) + self.ϵ)
        dState['mV']  = mV
        dState['mS']  = mS
        dState['ii']  = ii

        return mP, dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class AdamW:
    def __init__(self, lr=1e-3, β1=0.9, β2=0.99, ϵ=1e-8, wd=0):
        self.lr    = lr
        self.β1 = β1
        self.β2 = β2
        self.ϵ   = ϵ
        self.wd    = wd #-- weight decay

    def Step(self, mW, mDw, dState={}):
        mV            = dState.get('mV', np.zeros(mW.shape))
        mS            = dState.get('mS', np.zeros(mW.shape))
        ii            = dState.get('ii', 0) + 1

        mV            = self.β1 * mV + (1 - self.β1) * mDw
        mS            = self.β2 * mS + (1 - self.β2) * mDw * mDw

        mTildeV       = mV / (1 - self.β1**ii)
        mTildeS       = mS / (1 - self.β2**ii)

        mW           -= self.lr * mTildeV / (np.sqrt(mTildeS) + self.ϵ) + self.wd * mW
        dState['mV']  = mV
        dState['mS']  = mS
        dState['ii']  = ii

        return mW, dState