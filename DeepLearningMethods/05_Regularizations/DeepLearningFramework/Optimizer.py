import numpy             as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
#-- Basic gradient descent update rule:
class Sgd:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def Step(self, mW, mDw, dState={}):
        mW -= self.lr * mDw
        return mW, dState


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Sgdm:
    def __init__(self, lr=1e-3, beta=0.9):
        self.lr   = lr
        self.beta = beta

    def Step(self, mW, mDw, dState={}):
        mV            = dState.get('mV', np.zeros(mW.shape))
        mV            = self.beta * mV - self.lr * mDw
        mW           += mV
        dState['mV']  = mV

        return mW, dState

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps

    def Step(self, mW, mDw, dState={}):
        mV            = dState.get('mV', np.zeros(mW.shape))
        mS            = dState.get('mS', np.zeros(mW.shape))
        ii            = dState.get('ii', 0) + 1

        mV            = self.beta1 * mV + (1 - self.beta1) * mDw
        mS            = self.beta2 * mS + (1 - self.beta2) * mDw * mDw

        mTildeV       = mV / (1 - self.beta1**ii)
        mTildeS       = mS / (1 - self.beta2**ii)

        mW           -= self.lr * mTildeV / (np.sqrt(mTildeS) + self.eps)
        dState['mV']  = mV
        dState['mS']  = mS
        dState['ii']  = ii

        return mW, dState

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
            for sParam in oLayer.dGrads:
                #-- Get param, gradient and history:
                mW          = oLayer.dParams[sParam]
                mDw         = oLayer.dGrads[sParam]
                sLayerParam = str(ii) + sParam
                dState      = self.dStates.get(sLayerParam, {})

                #-- Apply step:
                mW, dState                = self.oUpdateRule.Step(mW, mDw, dState)

                #-- Set param and hisoty:
                oLayer.dParams[sParam]    = mW
                self.dStates[sLayerParam] = dState