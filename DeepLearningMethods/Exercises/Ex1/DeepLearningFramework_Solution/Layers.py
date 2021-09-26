import numpy as np

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class LinearLayer:
    def __init__(self, dIn, dOut, init='Kaiming'):
        if   init == 'Kaiming': mW = np.random.randn(dOut, dIn) * np.sqrt(2 / dIn)
        elif init == 'Xavier':  mW = np.random.randn(dOut, dIn) * np.sqrt(1 / dIn)
        else:                   mW = np.random.randn(dOut, dIn) / dIn #-- this is how we initialize previously

        vB = np.zeros(dOut)

        self.dParams = {'mW' : mW,   'vB': vB}
        self.dGrads  = {'mW' : None, 'vB' : None}

    def Forward(self, mX):
        mW      = self.dParams['mW']
        vB      = self.dParams['vB']
        self.mX = mX                   #-- store for Backward
        mZ      = mW @ mX + vB[:,None]

        return mZ

    def Backward(self, mDz):
        mW  = self.dParams['mW']
        mX  = self.mX

        vDb = mDz.sum(1)
        mDw = mDz  @ mX.T
        mDx = mW.T @ mDz

        self.dGrads['vB'] = vDb
        self.dGrads['mW'] = mDw

        return mDx
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class ReLULayer:
    def __init__(self):
        self.dParams = {}
        self.dGrads  = {}

    def Forward(self, mX):
        self.mX = mX                 #-- store for Backward
        mZ      = np.maximum(mX, 0)

        return mZ

    def Backward(self, mDz):
        mX    = self.mX
        mMask = (mX > 0).astype(np.float32)
        mDx   = mDz * mMask

        return mDx
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class LeakyReLULayer:
    def __init__(self, slope=0.01):
        self.dParams = {}
        self.dGrads  = {}
        self.slope   = slope

    def Forward(self, mX):
        self.mX = mX #-- store for Backward
        mZ      = np.maximum(mX, 0) + np.minimum(self.slope * mX, 0)

        return mZ

    def Backward(self, mDz):
        mX    = self.mX
        mMask = (mX > 0) + self.slope * (mX < 0)
        mDx   = mDz * mMask

        return mDx
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class DropoutLayer:
    def __init__(self, p=0.5):
        self.dParams = {}
        self.dGrads  = {}
        self.p       = p

    #-- Train time:
    def Forward(self, mX):
        self.mMask = (np.random.rand(*mX.shape) < self.p) / self.p
        mZ         = mX * self.mMask

        return mZ

    #-- Test time:
    def Predict(self, mX):
        return mX

    def Backward(self, mDz):
        mDx   = mDz * self.mMask

        return mDx
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#