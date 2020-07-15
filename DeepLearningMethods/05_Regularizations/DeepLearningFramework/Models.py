import numpy             as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
class SequentialModel:
    def __init__(self, lLayers):
        self.lLayers = lLayers
        self.train   = False

    def Forward(self, mX):
        for oLayer in self.lLayers:
            if self.train == False and hasattr(oLayer, 'Predict'):
                mX = oLayer.Predicd(mX)
            else:
                mX = oLayer.Forward(mX)
        return mX

    def Backward(self, mDz):
        for oLayer in reversed(self.lLayers):
            mDz = oLayer.Backward(mDz)


