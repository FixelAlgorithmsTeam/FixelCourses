
# General
import numpy as np

# Models
from ultralytics import YOLO

# Typing
from typing import Dict, List, Tuple



lImgSize    = [640, 640]
lRows       = [0, 440]
lCols       = [0, 320, 640, 960, 1280]


class BallDetector():
    def __init__(self, modelFilePath, confThr: float = 0.25, lImgSize: List[int] = lImgSize, lRows: List[int] = lRows, lCols: List[int] = lCols) -> None:

        modelYolo = YOLO(modelFilePath)

        self.modelYolo  = modelYolo
        self.confThr    = confThr
        self.lImgSize   = lImgSize
        self.lRows      = lRows
        self.lCols      = lCols
    
    def forward(self, mImg: np.ndarray):
        """
        mImg - RGB image in UINT8 with dimensions of 1920x1080x3.
        """

        if (mImg.shape[0] != 1080) or (mImg.shape[1] != 1920) or (mImg.shape[2] != 3):
            raise ValueError(f'The input image dimensions are {mImg.shape} instead of 1080x1920x3')

        modelYolo   = self.modelYolo
        confThr     = self.confThr

        lImgSize    = self.lImgSize
        lRows       = self.lRows
        lCols       = self.lCols

        mI = mImg[:, :, ::-1] #<! RGB -> BGR
        mT = np.zeros((*lImgSize, 3), dtype = np.uint8)

        vBoxCoord   = np.zeros(4)
        confLvl     = 0
        maxII       = 0
        maxJJ       = 0

        # TODO: Replace with itertools.product()
        # TODO: We can have higher confidence if we check intersections between tiles
        # TODO: Fast mode can be break on first detection above a threshold
        for ii, firstRowIdx in enumerate(lRows):
            lastRowIdx = firstRowIdx + lImgSize[0]
            for jj, firstColdIdx in enumerate(lCols):
                lastColIdx = firstColdIdx + lImgSize[1]
                mT[:, :, :] = mI[firstRowIdx:lastRowIdx, firstColdIdx:lastColIdx, :]

                lModelResults = modelYolo(mT, conf = confThr, verbose = False, show = False, save = False) #<! List (Each image as element)
                modelResults  = lModelResults[0] #<! Working on a single image!

                # Per Tile
                # We take the result of the tile with maximum confidence level
                for modelResult in modelResults:
                    modelResult = modelResult.cpu().numpy()
                    if((len(modelResult) > 0) and (modelResult.boxes.conf[0] > confLvl)):
                        confLvl = modelResult.boxes.conf[0]
                        vBoxCoord[:] = modelResult.boxes.xyxy[:]
                        maxII = ii
                        maxJJ = jj
        
        vBoxCoord[0] += lCols[maxJJ]
        vBoxCoord[1] += lRows[maxII]
        vBoxCoord[2] += lCols[maxJJ]
        vBoxCoord[3] += lRows[maxII]

        return vBoxCoord, confLvl #<! [topLeftCol, topLeftRow, bottomRightCol, bottomRightRow], [boxScore]
