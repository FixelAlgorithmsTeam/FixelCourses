
# General
import numpy as np

# Models
from ultralytics import YOLO

# Typing
from typing import Dict, List, Tuple
from numpy.typing import NDArray



tuTileSize = [640, 640]
tuRows     = [0, 440]
tuCols     = [0, 320, 640, 960, 1280]


class TiledDetector():
    def __init__(self, modelFilePath: str, confThr: float = 0.25, tuTileSize: Tuple[int, int] = tuTileSize, tuRows: Tuple[int, int] = tuRows, tuCols: Tuple[int, int] = tuCols) -> None:
        """
        Initializes the TiledDetector with a YOLO model and parameters for tiled inference.
        Inputs:
         - modelFilePath - Path to the YOLO model file.
         - confThr       - Confidence threshold for the detection.
         - tuTileSize    - Size of the tile to be used for inference.
         - tuRows        - Rows indices to start the tile from.
         - tuCols        - Columns indices to start the tile from.
        """

        oModelYolo = YOLO(modelFilePath)
        numCls     = oModelYolo.model.nc #<! Number of classes
        dCls       = oModelYolo.model.names #<! Class dictionary (`isx: name`)

        self._oModelYolo  = oModelYolo
        self._numCls      = numCls
        self._dCls        = dCls
        self._confThr     = confThr
        self._tuTileSize  = tuTileSize
        self._tuRows      = tuRows
        self._tuCols      = tuCols
    
    def Predict(self, mImg: NDArray) -> Tuple[NDArray, NDArray]:
        """
        mImg - RGB image in UINT8 with dimensions of 1920x1080x3.
        """

        if (mImg.shape[0] != 1080) or (mImg.shape[1] != 1920) or (mImg.shape[2] != 3):
            raise ValueError(f'The input image dimensions are {mImg.shape} instead of 1080x1920x3')

        modelYolo = self._oModelYolo
        confThr   = self._confThr
        numCls    = self._numCls

        tuImgSize = self._tuTileSize
        tuRows    = self._tuRows
        tuCols    = self._tuCols

        mI = mImg[:, :, ::-1] #<! RGB -> BGR
        mT = np.zeros((*tuImgSize, 3), dtype = np.uint8)

        # Per class: keep only the highest confidence detection across all tiles.
        mBoxCoord = np.full((numCls, 4), np.nan)
        vConfLvl  = np.zeros(numCls)

        # TODO: We can have higher confidence if we check intersections between tiles
        # TODO: Fast mode can be break on first detection above a threshold
        for ii, firstRowIdx in enumerate(tuRows):
            lastRowIdx = firstRowIdx + tuImgSize[0]
            for jj, firstColdIdx in enumerate(tuCols):
                lastColIdx = firstColdIdx + tuImgSize[1]
                mT[:, :, :] = mI[firstRowIdx:lastRowIdx, firstColdIdx:lastColIdx, :]

                lModelResults = modelYolo(mT, conf = confThr, verbose = False, show = False, save = False) #<! List (Each image as element)
                modelResults  = lModelResults[0] #<! Working on a single image!

                boxes = getattr(modelResults, 'boxes', None)
                if (boxes is None) or (len(boxes) == 0):
                    continue

                # Convert once per tile for efficiency.
                xyxy = boxes.xyxy.cpu().numpy() #<! (N, 4)
                conf = boxes.conf.cpu().numpy() #<! (N, )
                cls  = boxes.cls.cpu().numpy()  #<! (N, )

                # Update per class maxima.
                for kk in range(xyxy.shape[0]):
                    clsIdx = int(cls[kk])
                    if (clsIdx < 0) or (clsIdx >= numCls):
                        continue
                    if conf[kk] > vConfLvl[clsIdx]:
                        vConfLvl[clsIdx]  = conf[kk]
                        mBoxCoord[clsIdx] = xyxy[kk]

                        # Convert tile coords -> full image coords
                        mBoxCoord[clsIdx, 0] += firstColdIdx
                        mBoxCoord[clsIdx, 1] += firstRowIdx
                        mBoxCoord[clsIdx, 2] += firstColdIdx
                        mBoxCoord[clsIdx, 3] += firstRowIdx

        return mBoxCoord, vConfLvl #<! Per class: [topLeftCol, topLeftRow, bottomRightCol, bottomRightRow], [boxScore]
    
    def __call__(self, mImg: NDArray) -> Tuple[NDArray, NDArray]:
        
        return self.Predict(mImg)
    
    def GetLabelName(self, clsIdx: int) -> str:
        
        return self._dCls[clsIdx]