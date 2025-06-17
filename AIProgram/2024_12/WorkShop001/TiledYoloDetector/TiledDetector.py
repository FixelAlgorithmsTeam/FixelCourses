
# General
import numpy as np

# Models
from ultralytics import YOLO

# Typing
from typing import Dict, List, Tuple



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

        modelYolo = YOLO(modelFilePath)

        self.modelYolo  = modelYolo
        self.confThr    = confThr
        self.tuTileSize = tuTileSize
        self.tuRows     = tuRows
        self.tuCols     = tuCols
    
    def forward(self, mImg: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        mImg - RGB image in UINT8 with dimensions of 1920x1080x3.
        """

        if (mImg.shape[0] != 1080) or (mImg.shape[1] != 1920) or (mImg.shape[2] != 3):
            raise ValueError(f'The input image dimensions are {mImg.shape} instead of 1080x1920x3')

        modelYolo = self.modelYolo
        confThr   = self.confThr

        tuImgSize = self.tuTileSize
        tuRows    = self.tuRows
        tuCols    = self.tuCols

        mI = mImg[:, :, ::-1] #<! RGB -> BGR
        mT = np.zeros((*tuImgSize, 3), dtype = np.uint8)

        vBoxCoord = np.zeros(4)
        confLvl   = 0
        maxII     = 0
        maxJJ     = 0

        # TODO: Replace with itertools.product()
        # TODO: We can have higher confidence if we check intersections between tiles
        # TODO: Fast mode can be break on first detection above a threshold
        for ii, firstRowIdx in enumerate(tuRows):
            lastRowIdx = firstRowIdx + tuImgSize[0]
            for jj, firstColdIdx in enumerate(tuCols):
                lastColIdx = firstColdIdx + tuImgSize[1]
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
        
        # Calculate the coordinates on the input image (1920x1080)
        vBoxCoord[0] += tuCols[maxJJ]
        vBoxCoord[1] += tuRows[maxII]
        vBoxCoord[2] += tuCols[maxJJ]
        vBoxCoord[3] += tuRows[maxII]

        return vBoxCoord, confLvl #<! [topLeftCol, topLeftRow, bottomRightCol, bottomRightRow], [boxScore]
    
    def __call__(self, elf, mImg: np.ndarray) -> Tuple[np.ndarray, float]:
        
        return self.forward(mImg)
