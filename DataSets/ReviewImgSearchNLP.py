import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import imageio.v3 as iio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import yaml


def Main() -> None:
    oParser = argparse.ArgumentParser(description='Prepare half-size image previews for direct visual captioning.')
    oParser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parent / 'ImgSearchNLP')
    oParser.add_argument('--start', type=int, default=0)
    oParser.add_argument('--count', type=int, default=25)
    oParser.add_argument('--verify', action='store_true')
    oParser.add_argument('--allow-partial', action='store_true')
    args = oParser.parse_args()
    lRecords = json.loads((args.dataset / 'DownloadState.json').read_text(encoding='utf-8'))['images']
    if args.verify:
        dCaptions = yaml.safe_load((args.dataset / 'Captions.yaml').read_text(encoding='utf-8'))
        dImages = dCaptions['images']
        sExpected = {f'{record["category"]}/{record["filename"]}' for record in lRecords}
        lProblems = []
        if not set(dImages).issubset(sExpected):
            lProblems.append('Some caption paths do not exist in the download manifest.')
        if not args.allow_partial and set(dImages) != sExpected:
            lProblems.append(f'Coverage: {len(dImages)} captions entries for {len(sExpected)} downloaded images.')
        for imagePath, lText in dImages.items():
            if not (args.dataset / imagePath).is_file():
                lProblems.append(f'{imagePath}: image file is missing.')
            if (not isinstance(lText, list) or not 2 <= len(lText) <= 3
                    or any(not isinstance(text, str) or not text.strip() for text in lText)
                    or len({text.casefold().strip() for text in lText}) != len(lText)):
                lProblems.append(f'{imagePath}: requires 2-3 distinct nonempty captions.')
        for problem in lProblems[:25]:
            print('FAIL:', problem)
        print(f'Captioned images: {len(dImages)}; downloaded images: {len(lRecords)}; problems: {len(lProblems)}')
        raise SystemExit(1 if lProblems else 0)
    if args.start < 0 or not 1 <= args.count <= 25:
        oParser.error('start must be nonnegative and count must be in 1..25.')
    lBatch = lRecords[args.start:args.start + args.count]
    if not lBatch:
        oParser.error('No downloaded images in this range.')
    numCols = 5
    numRows = (len(lBatch) + numCols - 1) // numCols
    tileWidth, tileHeight = 210, 230
    tCanvas = np.full((numRows * tileHeight, numCols * tileWidth, 3), 255, dtype=np.uint8)
    for batchIdx, record in enumerate(lBatch):
        imagePath = args.dataset / record['category'] / record['filename']
        imageBytes = imagePath.read_bytes()
        if hashlib.sha256(imageBytes).hexdigest() != record['sha256']:
            raise ValueError(f'Source image changed: {imagePath}')
        tImage = iio.imread(imageBytes, extension='.jpg')
        if tImage.ndim == 2:
            tImage = np.repeat(tImage[:, :, None], 3, axis=2)
        tPreview = zoom(tImage, (0.5, 0.5, 1.0), order=1, prefilter=False)
        if max(tPreview.shape[:2]) != 200:
            raise ValueError(f'Unexpected half-size preview: {imagePath}')
        rowIdx, colIdx = divmod(batchIdx, numCols)
        top = rowIdx * tileHeight + 26 + (200 - tPreview.shape[0]) // 2
        left = colIdx * tileWidth + (tileWidth - tPreview.shape[1]) // 2
        tCanvas[top:top + tPreview.shape[0], left:left + tPreview.shape[1]] = tPreview
        print(f'{args.start + batchIdx + 1:05d}: {record["category"]}/{record["filename"]}')
    hF = plt.figure(figsize=(tCanvas.shape[1] / 100, tCanvas.shape[0] / 100), dpi=100)
    hA = hF.add_axes((0, 0, 1, 1))
    hA.imshow(tCanvas, interpolation='nearest')
    hA.axis('off')
    for batchIdx in range(len(lBatch)):
        rowIdx, colIdx = divmod(batchIdx, numCols)
        hA.text(colIdx * tileWidth + tileWidth / 2, rowIdx * tileHeight + 15,
                f'{args.start + batchIdx + 1:05d}', ha='center', va='center', fontsize=12, weight='bold')
    previewDir = Path(tempfile.gettempdir()) / 'FixelImgSearchNLPPreviews'
    previewDir.mkdir(exist_ok=True)
    previewPath = previewDir / f'Preview{args.start:05d}.png'
    hF.savefig(previewPath, dpi=100)
    plt.close(hF)
    print(f'PREVIEW: {previewPath}')
    print('Sources unchanged; every image was reduced by factor 0.5 without cropping.')


if __name__ == '__main__':
    Main()