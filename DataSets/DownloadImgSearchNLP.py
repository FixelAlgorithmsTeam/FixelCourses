import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import threading
from urllib.parse import urljoin, urlsplit

import imageio.v3 as iio
import requests
from bs4 import BeautifulSoup
import yaml


CATEGORIES = (
    ('Airplane', 'airplane'),
    ('Helicopter', 'helicopter'),
    ('Car', 'car automobile'),
    ('Motorcycle', 'motorcycle'),
    ('Bicycle', 'bicycle'),
    ('Train', 'train railway'),
    ('Bus', 'bus vehicle'),
    ('Sailboat', 'sailboat'),
    ('Hot Air Balloon', 'hot air balloon'),
    ('Tractor', 'tractor'),
    ('Dog', 'dog'),
    ('Cat', 'cat'),
    ('Horse', 'horse'),
    ('Elephant', 'elephant'),
    ('Dolphin', 'dolphin'),
    ('Lion', 'lion animal'),
    ('Butterfly', 'butterfly'),
    ('Penguin', 'penguin'),
    ('Giraffe', 'giraffe'),
    ('Sea Turtle', 'sea turtle'),
    ('Sunflower', 'sunflower'),
    ('Rose', 'rose flower'),
    ('Cactus', 'cactus'),
    ('Mushroom', 'mushroom'),
    ('Palm Tree', 'palm tree'),
    ('Pizza', 'pizza'),
    ('Hamburger', 'hamburger'),
    ('Sushi', 'sushi'),
    ('Ice Cream', 'ice cream'),
    ('Coffee', 'coffee cup'),
    ('Cake', 'cake'),
    ('Strawberry', 'strawberry'),
    ('Apple', 'apple fruit'),
    ('Bread', 'bread'),
    ('Pancakes', 'pancakes'),
    ('Mountain', 'mountain landscape'),
    ('Waterfall', 'waterfall'),
    ('Beach', 'beach'),
    ('Desert', 'desert landscape'),
    ('Forest', 'forest'),
    ('Lighthouse', 'lighthouse'),
    ('Castle', 'castle'),
    ('Bridge', 'bridge'),
    ('Windmill', 'windmill'),
    ('Harbor', 'harbor boats'),
    ('Guitar', 'guitar'),
    ('Violin', 'violin instrument'),
    ('Camera', 'vintage camera'),
    ('Book', 'open book'),
    ('Basketball', 'basketball'),
)

LICENSES = {
    4: ('CC BY 2.0', 'https://creativecommons.org/licenses/by/2.0/'),
    5: ('CC BY-SA 2.0', 'https://creativecommons.org/licenses/by-sa/2.0/'),
    9: ('CC0 1.0', 'https://creativecommons.org/publicdomain/zero/1.0/'),
    10: ('Public Domain Mark 1.0', 'https://creativecommons.org/publicdomain/mark/1.0/'),
}
TARGET_COUNT = 100
THREAD_STATE = threading.local()


class DownloadBlocked(RuntimeError):
    pass


def GetImageSession() -> requests.Session:
    if not hasattr(THREAD_STATE, 'session'):
        THREAD_STATE.session = requests.Session()
    return THREAD_STATE.session


def ParseSearchPage(pageText: str) -> list[dict]:
    oSoup = BeautifulSoup(pageText, 'html.parser')
    for oScript in oSoup.find_all('script'):
        scriptText = oScript.string or oScript.get_text()
        markerIdx = scriptText.find('modelExport:')
        if markerIdx < 0:
            continue
        jsonStart = scriptText.find('{', markerIdx)
        dExport, _ = json.JSONDecoder().raw_decode(scriptText[jsonStart:])
        lModels = dExport['main']['search-photos-lite-models']
        return [dPhoto['data'] for dPhoto in lModels[0]['data']['photos']['data']['_data']
                if isinstance(dPhoto, dict) and isinstance(dPhoto.get('data'), dict)]
    raise ValueError('Flickr search metadata was not found; no download attempted.')


def SearchPhotos(oSession: requests.Session, query: str, page: int = 1) -> list[dict]:
    oResponse = oSession.get(
        'https://www.flickr.com/search/',
        params={'text': query, 'license': '4,5,9,10', 'media': 'photos',
                'safe_search': 1, 'content_type': 1, 'sort': 'relevance', 'page': page},
        timeout=(10, 30),
    )
    if oResponse.status_code in (401, 403, 429):
        raise DownloadBlocked(f'Flickr search returned {oResponse.status_code}; stop and resume later. '
                              f'Retry-After: {oResponse.headers.get("Retry-After", "not specified")}')
    oResponse.raise_for_status()
    return ParseSearchPage(oResponse.text)


def PreparePhoto(dPhoto: dict, category: str, query: str) -> dict | None:
    licenseId = dPhoto.get('license')
    if licenseId not in LICENSES or not dPhoto.get('isPublic', False):
        return None
    dSize = dPhoto.get('sizes', {}).get('data', {}).get('w', {}).get('data', {})
    imageUrl = urljoin('https://www.flickr.com', dSize.get('url', ''))
    oUrl = urlsplit(imageUrl)
    fileName = Path(oUrl.path).name
    photoId = str(dPhoto.get('id', ''))
    ownerId = dPhoto.get('ownerNsid', '')
    width, height = dSize.get('width', 0), dSize.get('height', 0)
    if (oUrl.scheme != 'https' or oUrl.hostname != 'live.staticflickr.com'
            or not fileName.endswith('_w.jpg') or not photoId.isdigit()
            or not fileName.startswith(photoId + '_') or not ownerId
            or max(width, height) != 400 or min(width, height) < 64):
        return None
    licenseName, licenseUrl = LICENSES[licenseId]
    return {
        'category': category, 'query': query, 'photo_id': photoId,
        'filename': fileName, 'source_url': f'https://www.flickr.com/photos/{ownerId}/{photoId}',
        'image_url': imageUrl, 'title': dPhoto.get('title', ''),
        'author': dPhoto.get('realname') or dPhoto.get('username') or ownerId,
        'owner_id': ownerId, 'license_id': licenseId,
        'license': licenseName, 'license_url': licenseUrl,
        'width': width, 'height': height,
        'retrieved_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def DownloadPhoto(dPhoto: dict) -> tuple[dict, bytes | None, str | None]:
    try:
        oResponse = GetImageSession().get(dPhoto['image_url'], timeout=(10, 25))
        if oResponse.status_code == 429:
            raise DownloadBlocked('Flickr image delivery returned 429; stop and resume later.')
        oResponse.raise_for_status()
        imageBytes = oResponse.content
        if not imageBytes.startswith(b'\xff\xd8'):
            raise ValueError('The response is not a JPEG.')
        mImage = iio.imread(imageBytes, extension='.jpg')
        if mImage.shape[:2] != (dPhoto['height'], dPhoto['width']):
            raise ValueError('Image dimensions do not match Flickr Small 400 metadata.')
        if mImage.ndim not in (2, 3):
            raise ValueError('Unexpected image dimensions.')
        dPhoto = dict(dPhoto)
        dPhoto['sha256'] = hashlib.sha256(imageBytes).hexdigest()
        dPhoto['pixel_sha256'] = hashlib.sha256(str(mImage.shape).encode() + mImage.tobytes()).hexdigest()
        dPhoto['bytes'] = len(imageBytes)
        return dPhoto, imageBytes, None
    except (requests.RequestException, ValueError, OSError) as error:
        return dPhoto, None, f'{type(error).__name__}: {error}'


def MarkdownText(value: str) -> str:
    text = html.escape(' '.join(html.unescape(value).split()))
    for char in ('[', ']', '(', ')', '*', '_', '`'):
        text = text.replace(char, '\\' + char)
    return text


def SaveProgress(outputPath: Path, dState: dict) -> None:
    statePath = outputPath / 'DownloadState.json'
    tempPath = statePath.with_suffix('.json.tmp')
    tempPath.write_text(json.dumps(dState, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    tempPath.replace(statePath)
    lRecords = dState['images']
    lLines = [
        '# Images for Natural Language Visual Search', '',
        'A dataset by Royi Avital (https://github.com/RoyiAvital).',
        'Photos (Small 400) from [Flickr](https://www.flickr.com).', '',
        'Target: 50 categories with 100 images each (5,000 images).',
        f'Currently downloaded: {len(lRecords):,} images.', '',
        '## Collection Notes', '',
        '- Downloads use Flickr Small 400 (`_w.jpg`), with a 400-pixel longest edge.',
        '- Original downloaded bytes and Flickr aspect ratios are preserved. No cropping, resizing, or re-encoding is applied.',
        '- Categories come from Flickr search queries. They are weak labels, not individually verified annotations.',
        '- Duplicate Flickr photo IDs, identical files, and identical decoded pixels are excluded across categories.',
        '- At most 10 photos per author are included in each category.',
        '- Source pages, authors, and licenses are recorded from Flickr public search metadata at collection time.',
        '- Each photo retains its individual license. Attribution and ShareAlike obligations still apply.',
        '- Machine-readable provenance and hashes are in `DownloadState.json`.', '',
    ]
    captionPath = outputPath / 'Captions.yaml'
    if captionPath.exists():
        dCaptions = yaml.safe_load(captionPath.read_text(encoding='utf-8'))
        numCaptioned = len(dCaptions['images'])
        lLines.extend([
            '## Captions', '',
            f'Caption coverage: **{numCaptioned:,} / {len(lRecords):,} downloaded images**.', '',
            'The single root file [Captions.yaml](Captions.yaml) maps relative image paths to 2-3 captions.',
            'Captions are written by the assistant after direct inspection of previews resized by a factor of 0.5.',
            'No local captioning model is used. Preview generation does not modify the original images.',
            'Only inspected images have entries. Captioning is incomplete until every image has an entry.',
            'These are AI-written descriptions, not human-verified ground truth.', '',
        ])
    for category, query in CATEGORIES:
        lCategory = [record for record in lRecords if record['category'] == category]
        lLines.extend([f'## {category}', '', f'Search: `{query}`. Downloaded: {len(lCategory)} / {TARGET_COUNT}.', ''])
        for record in lCategory:
            title = MarkdownText(record['title'])
            author = MarkdownText(record['author'])
            lLines.append(f"{record['filename']} -> {record['source_url']} "
                          f"({title}; by {author}; [{record['license']}]({record['license_url']}))")
        lLines.append('')
    lLines.extend(['## License Notes', '',
                   'Only CC BY 2.0, CC BY-SA 2.0, CC0 1.0, and Public Domain Mark 1.0 photos are included.',
                   'License links are provided per image. This README does not relicense the photographs.', ''])
    readmePath = outputPath / 'README.md'
    readmeTemp = outputPath / 'README.md.tmp'
    readmeTemp.write_text('\n'.join(lLines), encoding='utf-8')
    readmeTemp.replace(readmePath)


def LoadProgress(outputPath: Path) -> dict:
    statePath = outputPath / 'DownloadState.json'
    if not statePath.exists():
        return {'version': 1, 'target_per_category': TARGET_COUNT, 'images': [], 'failures': []}
    dState = json.loads(statePath.read_text(encoding='utf-8'))
    for record in dState['images']:
        imagePath = outputPath / record['category'] / record['filename']
        if not imagePath.is_file():
            raise ValueError(f'Missing previously downloaded image: {imagePath}')
    return dState


def BuildCategory(oSession: requests.Session, oPool: ThreadPoolExecutor, outputPath: Path,
                  dState: dict, category: str, query: str, target: int, maxPages: int) -> None:
    categoryPath = outputPath / category
    categoryPath.mkdir(exist_ok=True)
    lExisting = [record for record in dState['images'] if record['category'] == category]
    numSaved = len(lExisting)
    if numSaved >= target:
        print(f'{category}: {numSaved}/{target} already present.', flush=True)
        return
    sPhotoIds = {record['photo_id'] for record in dState['images']}
    sFileHashes = {record['sha256'] for record in dState['images']}
    sPixelHashes = {record['pixel_sha256'] for record in dState['images']}
    dAuthorCounts = Counter(record['owner_id'] for record in lExisting)
    sSeen = set()
    for page in range(1, maxPages + 1):
        lPhotos = SearchPhotos(oSession, query, page)
        lCandidates = []
        dPendingAuthors = Counter()
        numNew = 0
        for dPhoto in lPhotos:
            record = PreparePhoto(dPhoto, category, query)
            if record is None or record['photo_id'] in sSeen:
                continue
            sSeen.add(record['photo_id'])
            numNew += 1
            if record['photo_id'] in sPhotoIds:
                continue
            ownerId = record['owner_id']
            if dAuthorCounts[ownerId] + dPendingAuthors[ownerId] >= 10:
                continue
            lCandidates.append(record)
            dPendingAuthors[ownerId] += 1
            if len(lCandidates) >= target - numSaved:
                break
        if page > 1 and numNew == 0:
            print(f'{category}: no new eligible photos on page {page}.', flush=True)
            break
        for record, imageBytes, error in oPool.map(DownloadPhoto, lCandidates):
            if error is not None or imageBytes is None:
                dState['failures'].append({'category': category, 'photo_id': record['photo_id'], 'error': error})
                continue
            if record['sha256'] in sFileHashes or record['pixel_sha256'] in sPixelHashes:
                continue
            imagePath = categoryPath / record['filename']
            if imagePath.exists():
                raise FileExistsError(f'Untracked file will not be overwritten: {imagePath}')
            with imagePath.open('xb') as hFile:
                hFile.write(imageBytes)
            dState['images'].append(record)
            sPhotoIds.add(record['photo_id'])
            sFileHashes.add(record['sha256'])
            sPixelHashes.add(record['pixel_sha256'])
            dAuthorCounts[record['owner_id']] += 1
            numSaved += 1
        SaveProgress(outputPath, dState)
        print(f'{category}: {numSaved}/{target} after page {page}.', flush=True)
        if numSaved >= target:
            return
    print(f'{category}: INCOMPLETE ({numSaved}/{target}).', flush=True)


def VerifyDataset(outputPath: Path, dState: dict) -> bool:
    dCounts = Counter(record['category'] for record in dState['images'])
    sIds, sHashes, sPixels, sPaths = set(), set(), set(), set()
    lProblems = []
    for category, _ in CATEGORIES:
        print(f'{category}: {dCounts[category]} / {TARGET_COUNT}')
        if dCounts[category] != TARGET_COUNT:
            lProblems.append(f'{category}: wrong image count')
    for record in dState['images']:
        imagePath = outputPath / record['category'] / record['filename']
        imageBytes = imagePath.read_bytes()
        mImage = iio.imread(imageBytes, extension='.jpg')
        fileHash = hashlib.sha256(imageBytes).hexdigest()
        pixelHash = hashlib.sha256(str(mImage.shape).encode() + mImage.tobytes()).hexdigest()
        if (fileHash != record['sha256'] or pixelHash != record['pixel_sha256']
                or mImage.shape[:2] != (record['height'], record['width'])
                or max(mImage.shape[:2]) != 400 or not imagePath.name.endswith('_w.jpg')
                or record['license_id'] not in LICENSES):
            lProblems.append(f'{imagePath}: invalid content or provenance')
        if record['photo_id'] in sIds or fileHash in sHashes or pixelHash in sPixels:
            lProblems.append(f'{imagePath}: duplicate')
        sIds.add(record['photo_id'])
        sHashes.add(fileHash)
        sPixels.add(pixelHash)
        sPaths.add(imagePath.resolve())
    sActual = {imagePath.resolve() for imagePath in outputPath.glob('*/*.jpg')}
    if sActual != sPaths:
        lProblems.append('Files and manifest do not match.')
    print(f'Total: {len(sPaths)} images; {sum(record["bytes"] for record in dState["images"]):,} bytes.')
    for problem in lProblems:
        print('FAIL:', problem)
    print('PASS' if not lProblems else 'INCOMPLETE OR INVALID')
    return not lProblems


def Main() -> None:
    oParser = argparse.ArgumentParser(description='Build the Flickr Small 400 image-search dataset.')
    oParser.add_argument('--inspect', action='store_true')
    oParser.add_argument('--verify', action='store_true')
    oParser.add_argument('--start', type=int, default=1)
    oParser.add_argument('--end', type=int, default=len(CATEGORIES))
    oParser.add_argument('--per-category', type=int, default=TARGET_COUNT)
    oParser.add_argument('--max-pages', type=int, default=30)
    oParser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'ImgSearchNLP')
    args = oParser.parse_args()
    assert len(CATEGORIES) == 50
    if args.inspect:
        with requests.Session() as oSession:
            lPhotos = SearchPhotos(oSession, CATEGORIES[0][1])
        print(f'Categories: {len(CATEGORIES)}; photos on search page: {len(lPhotos)}')
        print(json.dumps(lPhotos[0], indent=2, ensure_ascii=True))
        return
    if not (1 <= args.start <= args.end <= len(CATEGORIES) and 1 <= args.per_category <= TARGET_COUNT):
        oParser.error('Use category indices 1..50 and a per-category target 1..100.')
    args.output.mkdir(parents=True, exist_ok=True)
    dState = LoadProgress(args.output)
    if args.verify:
        raise SystemExit(0 if VerifyDataset(args.output, dState) else 1)
    for category, _ in CATEGORIES:
        (args.output / category).mkdir(exist_ok=True)
    SaveProgress(args.output, dState)
    try:
        with requests.Session() as oSession, ThreadPoolExecutor(max_workers=4) as oPool:
            for category, query in CATEGORIES[args.start - 1:args.end]:
                BuildCategory(oSession, oPool, args.output, dState, category, query, args.per_category, args.max_pages)
    except (DownloadBlocked, requests.RequestException, ValueError, OSError) as error:
        SaveProgress(args.output, dState)
        print(f'STOPPED: {type(error).__name__}: {error}', flush=True)
        raise SystemExit(2) from error
    SaveProgress(args.output, dState)
    dCounts = Counter(record['category'] for record in dState['images'])
    print(f'Total saved: {len(dState["images"])}; complete categories: '
          f'{sum(dCounts[category] >= TARGET_COUNT for category, _ in CATEGORIES)}/50.', flush=True)


if __name__ == '__main__':
    Main()