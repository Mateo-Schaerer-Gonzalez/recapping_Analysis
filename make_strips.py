"""Cut the scan pages into their 16 strips and write pre-labels for correcting in the Cellpose GUI.

Run inside the `wax-cells` env:   python make_strips.py

Writes   strips/train/<name>.png + <name>_masks.tif     (to label and train on)
         strips/test/<name>.png  + <name>_masks.tif     (HELD OUT: label them, never train on them)

Pre-labels = plain out-of-the-box Cellpose predictions (no filtering).  Re-running OVERWRITES the
_masks.tif files (move any `_seg.npy` you want to keep out of the way first).
"""
import glob
import os

import cv2
import numpy as np
from tifffile import imwrite

import cell_detection as cd

PDF = '20261109145816_SecureScan.pdf'

# page -> list of strip boxes (x0, y0, x1, y1) in 200-dpi pixels, top to bottom (s1..s4)
STRIPS = {
    0: [(210, 120, 1270, 630), (200, 680, 1250, 1180), (240, 1170, 1300, 1680), (280, 1870, 1330, 2330)],
    1: [(250, 20, 1330, 480), (260, 630, 1310, 1080), (240, 1140, 1290, 1580), (240, 1750, 1310, 2190)],
    2: [(120, 60, 1280, 540), (160, 740, 1240, 1200), (100, 1260, 1180, 1740), (90, 1780, 1160, 2190)],
    3: [(260, 110, 1400, 580), (240, 740, 1320, 1190), (320, 1250, 1360, 1700), (310, 1760, 1360, 2220)],
}
# held out for testing: one dark strip, one light strip
TEST = {(0, 3), (1, 3)}        # (page, strip number 1-4)
CROP_DIR = 'training'


def strip_name(page, s):
    return f'p{page}_s{s}'


def in_test_strip(page, cx, cy):
    """True if a point (full-page px) lies inside a held-out strip."""
    for (p, s) in TEST:
        if p == page:
            x0, y0, x1, y1 = STRIPS[p][s - 1]
            if x0 <= cx < x1 and y0 <= cy < y1:
                return True
    return False


def main():
    for split in ('train', 'test'):
        os.makedirs(os.path.join('strips', split), exist_ok=True)
    for page, boxes in STRIPS.items():
        print(f'page {page}: running Cellpose ...', flush=True)
        img = cd.load_page(PDF, page=page, dpi=200)
        lab = cd.segment(img).astype(np.int32)
        for s, (x0, y0, x1, y1) in enumerate(boxes, 1):
            split = 'test' if (page, s) in TEST else 'train'
            name = strip_name(page, s)
            path = os.path.join('strips', split, name)
            y1, x1 = min(y1, img.shape[0]), min(x1, img.shape[1])
            l = lab[y0:y1, x0:x1]
            ids = np.unique(l[l > 0])
            l = np.where(l > 0, np.searchsorted(ids, l) + 1, 0).astype(np.uint16)
            cv2.imwrite(path + '.png', img[y0:y1, x0:x1])
            imwrite(path + '_masks.tif', l)
            print(f'  {split:5s} {name}  {len(ids)} pre-labelled cells')


if __name__ == '__main__':
    main()
