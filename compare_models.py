"""Compare the out-of-the-box Cellpose model with the fine-tuned one on the held-out strips.

Run inside the `wax-cells` env:   python compare_models.py
Writes compare_<strip>.png (left = base model, right = fine-tuned) and prints cell counts.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np

import cell_detection as cd

MODEL = 'models/wax_cpsam'
STRIPS = ['strips/test/p0_s3.png', 'strips/test/p1_s3.png']


def main():
    for f in STRIPS:
        img = cv2.imread(f)
        base = cd.segment(img)
        tuned = cd.segment(img, model=MODEL)
        a = cd.draw_contours(img, base, color=(0, 0, 255))
        b = cd.draw_contours(img, tuned, color=(0, 200, 0))
        for im, t in ((a, f'base: {base.max()} cells'), (b, f'fine-tuned: {tuned.max()} cells')):
            cv2.putText(im, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
            cv2.putText(im, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        out = f'compare_{os.path.basename(f)}'
        cv2.imwrite(out, np.vstack([a, b]))
        print(f'{f}: base {base.max()} cells, fine-tuned {tuned.max()} cells -> {out}')


if __name__ == '__main__':
    main()
