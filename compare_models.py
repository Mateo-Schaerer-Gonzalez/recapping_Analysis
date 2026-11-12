"""Compare the out-of-the-box Cellpose model with the fine-tuned ones on the held-out strips.

Run inside the `wax-cells` env:   python compare_models.py
Writes compare_<strip>.png (panels stacked: base, v1, v2) and prints cell counts.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np

import cell_detection as cd

MODELS = [('base', None), ('v1 (2 strips)', 'models/wax_cpsam_v1'), ('v2 (14 strips)', 'models/wax_cpsam')]
COLORS = [(0, 0, 255), (255, 128, 0), (0, 200, 0)]
STRIPS = ['strips/test/p0_s3.png', 'strips/test/p1_s3.png']


def main():
    for f in STRIPS:
        img = cv2.imread(f)
        panels = []
        for (name, path), col in zip(MODELS, COLORS):
            lab = cd.segment(img, model=path)
            im = cd.draw_contours(img, lab, color=col)
            t = f'{name}: {lab.max()} cells'
            cv2.putText(im, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
            cv2.putText(im, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            panels.append(im)
            print(f'{f}: {t}')
        out = f'compare_{os.path.basename(f)}'
        cv2.imwrite(out, np.vstack(panels))


if __name__ == '__main__':
    main()
