"""End-to-end pipeline: scan PDF -> strips -> cell segmentation -> cell counts.

    import wax_pipeline as wp
    df = wp.run_pdf('scan.pdf', out_dir='results/scan', model='models/wax_cpsam')

Writes into out_dir:
    strips/<page>_<strip>.png          cropped strip image
    masks/<page>_<strip>_masks.tif     label image (0 = background, 1..N = cells)
    overlays/<page>_<strip>.png        strip with every cell outlined
    counts.csv                         one row per strip: page, strip, n_cells, mean/median cell area (px)
Strips are found automatically (no hardcoded coordinates), top to bottom on each page.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

import cell_detection as cd


# ---------------------------------------------------------------- strip detection
def _runs(profile, thr, min_len, smooth):
    on = uniform_filter1d(profile.astype(float), smooth) > thr
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def find_strips(bgr, pad=30, scale=0.25):
    """Locate the wax strips (brown honeycomb) on a page; returns [(x0, y0, x1, y1)] top to bottom.

    Strips are assumed stacked vertically: brown pixels are projected on the rows to split the bands, then on the
    columns to get each band's width.  Pen marks / paper stains are ignored because only orange-brown saturated
    pixels count.
    """
    sm = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] > 90) & (hsv[..., 0] >= 3) & (hsv[..., 0] <= 25) & (hsv[..., 2] > 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    rows = m.mean(1)
    if rows.max() == 0:
        return []
    H, W = bgr.shape[:2]
    boxes = []
    for y0, y1 in _runs(rows, 0.12 * rows.max(), 40, 5):
        cols = m[y0:y1].mean(0)
        xs = _runs(cols, 0.15 * cols.max(), 40, 5)
        if not xs:
            continue
        x0, x1 = xs[0][0], xs[-1][1]
        boxes.append((max(int(x0 / scale) - pad, 0), max(int(y0 / scale) - pad, 0),
                      min(int(x1 / scale) + pad, W), min(int(y1 / scale) + pad, H)))
    return boxes


# ---------------------------------------------------------------- pipeline
def load_pages(path, dpi=200):
    """Yield (page_number, BGR image) from a PDF or a single image file."""
    if path.lower().endswith('.pdf'):
        import pymupdf
        for i in range(len(pymupdf.open(path))):
            yield i, cd.load_page(path, page=i, dpi=dpi)
    else:
        yield 0, cv2.imread(path)


def run_pdf(path, out_dir='results', model=None, dpi=200, gpu=True, show=None):
    """Full pipeline on one PDF/image.  Returns a pandas DataFrame with one row per strip.

    show: optional callback(name, overlay_bgr) called per strip (used by the notebook to display results).
    """
    import pandas as pd
    from tifffile import imwrite
    for d in ('strips', 'masks', 'overlays'):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    rows = []
    for page, img in load_pages(path, dpi):
        boxes = find_strips(img)
        print(f'page {page}: {len(boxes)} strips found', flush=True)
        for s, (x0, y0, x1, y1) in enumerate(boxes, 1):
            name = f'p{page}_s{s}'
            crop = img[y0:y1, x0:x1]
            lab = cd.segment(crop, gpu=gpu, model=model)
            ids = np.unique(lab[lab > 0])
            lab = np.where(lab > 0, np.searchsorted(ids, lab) + 1, 0).astype(np.uint16)     # relabel 1..N
            areas = np.bincount(lab.ravel())[1:]
            overlay = cd.draw_contours(crop, lab)
            t = f'{name}: {len(ids)} cells'
            cv2.putText(overlay, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
            cv2.putText(overlay, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(out_dir, 'strips', name + '.png'), crop)
            cv2.imwrite(os.path.join(out_dir, 'overlays', name + '.png'), overlay)
            imwrite(os.path.join(out_dir, 'masks', name + '_masks.tif'), lab)
            rows.append(dict(page=page, strip=s, n_cells=len(ids),
                             mean_area_px=float(areas.mean()) if len(areas) else 0.0,
                             median_area_px=float(np.median(areas)) if len(areas) else 0.0,
                             box=f'{x0},{y0},{x1},{y1}'))
            print(f'  {t}', flush=True)
            if show:
                show(name, overlay)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'counts.csv'), index=False)
    return df


if __name__ == '__main__':
    import sys
    pdf = sys.argv[1]
    mdl = sys.argv[2] if len(sys.argv) > 2 else None
    print(run_pdf(pdf, out_dir=os.path.join('results', os.path.splitext(os.path.basename(pdf))[0]), model=mdl))
