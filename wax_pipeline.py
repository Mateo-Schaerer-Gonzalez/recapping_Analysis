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


# ---------------------------------------------------------------- Colab entry point
def collect_pdfs(uploader=None, in_dir='/content/input', drive_dir='/content/drive/MyDrive/wax_scans'):
    """Save the PDFs picked in the ipywidgets FileUpload `uploader` to in_dir and return their paths.
    Falls back to PDFs in /content, then in the Drive folder drive_dir."""
    import glob
    os.makedirs(in_dir, exist_ok=True)
    val = uploader.value if uploader is not None else ()
    # ipywidgets 7: {name: {content,...}}   ipywidgets 8: ({name, content, ...}, ...)
    items = list(val.items()) if isinstance(val, dict) else [(d['name'], d) for d in val]
    for name, d in items:
        with open(os.path.join(in_dir, name), 'wb') as f:
            f.write(bytes(d['content']))
    pdfs = sorted(glob.glob(os.path.join(in_dir, '*.pdf'))) or sorted(glob.glob('/content/*.pdf'))
    if not pdfs:
        # widget value did not arrive (common in Colab): use Colab's native upload dialog
        try:
            from google.colab import files
            print('Widget upload not received - choose the PDF(s) in the dialog below:')
            for name, content in files.upload().items():
                if name.lower().endswith('.pdf'):
                    with open(os.path.join(in_dir, name), 'wb') as f:
                        f.write(content)
            pdfs = sorted(glob.glob(os.path.join(in_dir, '*.pdf')))
        except Exception as e:
            print('native upload failed:', e)
    if not pdfs:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
        pdfs = sorted(glob.glob(os.path.join(drive_dir, '*.pdf')))
    print(len(pdfs), 'PDF(s):', [os.path.basename(p) for p in pdfs])
    assert pdfs, 'No PDFs: click Upload above and wait for the file name to appear (or put PDFs in %s)' % drive_dir
    return pdfs


def colab_main(pdfs, model_path='/content/drive/MyDrive/wax_cpsam', out_root='/content/results'):
    """Everything after the upload, for Google Colab: model from Drive -> pipeline on each PDF -> show -> zip -> download."""
    import shutil
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - enable a GPU runtime')

    model = None
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
        model = model_path if os.path.exists(model_path) else None
    except Exception:
        pass
    print('model:', model or 'generic cpsam_v2 (fine-tuned model not found at %s!)' % model_path)

    def show(name, overlay):
        plt.figure(figsize=(12, 6)); plt.imshow(overlay[..., ::-1]); plt.axis('off'); plt.title(name); plt.show()

    results = []
    for pdf in pdfs:
        stem = os.path.splitext(os.path.basename(pdf))[0]
        df = run_pdf(pdf, out_dir=os.path.join(out_root, stem), model=model, show=show)
        df.insert(0, 'file', stem)
        results.append(df)
    counts = pd.concat(results, ignore_index=True)
    counts.to_csv(os.path.join(out_root, 'all_counts.csv'), index=False)

    zip_path = shutil.make_archive(out_root, 'zip', out_root)
    try:
        from google.colab import files
        shutil.copy(zip_path, '/content/drive/MyDrive/wax_results.zip')      # backup in Drive
        files.download(zip_path)
    except Exception as e:
        print('zip saved at', zip_path, '-', e)
    return counts


if __name__ == '__main__':
    import sys
    pdf = sys.argv[1]
    mdl = sys.argv[2] if len(sys.argv) > 2 else None
    print(run_pdf(pdf, out_dir=os.path.join('results', os.path.splitext(os.path.basename(pdf))[0]), model=mdl))
