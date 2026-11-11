"""Segment honeycomb cells in scans of wax strips with Cellpose and outline them.

    import cell_detection as cd
    img = cd.load_page('20261109145816_SecureScan.pdf', page=0, dpi=200)   # BGR
    labels = cd.segment(img)                    # int image, 0 = background, 1..N = cells
    out = cd.draw_contours(img, labels)

Run inside the `wax-cells` conda env.  segment(img, model='models/wax_cpsam') uses a fine-tuned model.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # torch + conda both ship OpenMP on Windows

import cv2
import numpy as np

_MODELS = {}


def load_page(pdf_path, page=0, dpi=200):
    import pymupdf
    p = pymupdf.open(pdf_path)[page].get_pixmap(dpi=dpi)
    return np.frombuffer(p.samples, np.uint8).reshape(p.h, p.w, 3)[:, :, ::-1].copy()


def _model(gpu=True, model=None):
    """model=None -> default pretrained 'cpsam_v2' (cached in ~/.cellpose/models); else a path to a fine-tuned model."""
    if (model, gpu) not in _MODELS:
        from cellpose import models
        kw = {'pretrained_model': model} if model else {}
        _MODELS[(model, gpu)] = models.CellposeModel(gpu=gpu, **kw)
    return _MODELS[(model, gpu)]


def segment(bgr, gpu=True, model=None):
    """Plain Cellpose prediction (default parameters, no filtering). Returns an int label image."""
    return _model(gpu, model).eval(bgr[..., ::-1].copy())[0]       # Cellpose expects RGB


def draw_contours(bgr, labels, color=(0, 200, 0), thickness=2):
    """Outline each labelled cell with its exact shape."""
    from scipy import ndimage as ndi
    out = bgr.copy()
    for i, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        y0, x0 = max(sl[0].start - 1, 0), max(sl[1].start - 1, 0)
        sub = (labels[y0:sl[0].stop + 1, x0:sl[1].stop + 1] == i).astype(np.uint8)
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
        cv2.drawContours(out, cs, -1, color, thickness, cv2.LINE_AA)
    return out
