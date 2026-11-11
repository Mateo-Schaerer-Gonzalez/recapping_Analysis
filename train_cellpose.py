"""Fine-tune Cellpose (cpsam_v2) on the corrected strips.

Run inside the `wax-cells` env:   python train_cellpose.py

Training data : every strips/train/<name>.png that has a corrected `<name>_seg.npy` (saved from the GUI).
Test data     : strips/test/*.png with a `_seg.npy` -- used only to report validation loss.
Images WITHOUT a `_seg.npy` are skipped (not reviewed => not trusted).
The model is saved to models/wax_cpsam; use it with  cd.segment(img, model='models/wax_cpsam').
"""
import glob
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np
from cellpose import io, models, train

N_EPOCHS = int(os.environ.get('N_EPOCHS', 100))
LR = 1e-5          # Cellpose default for fine-tuning
SAVE_AS = 'wax_cpsam'


def load_reviewed(paths):
    imgs, labs, names = [], [], []
    for png in paths:
        seg = png[:-4] + '_seg.npy'
        if not os.path.exists(seg):
            continue
        masks = np.load(seg, allow_pickle=True).item()['masks']
        if masks.max() == 0:
            continue
        imgs.append(cv2.imread(png)[..., ::-1].copy())           # RGB
        labs.append(masks.astype(np.uint16))
        names.append(png)
    return imgs, labs, names


def main():
    io.logger_setup()                 # without this Cellpose prints no training progress
    tr_paths = sorted(glob.glob('strips/train/*.png'))
    tr_i, tr_l, tr_n = load_reviewed(tr_paths)
    te_i, te_l, te_n = load_reviewed(sorted(glob.glob('strips/test/*.png')))
    print(f'train: {len(tr_i)} reviewed images, {sum(int(l.max()) for l in tr_l)} cells')
    print(f'test : {len(te_i)} reviewed images')
    if not tr_i:
        raise SystemExit('nothing to train on: correct some images in the GUI and save (Ctrl+S) first')

    model = models.CellposeModel(gpu=True)          # starts from cpsam_v2
    path, _, _ = train.train_seg(
        model.net, train_data=tr_i, train_labels=tr_l,
        test_data=te_i or None, test_labels=te_l or None,
        batch_size=1,                # small: 6 GB GPU
        n_epochs=N_EPOCHS, learning_rate=LR, weight_decay=0.1,
        save_path='.', model_name=SAVE_AS)
    print('saved:', path)


if __name__ == '__main__':
    main()
