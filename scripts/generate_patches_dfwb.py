import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx

src = 'datasets/'
tar = 'datasets/DFWB'
os.makedirs(tar, exist_ok=True)

patch_size = 768
overlap = 128
p_max = 1200


def save_files(file_):
    path_contents = file_.split(os.sep)
    foldname = path_contents[-2]
    filename = os.path.splitext(path_contents[-1])[0]
    img = cv2.imread(file_)
    num_patch = 0
    w, h = img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                patch = img[i:i+patch_size, j:j+patch_size,:]
                savename = os.path.join(tar, foldname + '-' + filename + '-' + str(num_patch) + '.png')
                cv2.imwrite(savename, patch)

    else:
        savename = os.path.join(tar, foldname + '-' + filename + '.png')
        cv2.imwrite(savename, img)


files = []
for dataset in ['DF2K/DF2K_train_HR', 'WED/pristine_images', 'BSDS500/data/images/train', 'BSDS500/data/images/val', 'BSDS500/data/images/test']:
    df = natsorted(glob(os.path.join(src, dataset, '*.png')) + glob(os.path.join(src, dataset, '*.jpg')) + glob(os.path.join(src, dataset, '*.bmp')))
    files.extend(df)
    
from joblib import Parallel, delayed
import multiprocessing
num_cores = 10
Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(files))