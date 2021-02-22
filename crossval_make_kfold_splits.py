"""
Perform a small version of dap
1. Take dataset train/test split
2. Take train data
3. k-fold split function (k=5)
4. train on k-1 validate on 1 each one of the five time
5. make meaningful names to the various runs of the models
"""
# %%
from pathlib import Path
import json
import torch
import torchvision
import albumentations as A 
import numpy as np
import os

from datasets import DMapData
# Initialization

# %%
# just the train data as specified in DAP MAQC etc..
DATA_DIR = Path("/home/papa/ly_decount/dataset_nb_yolo_v2/train")
K = 5
IT = 4
seed = 1234

# %%
files = (x for x in os.listdir(DATA_DIR) if (DATA_DIR / x).suffix == ".png")
slide_ids = [int(x.split("_")[0]) for x in files]
slides_set, counts = np.unique(slide_ids, return_counts=True)

np.random.seed(seed)
fold_group_size = len(slides_set) // K
folds_groups = {}
for i in range(IT):
    perm = np.random.permutation(slides_set)
    tmp = {i//fold_group_size: [str(x) for x in perm[i:i+fold_group_size]] for i in range(0, len(perm), fold_group_size)}
    folds_groups[i] = tmp
folds_groups

with open("k_fold_splits.json", "w") as f:
    json.dump(folds_groups, f, indent=4)
