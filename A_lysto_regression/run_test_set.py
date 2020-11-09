"""
Run inference on test set with a pretrained model
"""


# %%
import torch
from torch import nn
import torchvision
from torchsummary import summary
import skimage
from skimage import io 
import re
import os
import numpy as np
import pandas as pd
import time
import bisect
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from ranger import Ranger
from datetime import datetime
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from datasets import DataLysto, grade
from models import ResNet_ref_b
from metrics import compute_reg_metrics, compute_cls_metrics

# %%
DATA_PATH = "./"
checkpoint_path = Path('/home/papa/ly_decount/A_lysto_regression/experiments/2020-11-08T16:43:50.424335/last.pth') # set '' if you dont want to load checkpoints
print(f"Attempt loading checkpoint {checkpoint_path.name}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
test_ds = DataLysto(DATA_PATH+"test.h5", "test")
len(test_ds)

# %%
# declare the model to be used
model = ResNet_ref_b()
model.to(device)

# %%
if checkpoint_path:
  try:
    checkpoint = torch.load(str(checkpoint_path))
    model.load_state_dict(checkpoint["model_state"])
    epochs_trained = checkpoint["train_epochs"]+1
    print(f"checkpoint loaded succesfully, model already trained for {epochs_trained} epochs")
  except FileNotFoundError:
    print("warning: no checkpoint to load")

model.eval()
# %%
ref = [0,5,10,20,50,200, 202]
print("starting ")
res = []
with torch.no_grad():
  for i, elem in enumerate(test_ds):
    print(f"\r{i+1}/{len(test_ds)}", end="", flush=True)
    img = torch.tensor(elem).float().cuda()
    out_reg, out_cls = model(img.unsqueeze(0))
    cls_ = np.argmax(out_cls.cpu().numpy())
    reg = max(0, round(out_reg.item()))

    if grade(out_reg.item()) == cls_:
      res.append((i, reg))
    else:
      reg_ = ref[cls_]
      res.append((i, reg_))

len(res)


# %%
with open("test_results.csv", "w") as f:
  f.write("id,count\n")
  for elem in res:
    f.write(f"{elem[0]+1},{elem[1]}\n")

# %%
# save some images
img_idr = "./sample_pred_img"
with torch.no_grad():
  for i, elem in enumerate(test_ds):
    if i > 10:
      break
    img = torch.tensor(elem).float().cuda()
    out_reg, out_cls = model(img.unsqueeze(0))
    cls_ = np.argmax(out_cls.cpu().numpy())
    reg = max(0, round(out_reg.item()))

    if grade(out_reg.item()) == cls_:
      res = reg
    else:
      reg_ = ref[cls_]
      res = reg_

    plt.imshow(elem.permute(1,2,0))
    plt.title(f"n: {res}")
    plt.savefig(f"{img_idr}/{i}.png")

len(res)
# %%
