"""
Using
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
and
https://github.com/lessw2020/mish
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
from datasets import DataLysto
from models import ResNet_ref_b
from metrics import compute_reg_metrics, compute_cls_metrics

# %%
EPOCHS = 70
BATCH_SIZE = 32
DATA_PATH = "./"
FREEZE_BASE_RESNET = True
LR = 1e-3
checkpoint_path = Path('/home/papa/ly_decount/A_lysto_regression/resnet50_ref_b.pth') # set '' if you dont want to load checkpoints
diagnostic_run = False

print(f"Attempt loading checkpoint {checkpoint_path.name}")
print(f"Training with base model layer frozen: {FREEZE_BASE_RESNET}")

run_id = str(datetime.today().isoformat())
writer = SummaryWriter(log_dir="./tb_runs/" + run_id)
print(f"run id: {run_id}")

EXP_DIR = Path("./experiments")
os.makedirs(EXP_DIR / run_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
train_ds = DataLysto(DATA_PATH+"train.h5", "train")
valid_ds = DataLysto(DATA_PATH+"train.h5", "valid")
#test_ds = DataLysto(DATA_PATH+"test.h5", "test")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)
#test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
len(train_ds), len(valid_ds)#, len(test_ds)

# %%
# save a sample image from taining dataset
example = train_ds[0]
plt.imshow(example[0].permute(1,2,0))
plt.title("Target: " +  str(example[1]))
plt.savefig(EXP_DIR / run_id / "train_example")

# %%
# declare the model to be used
model = ResNet_ref_b()
model.to(device)

# freeze original resnet 50 for now
if FREEZE_BASE_RESNET:
  for param in model.base_modules.parameters():
    param.requires_grad = False 
   
# load paarams for optim in three groups for other training phases and ability to reload optim state

# first run of resnet_ref_b was without params groups so load a simple optim
optimizer = Ranger([param for param in model.parameters()])
# optimizer = Ranger([
#                   {'params':model.base_modules.parameters(), 'lr':LR},
#                   {'params':model.cnn_head.parameters(), 'lr':LR},
#                   {'params':model.reg_head.parameters(), 'lr':LR}
# ])
# lr scheduler
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, epochs=EPOCHS, steps_per_epoch=len(train_loader))

# losses for regression and classification
criterion_reg = torch.nn.SmoothL1Loss()
criterion_cls = torch.nn.CrossEntropyLoss()

# %%
if checkpoint_path:
  try:
    checkpoint = torch.load(str(checkpoint_path))
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])
    epochs_trained = checkpoint["train_epochs"]+1
    print(f"checkpoint loaded succesfully, model already trained for {epochs_trained} epochs")
  except FileNotFoundError:
    print("warning: no checkpoint to load")


# %%
# if model loaded start from last epoch, else start from 0
try:
  start_ep = epochs_trained
except:
  start_ep = 0

# %%
valid_loss = []
for epoch in range(start_ep , EPOCHS + start_ep):

    model.train()
    tmp_loss = 0
    for i, (images, targets, classes) in enumerate(train_loader):
        if diagnostic_run and i > 5:
          break

        its = 1 / tf if i > 0 else 0
        print(f"\rEpoch {epoch+1}/{EPOCHS+start_ep} ({i+1}/{len(train_loader)} {its:.2f} it/s) cudamem {torch.cuda.memory_reserved()/1e9:.1f} GiB", end="", flush=True)
        ti = time.time()
        images = images.to(device).float()
        targets = targets.to(device).float()
        classes = classes.to(device)

        out_reg, out_cls = model(images)
        loss_reg = criterion_reg(out_reg.squeeze(), targets)
        loss_cls = criterion_cls(out_cls, classes)
        loss = loss_reg + loss_cls

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()

        tmp_loss += loss.item()
        tf = time.time() - ti
        writer.add_scalar('Loss_reg/train', loss_reg.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss_cls/train', loss_cls.item(), epoch * len(train_loader) + i)
    
    
    print("\nValidation step...")
    # validation
    model.eval()
    with torch.no_grad():
        loss_val = 0.
        preds_reg, preds_cls = [], []
        trgts_reg, trgts_cls = [], []
        for j, (images, targets, classes) in enumerate(valid_loader):
          if diagnostic_run and j > 5:
            break
          its = 1 / tf if i > 0 else 0
          print(f"\rValidation: ({j+1}/{len(valid_loader)} {its:.2f} it/s) cudamem {torch.cuda.memory_reserved()/1e9:.1f} GiB", end="", flush=True)
          ti = time.time()
          images = images.to(device).float()
          targets = targets.to(device).float()
          classes = classes.to(device)
          out_reg, out_cls = model(images)
          loss_reg = criterion_reg(out_reg.squeeze(), targets).detach().item()
          loss_cls = criterion_cls(out_cls, classes).detach().item()
          loss_val += (loss_reg + loss_cls)
          #
          writer.add_scalar('Loss_reg/valid', loss_reg, epoch * len(valid_loader) + j)
          writer.add_scalar('Loss_cls/valid', loss_cls, epoch * len(valid_loader) + j)
          # save 4 metrics
          valid_loss.append(loss_val)
          preds_reg.extend(out_reg.detach().cpu().numpy())
          preds_cls.extend(torch.max(out_cls,dim=1)[1].cpu().numpy())
          trgts_reg.extend(targets.cpu().numpy())
          trgts_cls.extend(classes.cpu().numpy())
          tf = time.time() - ti
        # compute metrics
        preds_reg = np.array(preds_reg).squeeze()
        trgts_reg = np.array(trgts_reg)
        preds_cls = np.array(preds_cls).squeeze()
        trgts_cls = np.array(trgts_cls)

        cae, mae, mse = compute_reg_metrics(preds_reg, trgts_reg)
        qk, mcc = compute_cls_metrics(preds_cls, trgts_cls)

        writer.add_scalar('Metrics/abs_err', cae, epoch)
        writer.add_scalar('Metrics/mae', mae, epoch)
        writer.add_scalar('Metrics/mse', mse, epoch)
        writer.add_scalar("Metrics/QKappa", qk, epoch)
        writer.add_scalar("Metrics/MCC", mcc, epoch)

    #print(f"\nEpoch {epoch+1}/{EPOCHS}, train/valid: {train_loss[-1]:.3f}/{valid_loss[-1]:.3f}")

    # model backup
    checkpoint = dict(
      model_state=model.state_dict(),
      optim_state=optimizer.state_dict(),
      train_epochs = epoch
    )
    torch.save(checkpoint, EXP_DIR/run_id/"last.pth")
    if (epoch > 0) and (valid_loss[-1] < min(valid_loss)):
      torch.save(checkpoint, EXP_DIR/run_id/"best.pth")


#######
# END #
#######