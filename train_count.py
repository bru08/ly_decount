"""
Adapt a CNN to perform regression for counting lymphocites
# TODO add tensorboard log
# TODO add parameters to checkpoint file
"""
# %%
import torch
from torch import nn
import torchvision
from pathlib import Path
from torchsummary import summary
import skimage
from skimage import io 
import re
import os
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from datetime import datetime
import os
# %%
# hyperparameters

EPOCHS = 20
BATCH_SIZE = 40
DATA_PATH = Path(__file__).parent.resolve()
IMG_SIZE = 199
RESUME = True

os.makedirs("./checkpoints/", exist_ok=True)


writer = SummaryWriter(log_dir=f"./tb_runs/{datetime.today().isoformat()}")


class DataLysto(Dataset):

    def __init__(self, path, mode="train"):
      """
      Pytorch dataset wrapper for hdf5 file
      """
      super(DataLysto, self).__init__()

      self.ds = h5py.File(path, 'r')
      self.n_tot = self.ds["x"].shape[0]
      train_up_id = int(self.n_tot *0.75)
      self.mode = mode
      if mode == "train":
        self.low, self.up = 0, train_up_id
      elif mode == "valid":
        self.low, self.up = train_up_id, self.n_tot
      elif mode == "test":
        self.low, self.up = 0, self.n_tot
      else:
        raise ValueError

      self.elements_ids = np.arange(self.low, self.up)

    def __getitem__(self, idx):
      img = self.ds["x"][idx][13:-13,13:-13]
      img = self.transforms(img)
      if self.mode in ["train", "valid"]:
        trgt = self.ds["y"][idx]
        return img, trgt
      else:

        return img
    
    def transforms(self, img):
      trans_list = []
      

      if self.mode == "train":
        trans_list.extend([
          T.ToPILImage(),
          T.RandomHorizontalFlip(),
          T.RandomVerticalFlip(),
          T.ColorJitter()
        ])
      trans_list.append(T.ToTensor())
      transformer = T.Compose(trans_list)
      return transformer(img)

    def __len__(self):
      return len(self.elements_ids)

  

# %%
# datasets and dataloaders
train_ds = DataLysto(DATA_PATH/"training.h5", "train")
valid_ds = DataLysto(DATA_PATH/"training.h5", "valid")
# test_ds = DataLysto("test.h5", "test")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)

# %%

model = torchvision.models.densenet121(pretrained=True, progress=True)
in_features = model.classifier.in_features
model.classifier = torch.nn.Sequential(
  nn.Linear(in_features, 512),
  nn.ReLU(),
  nn.Linear(512, 1)
)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is running on {device}")

# %%
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.SmoothL1Loss()


# %%
if RESUME:
  print("Trying to load weights...")
  try:
    chkpt = torch.load("/home/papa/ly_decount/checkpoints/checkpoint_last.pth")
    model.load_state_dict(chkpt["model_state"])
    optimizer.load_state_dict(chkpt["optim_state"])
    print("Model and optim state loaded succesfully.")
  except FileNotFoundError:
    print("warning: no checkpoint to load")
#%%
ti = time.time()
_ = model(torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).cuda().float()).detach()
tf = time.time()
iter_time = tf-ti
eta = iter_time * len(train_loader) * EPOCHS
eta_h = eta / (60 * 60)
print(f"Estimated required time for training {EPOCHS} epochs: {eta_h:.1f} hours")

# %%

train_loss = []
valid_loss = []
val_metrics = {"abs_err": [], "mae": [], "mse": []}

for epoch in range(EPOCHS):

    model.train()
    tmp_loss = 0
    for i, (images, targets) in enumerate(train_loader):
        # if i > 5:
        #   break

        its = 1 / tf if i > 0 else 0
        print(f"\rEpoch {epoch+1}/{EPOCHS} ({i+1}/{len(train_loader)} {its:.2f} it/s) cudamem {torch.cuda.memory_reserved()/1e9:.1f} GiB", end="", flush=True)
        ti = time.time()
        images = images.to(device).float()
        targets = targets.to(device).float()

        out = model(images)
        loss = criterion(out.squeeze(), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tmp_loss += loss.item()
        tf = time.time() - ti
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
    train_loss.append(tmp_loss/len(train_loader))
    
    print("Validation step...")
    # validation
    model.eval()
    with torch.no_grad():
        loss_val = 0.
        preds = []
        trgts = []
        for j, (images, targets) in enumerate(valid_loader):
          # if j > 5:
          #   break
            its = 1 / tf if i > 0 else 0
            print(f"\rValidation: ({j+1}/{len(valid_loader)} {its:.2f} it/s) cudamem {torch.cuda.memory_reserved()/1e9:.1f} GiB", end="", flush=True)
            ti = time.time()
            images = images.to(device).float()
            targets = targets.to(device).float()
            out = model(images)
            loss = criterion(out.squeeze(), targets).detach().item()
            loss_val += loss
            writer.add_scalar('Loss/valid', loss, epoch * len(valid_loader) + j)
            preds.extend(out.detach().cpu().numpy())
            trgts.extend(targets.cpu().numpy())
            tf = time.time() - ti
          
        preds = np.array(preds).squeeze()
        trgts = np.array(trgts)
        valid_loss.append(loss_val/len(valid_loader))
        abs_err = np.abs(preds - trgts).sum()
        mean_abs_err = abs_err / len(preds)
        mse = ((preds - trgts) ** 2).mean()
        writer.add_scalar('Metrics/abs_err', abs_err, epoch)
        writer.add_scalar('Metrics/mae', mean_abs_err, epoch)
        writer.add_scalar('Metrics/mse', mse, epoch)

    print(f"\nEpoch {epoch+1}/{EPOCHS}, train/valid: {train_loss[-1]:.3f}/{valid_loss[-1]:.3f}")

    # model backup
    checkpoint = dict(
      model_state=model.state_dict(),
      optim_state=optimizer.state_dict()
    )
    torch.save(checkpoint, f"./checkpoints/checkpoint_last.pth")
    if (epoch > 0) and (valid_loss[-1] < min(valid_loss)):
      torch.save(checkpoint, f"./checkpoints/checkpoint_best.pth")


with open("results.csv", "w") as f:
    f.write("epoch,train,valid\n")
    for i, (tr, val) in enumerate(zip(train_loss, valid_loss)):
        f.write(f"{i+1},{tr},{val}\n")

