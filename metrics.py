# %%
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import h5py

# %%
model = torchvision.models.densenet121(pretrained=True, progress=True)
in_features = model.classifier.in_features
model.classifier = torch.nn.Sequential(
  nn.Linear(in_features, 512),
  nn.ReLU(),
  nn.Linear(512, 1)
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# %%
print("Trying to load weights...")
try:
  model.load_state_dict(torch.load("checkpoint_0"))
except FileNotFoundError:
  print("warning: no checkpoint to load")
# %%
summary(model, (3,199,199))
# %%

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
        idx = self.elements_ids[idx]
        img = self.ds["x"][idx].transpose(2,0,1)
        if self.mode in ["train", "valid"]:
            trgt = self.ds["y"][idx]
            return img, trgt
        else:
            return img

    def __len__(self):
      return len(self.elements_ids)

  

# %%
# datasets and dataloaders
train_ds = DataLysto("training.h5", "train")
valid_ds = DataLysto("training.h5", "valid")
val_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

# %%
# take prediction and targets from validation set
model.eval()
preds, targets = [], []
with torch.no_grad():
  for i, (img, trgt) in enumerate(val_loader):
    print(f"\r{i+1}/{len(val_loader)}", end="", flush=True)
    img = torch.tensor(img).float().cuda()
    out = model(img)
    preds.extend(out.cpu().numpy())
    targets.extend(trgt.cpu().numpy())

# %%
preds, targets = np.array(preds).squeeze(), np.array(targets)
err = preds - targets
abs_err = np.abs(preds - targets)

# %%
plt.hist(abs_err, bins=60)
plt.show()

# %%
plt.hist(err, bins=40)
plt.show()
# %%
err.min(), err.max()

# %%
plt.boxplot(err)
# %%
print(f"Absolute error {abs_err.mean():.2f}, mean squared error: {(abs_err**2).mean()}")

ex_id = 3
img, y = train_ds[ex_id]
pred = model(torch.tensor(img).cuda().float().unsqueeze(0)).item()
plt.imshow(img.transpose(1,2,0))
plt.title(f"Target: {y} Prediction: {pred:.2f}")


# %%



# %%
# run inference on test set
valid_ds = DataLysto("test.h5", "test")
model.eval()
res = []
with torch.no_grad():
  for i, elem in enumerate(test_ds):
    print(f"{i+1}/{len(test_ds)}")

    img = torch.tensor(elem).float().cuda()
    out = model(img.unsqueeze(0)).item()
    res.append((i, out))

with open("test_res.csv", "w+") as f:
  f.write("id,count\n")
  for elem in res:
    f.write(f"{elem[0]+1},{round(elem[1])}\n")

#%%


tr = [0.78, 0.61, 0.57, 0.516, 0.468, 0.44, 0.4, 0.38, 0.34, 0.33, 0.31, 0.31, 0.28, 0.27, 0.25, 0.25, 0.23, 0.22, 0.20]
plt.plot(np.arange(len(tr))+1, tr)
plt.title("training loss (huber)")
plt.xlabel("epochs")
# %%
