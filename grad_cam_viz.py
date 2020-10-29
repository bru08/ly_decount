"""
use grad cam algorithm with a pretrained model.
Visualize relevant area in inference task.
"""

# %%
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import h5py
from skimage import transform

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
model = torchvision.models.densenet121(pretrained=True, progress=True)
in_features = model.classifier.in_features
model.classifier = torch.nn.Sequential(
  nn.Linear(in_features, 512),
  nn.ReLU(),
  nn.Linear(512, 1)
)



# %%
print("Trying to load weights...")
try:
  model.load_state_dict(torch.load("/home/papa/ly_decount/checkpoints/checkpoint_last.pth")["model_state"])
  print("Model weights laoded succesfully")
except FileNotFoundError:
  print("warning: no checkpoint to load")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# %% 
# register forward hook to get activations
hook_res = {}
def get_activation(name):
    def hook(model, input, output):
        hook_res[name] = output.detach()
    return hook
def get_grads(name):
    def hook(model, input, output):
        hook_res[name] = output[0]
    return hook

model.features.denseblock4.denselayer16.relu2.register_forward_hook(get_activation('activ'))
model.features.denseblock4.denselayer16.relu2.register_backward_hook(get_grads('grad'))


# %%
# probe with single image

img, t = train_ds[135]
img_plot = img.transpose(1,2,0)
img_tens = torch.tensor(img).float().cuda().unsqueeze(0)

# %%
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
optim.zero_grad()
# %%
# process input
model.eval()
pred = model(img_tens)
pred_vav = pred.item()
# backward from prediction
pred.backward()



#%%
print(f'activation shape: {hook_res["activ"].shape}')
print(f'gradients shape: {hook_res["grad"].shape}')

# %%
# average gradients in each channel
pooled_gradients = torch.mean(hook_res["grad"], dim=[0,2,3])
# weight activation by channel gradient weight
for i in range(len(pooled_gradients)):
    hook_res["activ"][:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(hook_res["activ"], dim=1).squeeze().cpu().numpy()
# %%
# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)
# normalize the heatmap
heatmap /= np.max(heatmap)
# draw the heatmap
plt.imshow(heatmap.squeeze())

# %%
# adapt heatmatp to input image
heatmap = transform.resize(heatmap, (199, 199))
heatmap = np.uint8(255 * heatmap)
# %%
plt.figure(figsize=(6,6))
plt.title(f"Pred n lymph: {pred_vav:.1f} vs gt:{t}")
plt.imshow(img_plot)
plt.imshow(heatmap, alpha=0.6)


# %%

# %%
