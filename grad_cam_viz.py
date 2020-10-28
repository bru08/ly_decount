"""
use grad cam algorithm with a pretrained model.
Visualize relevant area in inference task.
"""


# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

# %%
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
model.eval()
model(torch.tensor(img).float().unsqueeze(0).cuda())

img = valid_ds[2][0]
inp = torch.tensor(img).float().unsqueeze(0).cuda()
pred_value = model(inp)
img = img.transpose(1,2,0)
pred_value, train_ds[0][1]

plt.imshow(img)

pred_value.item(), valid_ds[2][1]

optimizer.zero_grad()
pred_value.backward()

# pull the gradients out of the model
gradients = model.get_activations_gradient()
print(gradients.shape)

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = model.get_activations(inp).detach()

# weight the channels by corresponding gradients
print(activations.shape)
print(pooled_gradients.shape)

for i in range(len(pooled_gradients)):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze().cpu()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())



from skimage import transform
heatmap = transform.resize(heatmap, (256, 256))
heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
plt.figure(figsize=(6,6))
plt.imshow(heatmap)
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.imshow(heatmap, alpha=0.7)







