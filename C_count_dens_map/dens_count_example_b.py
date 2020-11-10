"""
Frame the task as the prediction of the lymphocytes density map

* Use Unet Segmentation paradigm
* try more sophisticated encoder structure with pretrained weights from
    https://github.com/qubvel/segmentation_models.pytorch
* Density map based lables are built in the following way:
    * Input: point-like annotations
    * Create mask with same width and height as image
    * assign value t to pixel corresponding to point annotations
    * Convolve with gaussian kernel with dev = s

Values for t and s are hyperparamters.
"""
# %%
import torch
from torch import nn
import time
import re
import torchvision
import segmentation_models_pytorch as smp
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
# %%

def load_lysto_weights(model, state_path):
    assert ENCODER_ARCH == "resnet50"
    state = torch.load(state_path)["model_state"]
    #
    state_keys = [x for x in state.keys() if x.startswith("base_modules")]
    model_keys = list(model.encoder.state_dict().keys())
    #
    state_renamed = OrderedDict()
    for i, skey in enumerate(state_keys[1:]):
        try:
            state_renamed[model_keys[i+1]] = state[skey]
        except Exception as e:
            print(e)
    # needed for toch segmentation models that by defualt assume resnets have fc layers at the end
    state_renamed["fc.bias"] = 1
    state_renamed["fc.weight"] = 1
    #
    model.encoder.load_state_dict(state_renamed, strict=False)
    print("Succesfully converted and loaded resnet50 weights from lysto pretraining")

# %%
ENCODER_ARCH = "resnet50"
PRETRAIN = "imagenet" # or lysto only for resnet50
lysto_checkpt_path = "/home/papa/ly_decount/A_lysto_regression/experiments/2020-11-08T16:43:50_resnetrefb_30ep_freeze_5ep_difflr/last.pth"
FREEZE_ENCODER = True
DATASET_DIR = "/home/riccardi/neuroblastoma_project_countCD3/try_yolo_ultralytics/dataset_nb_yolo_trail" 
LR = 1e-3
EPOCHS = 10

# %%
class DMapData(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, mode, point_label_init=100, lbl_sigma_gauss=1):
        super().__init__()
        tmp_path = Path(dataset_dir) / mode
        self.pt_lbl_init = point_label_init
        self.lbl_sigma_gauss = lbl_sigma_gauss
        self.img_files = [tmp_path/x for x in os.listdir(tmp_path) if (tmp_path/x).suffix == ".png"]

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_files[idx]).convert("RGB")) / 255.
        
        annot_file = re.sub(r".png$", r".txt", str(self.img_files[idx]))
        with open(annot_file, "r") as f:
            annots = f.read().split("\n")
        annots = [
                [
                    round(float(x) *  img.shape[0])
                    for x in y.split(",")[1:3]
                    ] # take only xy of center in relative coord
            for y in annots if y
            ]
        mask_lbl = self.generate_label_mask(img, annots)

        img = torchvision.transforms.ToTensor()(img)
        mask_lbl = torchvision.transforms.ToTensor()(mask_lbl)

        return img, mask_lbl



    def generate_label_mask(self, img, annotations):
        mask = np.zeros(img.shape[:2])
        s = self.lbl_sigma_gauss
        for annot in annotations:
            mask[annot[1], annot[0]] = self.pt_lbl_init
        mask_blur = gaussian_filter(mask, (s,s), order=0)
        mask_blur /= mask_blur.sum()
        mask_blur *= (100 * len(annotations))
        return mask_blur
        
    def show_example(self):
        img, lbl = self.__getitem__(1)
        fig = plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.imshow(lbl, alpha=0.5)
        plt.title(f"Sum: {lbl.sum():.2f}")
        plt.show()

    def __len__(self):
        return len(self.img_files)

dataset_train = DMapData(DATASET_DIR, "train", lbl_sigma_gauss=6)
# dataset_train.show_example()
loader_train = DataLoader(dataset_train, batch_size=3, shuffle=True)

# %%
model = smp.Unet()
if PRETRAIN == "imagenet":
    model = smp.Unet(ENCODER_ARCH, encoder_weights=PRETRAIN)
else:
    model = smp.Unet(ENCODER_ARCH)

if PRETRAIN == "lysto":
    load_lysto_weights(model, lysto_checkpt_path)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if FREEZE_ENCODER:
    for param in model.encoder.parameters():
        param.requires_grad = False


optimizer = torch.optim.Adam([
                  {'params': model.encoder.parameters(), 'lr':LR},
                  {'params': model.decoder.parameters(), 'lr':LR},
                  {'params': model.segmentation_head.parameters(), 'lr':LR}
])
segm_criterion = torch.nn.MSELoss()
prob_cons_criterion = torch.nn.L1Loss()

# %%

for epoch in range(EPOCHS):

    model.train()
    for i, (img, mask) in enumerate(loader_train):
        # move data to device
        img = img.to(device).float()
        mask = mask.to(device).float()

        # run inference
        out = model(img)
        print(out.shape)
        # compute losses
        segm_loss = segm_criterion(out, mask)

        conservation_loss = prob_cons_criterion(out.sum(dim=[1,2,3]), mask.sum(dim=[1,2,3]))
        # losses aggregation
        loss = segm_loss + conservation_loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log 
        print(
            f"\rEpoch {epoch + 1}/{EPOCHS} loss:{loss.item():.2f}|segm_loss:{segm_loss.item():.2f} |cons_loss: {conservation_loss.item():.2f}",
            end="", flush=True
            )


# %%
