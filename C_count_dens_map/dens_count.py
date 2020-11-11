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
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

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
PRETRAIN = "lysto" # or lysto - imagenet only for resnet50
lysto_checkpt_path = "/home/papa/ly_decount/A_lysto_regression/experiments/2020-11-08T16:43:50_resnetrefb_30ep_freeze_5ep_difflr/last.pth"
FREEZE_ENCODER = True
DATASET_DIR = "/home/riccardi/neuroblastoma_project_countCD3/try_yolo_ultralytics/dataset_nb_yolo_trail" 
LR = 1e-3
EPOCHS = 120
BATCH_SIZE = 16
timestamp = str(datetime.today().isoformat())
TRIAL_RUN = False

exp_title = f"dens_count_{ENCODER_ARCH}_{PRETRAIN}_ep_{EPOCHS}_bs_{BATCH_SIZE}_{timestamp}"
writer = SummaryWriter(log_dir=f"./tb_runs/{exp_title}")
EXP_DIR = f"./experiments/{exp_title}/"
os.makedirs(EXP_DIR)

# %%
class DMapData(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, mode, transformations, point_label_init=100, lbl_sigma_gauss=1):
        super().__init__()
        tmp_path = Path(dataset_dir) / mode
        self.pt_lbl_init = point_label_init
        self.lbl_sigma_gauss = lbl_sigma_gauss
        self.img_files = [tmp_path/x for x in os.listdir(tmp_path) if (tmp_path/x).suffix == ".png"]
        self.transforms = transformations

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_files[idx]).convert("RGB"))
        
        annot_file = re.sub(r".png$", r".txt", str(self.img_files[idx]))
        with open(annot_file, "r") as f:
            annots = f.read().split("\n")
        annots = [
                [
                    int(float(x) *  (img.shape[0]-1))
                    for x in y.split(",")[1:3]
                    ] # take only xy of center in relative coord
            for y in annots if y
            ]
        mask_lbl = self.generate_label_mask(img, annots)

        transformed = self.transforms(image=img, mask=mask_lbl)

        img = transformed["image"] / 255.
        mask = transformed["mask"]

        return img, mask

    def generate_label_mask(self, img, annotations):
        mask = np.zeros(img.shape[:2])
        s = self.lbl_sigma_gauss
        for annot in annotations:

            mask[annot[1], annot[0]] = self.pt_lbl_init
        mask_blur = gaussian_filter(mask, (s,s), order=0)
        mask_blur /= mask_blur.sum()
        mask_blur *= (100 * len(annotations))
        return mask_blur
        
    def show_example(self, with_mask=True):
        img, lbl = self.__getitem__(1)
        fig = plt.figure(figsize=(6,6))
        plt.imshow(img.permute(1,2,0).numpy())
        if with_mask:
            plt.imshow(lbl.numpy(), alpha=0.5)
        plt.title(f"Sum: {lbl.sum():.2f}")
        plt.show()

    def __len__(self):
        return len(self.img_files)




# %%
# define albumentations transpose
# execution is first to last
transforms = A.Compose([
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, always_apply=False, p=0.99),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
])
dataset_train = DMapData(DATASET_DIR, "train", transforms, lbl_sigma_gauss=6)
dataset_valid = DMapData(DATASET_DIR, "valid", transforms, lbl_sigma_gauss=6)
# dataset_train.show_example(False)
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)
# %%
if PRETRAIN == "imagenet":
    model = smp.Unet(ENCODER_ARCH, encoder_weights=PRETRAIN, decoder_attention_type="scse")
else:
    model = smp.Unet(ENCODER_ARCH, decoder_attention_type="scse")

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
# warmp up 
print("Warm up...")
for i, (img, mask) in enumerate(loader_train):
    img = img.to(device).float()
    mask = mask.to(device).unsqueeze(1).float()
    out = model(img)
    segm_loss = segm_criterion(out, mask)
    optimizer.zero_grad()
    segm_loss.backward()
    optimizer.step()

print("Starting training..")
# %%
losses_tr = dict(segment=[], conserv=[])
losses_val = dict(segment=[], conserv=[])
best_cons_loss = np.inf
for epoch in range(EPOCHS):
    if TRIAL_RUN and epoch >3:
        break

    model.train()
    print("Training step..")
    for i, (img, mask) in enumerate(loader_train):
        if TRIAL_RUN and i>3:
            break
        # move data to device
        img = img.to(device).float()
        mask = mask.to(device).unsqueeze(1).float()

        # run inference
        out = model(img)
        # compute losses
        segm_loss = segm_criterion(out, mask)
        conservation_loss = prob_cons_criterion(out.sum(dim=[1,2,3]), mask.sum(dim=[1,2,3]))
        # losses aggregation
        loss = segm_loss #+ conservation_loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log 
        print(
            f"\rEpoch {epoch + 1}/{EPOCHS} ({i+1}/{len(loader_train)}) loss:{loss.item():.4f}|segm_loss:{segm_loss.item():.2f} |cons_loss: {conservation_loss.item():.2f}",
            end="", flush=True
            )
        # store losses
        losses_tr["segment"].append(segm_loss.item())
        losses_tr["conserv"].append(conservation_loss.item())
        writer.add_scalar("segmentation_loss/Train", segm_loss.item(), epoch * len(loader_train) + i)
        writer.add_scalar("regression_loss/Train", conservation_loss.item(), epoch * len(loader_train) + i)

    print("\nValidation step...")
    model.eval()
    with torch.no_grad():
        for j, (img, mask) in enumerate(loader_valid):
            if TRIAL_RUN and j>3:
                break
            # move data to device
            img = img.to(device).float()
            mask = mask.to(device).unsqueeze(1).float()

            # run inference
            out = model(img)
            # compute losses
            segm_loss = segm_criterion(out, mask)
            conservation_loss = prob_cons_criterion(out.sum(dim=[1,2,3]), mask.sum(dim=[1,2,3]))

    
            # store losses
            losses_val["segment"].append(segm_loss.item())
            losses_val["conserv"].append(conservation_loss.item())
            writer.add_scalar("segmentation_loss/Valid", segm_loss.item(), epoch * len(loader_valid) + j)
            writer.add_scalar("regression_loss/Valid", conservation_loss.item(), epoch * len(loader_valid) + j)
            # log 
            
            print(
                f"\rValid epoch {epoch + 1}/{EPOCHS} ({j+1}/{len(loader_valid)}) loss:{loss.item():.4f}|segm_loss:{segm_loss.item():.2f} |cons_loss: {conservation_loss.item():.2f}",
                end="", flush=True
                )
    
    # save checkpoint
    last_checkpoint = {
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "losses_tr":losses_tr,
        "losses_val":losses_val,
        "epochs":epoch+1
    }

    avg_val_loss = np.mean(losses_val["conserv"][-len(loader_valid):])
    if avg_val_loss < best_cons_loss:
        best_cons_loss = avg_val_loss
        name = "best"
    else:
        name = "last"

    torch.save(last_checkpoint, EXP_DIR + name + ".pth")

