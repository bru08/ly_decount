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
import argparse
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from metrics import compute_cls_metrics, compute_reg_metrics
from datasets import DMapData
from utils import load_lysto_weights


# %%
ENCODER_ARCH = "resnet50"
PRETRAIN = "lysto" # or lysto - imagenet only for resnet50
lysto_checkpt_path = "/home/papa/ly_decount/A_lysto_regression/experiments/2020-11-08T16:43:50_resnetrefb_30ep_freeze_5ep_difflr/last.pth"
FREEZE_ENCODER = True
checkpoint_path = ""
DATASET_DIR = "/home/riccardi/neuroblastoma_project_countCD3/try_yolo_ultralytics/dataset_nb_yolo_trail" 
LR = 1e-3
EPOCHS = 120
BATCH_SIZE = 16
timestamp = str(datetime.today().isoformat())
TRIAL_RUN = False

# select gpu to run the process
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu-id', default='0', type=str, metavar='N')
parser.add_argument('-e', '--encoder-architecture', default="resnet50", type=str)
parser.add_argument('-p', '--weights', default='imagenet', type=str)
parser.add_argument('-r', '--resume', default='', type=str)
parser.add_argument('-f', '--freeze-encoder', default=False, type=bool)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id
ENCODER_ARCH = args.encoder_architecture
PRETRAIN = args.weights
FREEZE_ENCODER = args.freeze_encoder

exp_title = f"dens_count_{ENCODER_ARCH}_{PRETRAIN}_ep_{EPOCHS}_bs_{BATCH_SIZE}_{'resume' if args.resume else 'scratch'}_{timestamp}"
writer = SummaryWriter(log_dir=f"./tb_runs/{exp_title}")
EXP_DIR = f"./experiments/{exp_title}/"
os.makedirs(EXP_DIR)

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
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)
# %%
if (not args.resume) and (PRETRAIN == "imagenet"):
    model = smp.Unet(ENCODER_ARCH, encoder_weights=PRETRAIN, decoder_attention_type="scse")
    print("starting training with iamgenet weights")
else:
    model = smp.Unet(ENCODER_ARCH, decoder_attention_type="scse")

if PRETRAIN == "lysto":
    load_lysto_weights(model, lysto_checkpt_path, "resnet50")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if FREEZE_ENCODER:
    for param in model.encoder.parameters():
        param.requires_grad = False


optimizer = torch.optim.Adam([
                  {'params': model.encoder.parameters(), 'lr':LR*1e-3},
                  {'params': model.decoder.parameters(), 'lr':LR},
                  {'params': model.segmentation_head.parameters(), 'lr':LR}
])
segm_criterion = torch.nn.MSELoss()
prob_cons_criterion = torch.nn.L1Loss()

# %%
# warmp up 
start_ep = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"loaded checkpoint {args.resume}")
    start_ep = checkpoint["epochs"]
else:
    print("Warm up...")
    for i, (img, mask) in enumerate(loader_train):
        print(f"\rWarming up {i+1}/{len(loader_train)}", end="", flush=True)
        img = img.to(device).float()
        mask = mask.to(device).unsqueeze(1).float()
        out = model(img)
        segm_loss = segm_criterion(out, mask)
        optimizer.zero_grad()
        segm_loss.backward()
        optimizer.step()

print("\nStarting training..")
# %%
losses_tr = dict(segment=[], conserv=[])
losses_val = dict(segment=[], conserv=[])
best_cons_loss = np.inf
for epoch in range(start_ep, EPOCHS + start_ep):
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
            f"\rEpoch {epoch + 1}/{EPOCHS+start_ep} ({i+1}/{len(loader_train)}) loss:{loss.item():.4f}|segm_loss:{segm_loss.item():.2f} |cons_loss: {conservation_loss.item():.2f}",
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
        reg_met = dict(pred=[],trgt=[])

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


            counts_pred = out.sum(dim=[1,2,3]).detach().cpu().numpy() / 100.
            counts_gt = mask.sum(dim=[1,2,3]).detach().cpu().numpy() / 100.
            reg_met["pred"].extend(counts_pred)
            reg_met["trgt"].extend(counts_gt)

            # store losses
            losses_val["segment"].append(segm_loss.item())
            losses_val["conserv"].append(conservation_loss.item())
            writer.add_scalar("segmentation_loss/Valid", segm_loss.item(), epoch * len(loader_valid) + j)
            writer.add_scalar("regression_loss/Valid", conservation_loss.item(), epoch * len(loader_valid) + j)


            # log 
            print(
                f"\rValid epoch {epoch + 1}/{EPOCHS + start_ep} ({j+1}/{len(loader_valid)}) loss:{loss.item():.4f}|segm_loss:{segm_loss.item():.2f} |cons_loss: {conservation_loss.item():.2f}",
                end="", flush=True
                )
        cae, mae, mse = compute_reg_metrics(reg_met)
        qk, mcc, acc = compute_cls_metrics(reg_met)
        writer.add_scalar("metrics/cae", cae, epoch)
        writer.add_scalar("metrics/mae", mae, epoch)
        writer.add_scalar("metrics/mse", mse, epoch)
        writer.add_scalar("metrics/qkappa", qk, epoch)
        writer.add_scalar("metrics/mcc", mcc, epoch)
        writer.add_scalar("metrics/accuracy",acc,epoch)

    


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

