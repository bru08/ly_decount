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
import json
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
from utils import load_lysto_weights, compute_det_metrics

from mish.ranger import Ranger
from mish.rangerlars import RangerLars



# %%
# ------------
# A - SETTINGS
# ------------

# fixed settings
DATASET_OPBG_V1 = "/home/papa/ly_decount/dataset_nb_yolo_v1" 
DATASET_OPBG_V2 = "/home/papa/ly_decount/dataset_nb_yolo_v2"
BASE_DIR = Path(__file__).parent.resolve()
SAVE_DIR = Path("/datadisk/neuroblastoma_checkpoints_backup/experiments_dens_map_batch3")
lysto_checkpt_path = "/home/papa/ly_decount/A_lysto_regression/experiments/2020-11-08T16:43:50_resnetrefb_30ep_freeze_5ep_difflr/last.pth"

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu-id',               default='0', type=str, metavar='N')
parser.add_argument('-en', '--encoder-architecture',default="resnet50", type=str, help="encoder architecture param for pytorch_segmentation_models")
parser.add_argument('-ep', '--epochs',              default=120, type=int)
parser.add_argument('-p', '--weights',              default='imagenet', type=str)
parser.add_argument('-r', '--resume',               default='', type=str, help="path to checkpoint to use to resume trainig")
parser.add_argument('-f', '--freeze-encoder',       default=False, type=bool)
parser.add_argument('-s', '--label-sigma',          default=1, type=float, help="gauss kernel sigma applied to label mask")
#parser.add_argument('-d', '--label-value',          default=100, type=float, help="initialization value for pixel with annotation")
parser.add_argument('-bs', '--batch-size',          default=16, type=int)
parser.add_argument('-dr', '--diagnostic-run',      default=False, type=bool, help="whether to run just few iteration to see if everything is working")
parser.add_argument('-lr', '--learning-rate',       default=1e-3, type=float)
parser.add_argument('-o', '--optimizer',            default="adam", type=str, help="Optimizer to use either adam or ranger or rangerlars")
parser.add_argument('-ua', '--decoder-scse',        default=True, type=bool, help="whether to use scse attention blocks in UNet decoder")
parser.add_argument('-lre', '--lr-coef-encoder',    default=1e-2, type=float, help="Downscale factor for encoder")
parser.add_argument('-nt', '--notes',               default="", type=str, help="Optional: notes about the run")
parser.add_argument('-lrf', '--lr-scheduler-factor',default=1.0, type=float, help="Downscale lr factor for lr scheduler on plateau")
parser.add_argument('-a', '--activation',           default="relu", type=str, help="Activation function either: relu or mish")
parser.add_argument('-sh', '--saturation-hue-jitter', default=0.1, type=float, help="Color jitter parameter for hue and saturation in data augmentation")
parser.add_argument('-d', '--dataset',              default='opbgv2', type=str)
args = parser.parse_args()

# use the cl arguments
# set up which gpu to use
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id
ENCODER_ARCH = args.encoder_architecture
PRETRAIN = args.weights
FREEZE_ENCODER = args.freeze_encoder
TRIAL_RUN = args.diagnostic_run
LR = args.learning_rate
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LBL_SIGMA = args.label_sigma
SCSE = args.decoder_scse
OPTIMIZER = args.optimizer
DF_ENC = args.lr_coef_encoder
LR_FACTOR = args.lr_scheduler_factor
SH_FACTOR = args.saturation_hue_jitter

if args.dataset == "opbgv2":
    DATASET_DIR = DATASET_OPBG_V2
elif args.datset == "opbgv1":
    DATASET_DIR = DATASET_OPBG_V1
else:
    raise ValueError("Dataset argument not recognised: either opbgv1 or opbgv2")

# set up logging folders/files
timestamp = str(datetime.today().isoformat())
exp_title = f"dens_count_{ENCODER_ARCH}_{PRETRAIN}_ep_{EPOCHS}_bs_{BATCH_SIZE}_{'resume' if args.resume else 'scratch'}_{timestamp}"
writer = SummaryWriter(log_dir=f"./tb_runs/{exp_title}")
# TODO occhio che salvo ora sull'altro hard disk direttamente
EXP_DIR = SAVE_DIR / exp_title
os.makedirs(EXP_DIR, exist_ok=False)

print(f"Freezend encoder: {args.freeze_encoder}")

# saving settings as json
with open(EXP_DIR / "settings.json", "w") as fp:
    tmp = vars(args)
    tmp["exp_title"] = exp_title
    json.dump(tmp, fp, indent=4)

#####################################
# %%
# ------------
# B - DATA PREPARATION
# ------------
# define albumentations transpose
# execution is first to last
transforms = A.Compose([
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=SH_FACTOR, hue=SH_FACTOR, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
])
transforms_val = A.Compose([
    ToTensorV2(),
])

dataset_train = DMapData(DATASET_DIR, "train", transforms, lbl_sigma_gauss=LBL_SIGMA)
dataset_valid = DMapData(DATASET_DIR, "valid", transforms_val, lbl_sigma_gauss=LBL_SIGMA)
# dataset_train.show_example(False)
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)


# %%
# ----------------------------------
# C - MODEL, CHECKPOINTS AND RELATED
# ----------------------------------
if (not args.resume) and (PRETRAIN == "imagenet"):
    model = smp.Unet(ENCODER_ARCH, encoder_weights=PRETRAIN, decoder_attention_type="scse")
    print("starting training with iamgenet weights")
else:
    model = smp.Unet(ENCODER_ARCH, decoder_attention_type="scse")

if PRETRAIN == "lysto":
    load_lysto_weights(model, lysto_checkpt_path, "resnet50")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(0)




segm_criterion = torch.nn.MSELoss()
prob_cons_criterion = torch.nn.L1Loss()

# %%

# warmp up 
start_ep = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["model"])
    #optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"loaded checkpoint {args.resume}")
    start_ep = checkpoint["epochs"]   

model = nn.DataParallel(model)
model.cuda(0)

if FREEZE_ENCODER:
    for param in model.module.encoder.parameters():
        param.requires_grad = False

param_groups = [
                  {'params': model.module.encoder.parameters(), 'lr':LR*DF_ENC},
                  {'params': model.module.decoder.parameters(), 'lr':LR},
                  {'params': model.module.segmentation_head.parameters(), 'lr':LR}]

# choosing the optimizer
if OPTIMIZER == "adam":
    optimizer = torch.optim.Adam(param_groups)
elif OPTIMIZER == "ranger":
    optimizer = Ranger(param_groups)
elif OPTIMIZER == "rangerlars":
    optimizer = RangerLars(param_groups)
else:
    raise ValueError("Specified optimizer not implemented")

# setting up lr scheduler
if LR_FACTOR<1.0:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR)

if not args.resume:
    print("Warm up...")
    for i, (img, mask) in enumerate(loader_train):
        if TRIAL_RUN:
            break

        print(f"\rWarming up {i+1}/{len(loader_train)}", end="", flush=True)
        img = img.cuda().float()
        mask = mask.cuda().unsqueeze(1).float()
        out = model(img)
        segm_loss = segm_criterion(out, mask)
        optimizer.zero_grad()
        segm_loss.backward()
        optimizer.step()

print("\nStarting training..")
# %%
# ----------------------------------
# D - TRAINING SECTION
# ----------------------------------

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
        img = img.cuda().float()
        mask = mask.cuda().unsqueeze(1).float()

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

    if True:
        print("\nValidation step...")
        model.eval()
        with torch.no_grad():
            reg_met = dict(pred=[],trgt=[])
            # det_metrics = dict(precision=[], recall=[], f1=[])
            loss_segm = []
            loss_reg = []
            for j, (img, mask) in enumerate(loader_valid):
                if TRIAL_RUN and j>3:
                    break
                # move data to device
                img = img.cuda().float()
                mask = mask.cuda().unsqueeze(1).float()

                # run inference
                out = model(img)
                # compute losses
                segm_loss = segm_criterion(out, mask)
                conservation_loss = prob_cons_criterion(out.sum(dim=[1,2,3]), mask.sum(dim=[1,2,3]))


                counts_pred = out.sum(dim=[1,2,3]).detach().cpu().numpy() / 100.
                counts_gt = mask.sum(dim=[1,2,3]).detach().cpu().numpy() / 100.
                reg_met["pred"].extend(counts_pred)
                reg_met["trgt"].extend(counts_gt)
                loss_segm.append(segm_loss.item())
                loss_reg.append(conservation_loss.item())
                # compute det metrics
                # mask = mask.cpu().squeeze().numpy()
                # pred = out.detach().cpu().squeeze().numpy()
                # for k in range(out.shape[0]):
                #     msk_pred = pred[k]
                #     msk_gt = mask[k]
                #     #print(msk_pred.shape, msk_gt.shape, type(msk_pred), type(msk_gt))
                #     prec, rec, f1 = compute_det_metrics(msk_gt, msk_pred)
                #     det_metrics['precision'].append(prec)
                #     det_metrics['recall'].append(rec)
                #     det_metrics["f1"].append(f1)

                # store losses
                
                writer.add_scalar("segmentation_loss/Valid", segm_loss.item(), epoch * len(loader_valid) + j)
                writer.add_scalar("regression_loss/Valid", conservation_loss.item(), epoch * len(loader_valid) + j)


                # log 
                print(
                    f"\rValid epoch {epoch + 1}/{EPOCHS + start_ep} ({j+1}/{len(loader_valid)}) loss:{loss.item():.4f}|segm_loss:{segm_loss.item():.2f} |cons_loss: {conservation_loss.item():.2f}",
                    end="", flush=True
                    )
            losses_val["segment"].append(np.mean(loss_segm))
            losses_val["conserv"].append(np.mean(loss_reg))
            cae, mae, mse = compute_reg_metrics(reg_met)
            qk, mcc, acc = compute_cls_metrics(reg_met)
            writer.add_scalar("metrics/cae", cae, epoch)
            writer.add_scalar("metrics/mae", mae, epoch)
            writer.add_scalar("metrics/mse", mse, epoch)
            writer.add_scalar("metrics/qkappa", qk, epoch)
            writer.add_scalar("metrics/mcc", mcc, epoch)
            writer.add_scalar("metrics/accuracy",acc,epoch)
            # writer.add_scalar("metrics/precision_10px", np.mean(det_metrics["precision"]), epoch)
            # writer.add_scalar("metrics/recall_10px", np.mean(det_metrics["recall"]), epoch)
            # writer.add_scalar("metrics/f1_10px", np.mean(det_metrics["f1"]), epoch)

    # update learning rate on possible plateau with segmentation loss
    if LR_FACTOR<1.0:
        lr_scheduler.step(losses_val["segment"][-1])

        


    # save checkpoint
    last_checkpoint = {
        "model":model.module.state_dict(),
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

    torch.save(last_checkpoint, EXP_DIR / (name + ".pth"))

