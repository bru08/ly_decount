"""
Perform a small version of dap
1. Take dataset train/test split
2. Take train data
3. k-fold split function (k=5)
4. train on k-1 validate on 1 each one of the five time
5. make meaningful names to the various runs of the models
"""
# %%
# imports
import torch
import json
from torch import nn
import time
import re
import gc
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
from datasets import DMapData, DMapDataFiltered
from utils import load_lysto_weights, compute_det_metrics
from ranger2020 import Ranger


def train_func(loader_train,loader_valid, args, writer=None, checkpoint=None):
    """
    Start training from a configuration file and a given training set
    """

    ENCODER_ARCH = args.get('encoder_architecture')
    PRETRAIN = args.get('weights')
    FREEZE_ENCODER = args.get('freeze_encoder')
    TRIAL_RUN = args.get('diagnostic_run')
    LR = args.get('learning_rate')
    EPOCHS = args.get('epochs')
    OPTIMIZER = args.get('optimizer')
    DF_ENC = args.get('lr_coef_encoder')
    LR_FACTOR = args.get('lr_scheduler_factor')
    RESUME = args.get('resume')
    print(f"check boolean: {FREEZE_ENCODER}")

    # ----------------------------------
    # C - MODEL, CHECKPOINTS AND RELATED
    # ----------------------------------
    if (not RESUME) and (PRETRAIN == "imagenet"):
        model = smp.Unet(ENCODER_ARCH, encoder_weights=PRETRAIN, decoder_attention_type="scse")
        print("starting training with iamgenet weights")
    else:
        model = smp.Unet(ENCODER_ARCH, decoder_attention_type="scse")

    if PRETRAIN == "lysto":
        load_lysto_weights(model, lysto_checkpt_path, "resnet50")

    segm_criterion = torch.nn.MSELoss()


    # %%
    start_ep = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        print(f"loaded checkpoint weights")

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
    else:
        raise ValueError("Specified optimizer not implemented")

    #if checkpoint is not None:
        #optimizer.load_state_dict(checkpoint["optimizer"])

    # setting up lr scheduler
    if LR_FACTOR<1.0:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR)
    losses_tr = []
    losses_val = []
    for epoch in range(start_ep, EPOCHS + start_ep):
        if TRIAL_RUN and (start_ep + epoch) >3:
            break

        model.train()
        print(f"Start training epochs {epoch + 1}..")
        for i, (img, mask) in enumerate(loader_train):
            if TRIAL_RUN and i>3:
                break

            # move data to device
            img = img.cuda().float()
            mask = mask.cuda().unsqueeze(1).float()

            # run inference
            out = model(img)
            # compute losses
            loss = segm_criterion(out, mask)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log 
            print(f"\rEpoch {epoch + 1}/{EPOCHS+start_ep} ({i+1}/{len(loader_train)})", end="", flush=True)
            # store losses
            losses_tr.append(loss.item())
            if writer:
                writer.add_scalar("segmentation_loss/Train", loss.item(), epoch * len(loader_train) + i)
        print("")


        if True:
            print("\nValidation step...")
            segm_criterion = torch.nn.MSELoss()
            model.eval()
            with torch.no_grad():
                reg_met = dict(pred=[],trgt=[])
                # det_metrics = dict(precision=[], recall=[], f1=[])
                loss_segm = []
                for j, (img, mask) in enumerate(loader_valid):
                    # move data to device
                    img = img.cuda().float()
                    mask = mask.cuda().unsqueeze(1).float()

                    # run inference
                    out = model(img)
                    # compute losses
                    segm_loss = segm_criterion(out, mask)

                    counts_pred = out.sum(dim=[1,2,3]).detach().cpu().numpy() / 100.
                    counts_gt = mask.sum(dim=[1,2,3]).detach().cpu().numpy() / 100.
                    reg_met["pred"].extend(counts_pred)
                    reg_met["trgt"].extend(counts_gt)
                    loss_segm.append(segm_loss.item())
            
                    
                    writer.add_scalar("segmentation_loss/Valid", segm_loss.item(), epoch * len(loader_valid) + j)
                    # writer.add_scalar("regression_loss/Valid", conservation_loss.item(), epoch * len(loader_valid) + j)


                    # log 
                    print(f"\rValid epoch ({j+1}/{len(loader_valid)})",end="", flush=True)
                losses_val.append(np.mean(loss_segm))
                cae, mae, mse = compute_reg_metrics(reg_met)
                qk, mcc, acc = compute_cls_metrics(reg_met)
                writer.add_scalar("metrics/cae", cae, epoch)
                writer.add_scalar("metrics/mae", mae, epoch)
                writer.add_scalar("metrics/mse", mse, epoch)
                writer.add_scalar("metrics/qkappa", qk, epoch)
                writer.add_scalar("metrics/mcc", mcc, epoch)
                writer.add_scalar("metrics/accuracy",acc,epoch)
                metrics = dict(
                    cae=cae,
                    mae=mae,
                    mse=mse,
                    qk=qk,
                    mcc=mcc,
                    acc=acc
                )

        if LR_FACTOR<1.0:
            lr_scheduler.step(losses_val[-1])

    return model, losses_tr, optimizer, epoch, metrics, loss_segm, 



def validate(loader_valid,model, writer):
    print("\nValidation step...")
    segm_criterion = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        reg_met = dict(pred=[],trgt=[])
        # det_metrics = dict(precision=[], recall=[], f1=[])
        loss_segm = []
        for j, (img, mask) in enumerate(loader_valid):
            # move data to device
            img = img.cuda().float()
            mask = mask.cuda().unsqueeze(1).float()

            # run inference
            out = model(img)
            # compute losses
            segm_loss = segm_criterion(out, mask)

            counts_pred = out.sum(dim=[1,2,3]).detach().cpu().numpy() / 100.
            counts_gt = mask.sum(dim=[1,2,3]).detach().cpu().numpy() / 100.
            reg_met["pred"].extend(counts_pred)
            reg_met["trgt"].extend(counts_gt)
            loss_segm.append(segm_loss.item())
    
            
            # writer.add_scalar("segmentation_loss/Valid", segm_loss.item(), epoch * len(loader_valid) + j)
            # writer.add_scalar("regression_loss/Valid", conservation_loss.item(), epoch * len(loader_valid) + j)


            # log 
            print(f"\rValid epoch ({j+1}/{len(loader_valid)})",end="", flush=True)
        #losses_val.append(np.mean(loss_segm))
        cae, mae, mse = compute_reg_metrics(reg_met)
        qk, mcc, acc = compute_cls_metrics(reg_met)
        # writer.add_scalar("metrics/cae", cae, epoch)
        # writer.add_scalar("metrics/mae", mae, epoch)
        # writer.add_scalar("metrics/mse", mse, epoch)
        # writer.add_scalar("metrics/qkappa", qk, epoch)
        # writer.add_scalar("metrics/mcc", mcc, epoch)
        # writer.add_scalar("metrics/accuracy",acc,epoch)
        metrics = dict(
            cae=cae,
            mae=mae,
            mse=mse,
            qk=qk,
            mcc=mcc,
            acc=acc
        )
    return model, losses_tr, optimizer, epoch, metrics, loss_segm

# %%
# just the train data as specified in DAP MAQC etc..
DATA_DIR = Path("/home/papa/ly_decount/dataset_nb_yolo_v1")
K = 5
effnet_settings_path = "/datadisk/neuroblastoma_checkpoints_backup/validations_effnet/settings_effnet.json"
out_effnet = "/datadisk/neuroblastoma_checkpoints_backup/validations_effnet"
resnet_settings_path = "/datadisk/neuroblastoma_checkpoints_backup/validations_resnet/settings_resnet.json"
out_resnet = "/datadisk/neuroblastoma_checkpoints_backup/validations_resnet"

settings_path = resnet_settings_path
OUT_DIR = Path(out_resnet)
lysto_checkpt_path = "/home/papa/ly_decount/A_lysto_regression/resnet50_ref_b_lysto_trained_5epochfinetuned.pth"
common_checkpoint = ""
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
fields = "mcc qk acc mae mse".split()

with open(settings_path, "r") as f:
    settings = json.load(f)

with open("k_fold_splits_v1.json", "r") as f:
    splits_list = json.load(f)

# %%
transforms = A.Compose([
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
])

transforms_val = A.Compose([
    ToTensorV2(),
])



for rep in range(1,4):
    # for loop over different cross val repetition (4)
    t = str(rep)
    splits = splits_list[t]
    # compose rounds of a single k-fold validation
    rounds = []
    folds = list(splits.keys())
    for key in folds:
        train, test = [], []
        for k, v in splits.items():
            if k == key:
                test.extend(splits[k])
            else:
                train.extend(splits[k])   
        rounds.append((train, test))

    #iterate over folds in single cross validation
    for i, (train_ids, valid_ids) in enumerate(rounds):
        print(f"Starting round {i+1} of {len(rounds)}")
        ti = time.time()
        name_fmt = "resnet50_rep_{}_round_{}"
        name = name_fmt.format(rep, i)
        writer = SummaryWriter(OUT_DIR/"tb_runs"/name)

        # DATASET DEFINITION FOR THE fold
        train_ds = DMapDataFiltered(
            DATA_DIR, "train", transforms, slide_ids=train_ids, lbl_sigma_gauss=5
            )
        valid_ds = DMapDataFiltered(
            DATA_DIR, "train", transforms_val, slide_ids=valid_ids, lbl_sigma_gauss=5
            )
        print(f"Round {i+1}/{len(rounds)} train/valid: {len(train_ds)}/{len(valid_ds)}")

        train_loader = DataLoader(train_ds, batch_size=settings["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=settings["batch_size"], shuffle=False)

        if common_checkpoint:
            checkpoint = torch.load(common_checkpoint)
        else:
            # possible_checkpoint = name_fmt.format(rep-1,i) + ".pth"
            # print(f"Checking for possible {possible_checkpoint} checkpoint")
            # if rep > 0 and possible_checkpoint in os.listdir(OUT_DIR):
            #     checkpoint = torch.load(OUT_DIR / possible_checkpoint)
            #     print(f"checkpoint found")
            # else:
            checkpoint = None
            print("checkpoint not found")
        

        # Train the model
        model, losses_tr, optimizer, epoch, metrics, losses_val = train_func(
            train_loader,valid_loader, settings, writer, checkpoint
            )

        # tt = (time.time() - ti) / (60**2) # time in hours
        # print(f"Completed Round {i+1} in {tt:.2f} hours.")
        # print("Validating..")

        # # Validate model trained on i-th fold split
        # metrics, losses_val = validate(valid_loader, model, writer)

        # save metrics and close k-fold validation round
        with open(OUT_DIR/f"results_log_rep{rep}.csv", "a") as f:
            # rep fold *fields trainsize validsize
            f.write(f"{rep}")  # kfold validation repetition in our case just one so always 0
            f.write(f",{i}")
            for key in fields:
                f.write(f",{metrics[key]}")
            f.write(f",{len(train_ds)},{len(valid_ds)}")
            f.write("\n")


        last_checkpoint = {
            "model":model.module.state_dict(),
            "losses_tr":losses_tr,
            "losses_val":losses_val,
            "train_ids":train_ids,
            "val_ids":valid_ids,
            "metrics": metrics,
            "optimizer":optimizer.state_dict(),
            "epochs":epoch
        }
        
        torch.save(last_checkpoint, OUT_DIR / (name + ".pth"))
        del model
        gc.collect()





