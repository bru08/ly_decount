"""
Use pytorch DDP to imrpove training speed
"""
# %%
import torch
from torch import nn
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from utils import load_lysto_weights
from datasets import DMapData
import torch.multiprocessing as mp
import torch.distributed as dist

# %%
def main():
    settings = dict(
    ENCODER_ARCH = "resnet50",
    PRETRAIN = "checkpoint" ,# or lysto - imagenet only for resnet50
    lysto_checkpt_path = "/home/papa/ly_decount/A_lysto_regression/experiments/2020-11-08T16:43:50_resnetrefb_30ep_freeze_5ep_difflr/last.pth",
    checkpt_path = "/home/papa/ly_decount/C_count_dens_map/experiments/dens_count_resnet50_lysto_ep_120_bs_16_2020-11-11T03:25:01.266188/best.pth",
    FREEZE_ENCODER = True,
    DATASET_DIR = "/home/riccardi/neuroblastoma_project_countCD3/try_yolo_ultralytics/dataset_nb_yolo_trail" ,
    LR = 1e-3,
    EPOCHS = 200,
    BATCH_SIZE = 16,
    timestamp = str(datetime.today().isoformat()),
    TRIAL_RUN = False,
    N_GPUS = 2,
    N_NODES = 1,
    nr=0,
    )
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    ENCODER_ARCH = settings["ENCODER_ARCH"]
    PRETRAIN = settings["PRETRAIN"]
    EPOCHS = settings["EPOCHS"]
    BATCH_SIZE = settings["BATCH_SIZE"]
    timestamp = settings["timestamp"]

    # set experiment directories and logs
    exp_title = f"dens_count_{ENCODER_ARCH}_{PRETRAIN}_ep_{EPOCHS}_bs_{BATCH_SIZE}__distr_{timestamp}"
    #writer = SummaryWriter(log_dir=f"./tb_runs/{exp_title}")
    settings["writer_dir"] = f'./tb_runs/{exp_title}'
    print(f"Log writing as {settings['writer_dir']}")
    EXP_DIR = f"./experiments/{exp_title}/"
    settings["EXP_DIR"] = EXP_DIR
    os.makedirs(EXP_DIR)

    # start processes
    world_size = settings["N_GPUS"] * settings["N_NODES"]
    settings["world_size"] = world_size
    os.environ['MASTER_ADDR'] = '127.0.0.1' 
    os.environ['MASTER_PORT'] = '9123'   
    print("spawning..")
    mp.spawn(train, nprocs=settings["N_GPUS"], args=(settings,)) 



def train(gpu, settings):


    ENCODER_ARCH = settings["ENCODER_ARCH"]
    PRETRAIN = settings["PRETRAIN"]
    EPOCHS = settings["EPOCHS"]
    BATCH_SIZE = settings["BATCH_SIZE"]
    nr = settings["nr"]
    N_GPUS = settings["N_GPUS"]
    DATASET_DIR = settings["DATASET_DIR"]
    lysto_checkpt_path = settings["lysto_checkpt_path"]
    world_size = settings["world_size"]
    TRIAL_RUN = settings["TRIAL_RUN"]
    FREEZE_ENCODER = settings["FREEZE_ENCODER"]
    EXP_DIR = settings["EXP_DIR"]
    LR = settings["LR"]

    rank = nr * N_GPUS + gpu	# should be == gpu                         
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=world_size,                              
    	rank=rank,                                            
    )  
    print(f"process on gpu {gpu} has started")

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    writer = SummaryWriter(log_dir=settings["writer_dir"])

    # %%
    # define albumentations transpose
    # execution is first to last
    transforms = A.Compose([
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, always_apply=False, p=0.99),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    transforms_valid = A.Compose([
        ToTensorV2(),
    ])
    dataset_train = DMapData(DATASET_DIR, "train", transforms, lbl_sigma_gauss=6)
    dataset_valid = DMapData(DATASET_DIR, "valid", transforms, lbl_sigma_gauss=6)
    # dataset_train.show_example(False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset_train,
    	num_replicas=N_GPUS,
    	rank=rank
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset_valid,
    	num_replicas=N_GPUS,
    	rank=rank
    )

    loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, sampler=train_sampler
        )
    loader_valid = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, sampler=valid_sampler
        )
    # %%
    
    if PRETRAIN == "imagenet":
        model = smp.Unet(ENCODER_ARCH, encoder_weights=PRETRAIN, decoder_attention_type="scse")
        print("Loaded model pretrained on imagenet")
    elif PRETRAIN == "checkpoint":
        model = smp.Unet(ENCODER_ARCH, decoder_attention_type="scse")
        checkpoint = torch.load(settings["checkpt_path"])
        model.load_state_dict(checkpoint["model"])
        print(f"loaded model from checkpoint {checkpoint['epochs']} epochs")
    else:
        model = smp.Unet(ENCODER_ARCH, decoder_attention_type="scse")

    if PRETRAIN == "lysto":
        load_lysto_weights(model, lysto_checkpt_path, ENCODER_ARCH)

    # %%
    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    if FREEZE_ENCODER:
        for param in model.module.encoder.parameters():
            param.requires_grad = False


    optimizer = torch.optim.Adam([
                    {'params': model.module.encoder.parameters(), 'lr':LR},
                    {'params': model.module.decoder.parameters(), 'lr':LR},
                    {'params': model.module.segmentation_head.parameters(), 'lr':LR}
    ])
    if PRETRAIN == "checkpoint":
        optimizer.load_state_dict(checkpoint["optimizer"])

    segm_criterion = torch.nn.MSELoss()
    prob_cons_criterion = torch.nn.L1Loss()

    # %%
    # warmp up 
    if not PRETRAIN == "checkpoint":
        print("Warm up...")
        for i, (img, mask) in enumerate(loader_train):
            if TRIAL_RUN and i > 3:
                break
            img = img.cuda(non_blocking=True).float()
            mask = mask.cuda(non_blocking=True).float()
            out = model(img)
            segm_loss = segm_criterion(out.squeeze(), mask.squeeze())
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
            img = img.cuda(non_blocking=True).float()
            mask = mask.cuda(non_blocking=True).float()

            # run inference
            out = model(img)
            # compute losses
            segm_loss = segm_criterion(out.squeeze(), mask.squeeze())
            conservation_loss = prob_cons_criterion(out.squeeze().sum(dim=[1,2]), mask.squeeze().sum(dim=[1,2]))
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

            writer.add_scalar(f"segmentation_loss_gpu_{gpu}/Train", segm_loss.item(), epoch * len(loader_train) + i)
            writer.add_scalar(f"regression_loss_gpu_{gpu}/Train", conservation_loss.item(), epoch * len(loader_train) + i)

        print("\nValidation step...")
        model.eval()
        with torch.no_grad():
            for j, (img, mask) in enumerate(loader_valid):
                if TRIAL_RUN and j>3:
                    break
                # move data to device
                img = img.cuda(non_blocking=True).float()
                mask = mask.cuda(non_blocking=True).float()

                # run inference
                out = model(img)
                # compute losses
                segm_loss = segm_criterion(out.squeeze(), mask.squeeze())
                conservation_loss = prob_cons_criterion(out.squeeze().sum(dim=[1,2]), mask.squeeze().sum(dim=[1,2]))

        
                # store losses
                losses_val["segment"].append(segm_loss.item())
                losses_val["conserv"].append(conservation_loss.item())
                writer.add_scalar(f"segmentation_loss_gpu_{gpu}/Valid", segm_loss.item(), epoch * len(loader_valid) + j)
                writer.add_scalar(f"regression_loss_gpu_{gpu}/Valid", conservation_loss.item(), epoch * len(loader_valid) + j)
                # log 
                
                print(
                    f"\rValid epoch {epoch + 1}/{EPOCHS} ({j+1}/{len(loader_valid)}) loss:{loss.item():.4f}|segm_loss:{segm_loss.item():.2f} |cons_loss: {conservation_loss.item():.2f}",
                    end="", flush=True
                    )
        
        # save checkpoint
        last_checkpoint = {
            "model":model.module.state_dict(),
            "optimizer":optimizer.state_dict(),
            "losses_tr":losses_tr,
            "losses_val":losses_val,
            "epochs":epoch+1,
            "settings":settings
        }

        avg_val_loss = np.mean(losses_val["conserv"][-len(loader_valid):])
        if avg_val_loss < best_cons_loss:
            best_cons_loss = avg_val_loss
            name = "best"
        else:
            name = "last"

        if gpu == 0:
            torch.save(last_checkpoint, EXP_DIR + name + ".pth")


if __name__ == "__main__":
    main()