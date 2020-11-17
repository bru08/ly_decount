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
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from utils import load_lysto_weights
from datasets import DMapData
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
from metrics import compute_cls_metrics, compute_reg_metrics

# %%
def main():
    # ------------
    # A - SETTINGS
    # ------------

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu-id',               default='0', type=str, metavar='N')
    parser.add_argument('-en', '--encoder-architecture',default="resnet50", type=str, help="encoder architecture param for pytorch_segmentation_models")
    parser.add_argument('-ep', '--epochs',              default=120, type=int)
    parser.add_argument('-p', '--weights',              default='imagenet', type=str)
    parser.add_argument('-r', '--resume',               default='', type=str, help="path to checkpoint to use to resume trainig")
    parser.add_argument('-f', '--freeze-encoder',       default=False, type=bool)
    parser.add_argument('-s', '--label-sigma',          default=1, type=float, help="gauss kernel sigma applied to label mask")
    parser.add_argument('-d', '--label-value',          default=100, type=float, help="initialization value for pixel with annotation")
    parser.add_argument('-bs', '--batch-size',          default=16, type=int)
    parser.add_argument('-dr', '--diagnostic-run',      default=False, type=bool, help="whether to run just few iteration to see if everything is working")
    parser.add_argument('-lr', '--learning-rate',       default=1e-3, type=float)
    parser.add_argument('-o', '--optimizer',            default="adam", type=str)
    parser.add_argument('-ua', '--decoder-scse',        default=True, type=bool, help="whether to use scse attention blocks in UNet decoder")
    parser.add_argument('-ds', '--distributed',         default=True, type=bool, help="whether to use distributed dataparallel")
    args = parser.parse_args()

    # set up logging folders/files
    timestamp = str(datetime.today().isoformat())
    exp_title = f"""
        dens_count_{args.encoder_architecture}_{args.weights}_ep_{args.epochs}_bs_{args.batch_size}_
        {'resume' if args.resume else 'scratch'}_{'distributed' if args.distributed else 'mono'}_{timestamp}
    """
    writer = SummaryWriter(log_dir=f"./tb_runs/{exp_title}")
    EXP_DIR = Path(__file__).parent.resolve() / "experiments" / exp_title
    os.makedirs(EXP_DIR, exist_ok=False)


    # fixed settings
    references = dict(
        DATASET_DIR = "/home/riccardi/neuroblastoma_project_countCD3/try_yolo_ultralytics/dataset_nb_yolo_trail",
        BASE_DIR = str(Path(__file__).parent.resolve()),
        lysto_checkpt_path = "/home/papa/ly_decount/A_lysto_regression/experiments/2020-11-08T16:43:50_resnetrefb_30ep_freeze_5ep_difflr/last.pth",
        experiment_title = exp_title,
        EXP_DIR = str(EXP_DIR)
    )

    # saving settings as json
    with open(EXP_DIR / "settings.json", "w") as fp:
        tmp = vars(args)
        tmp["exp_title"] = exp_title
        json.dump(tmp, fp, indent=4)


    # start processes
    world_size = len(args.gpu_id.split(","))
    os.environ['MASTER_ADDR'] = '127.0.0.1' 
    os.environ['MASTER_PORT'] = '9123'   
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id

    print("spawning processes...")
    mp.spawn(train, nprocs=world_size, args=(args, references)) 


def train(gpu_proc_id, args, references):
    
    # train loop variables

    DATASET_DIR = references["DATASET_DIR"]
    lysto_checkpt_path = references["lysto_checkpt_path"]
    EXP_DIR = references["EXP_DIR"]
    exp_title = references["experiment_title"]

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


    # start process and start coordination with nccl backend
                    
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=len(args.gpu_id.split(",")),                              
    	rank=gpu_proc_id                                               
    )  
    print(f"Process on gpu {gpu_proc_id} has started")

    # set seed and log path

    torch.manual_seed(0)
    torch.cuda.set_device(gpu_proc_id)
    writer = SummaryWriter(log_dir=f"./tb_runs/{exp_title}")

    # %%
    # define albumentations transforms and datasets/datasamplers
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

    dataset_train = DMapData(DATASET_DIR, "train", transforms, lbl_sigma_gauss=LBL_SIGMA)
    dataset_valid = DMapData(DATASET_DIR, "valid", transforms, lbl_sigma_gauss=LBL_SIGMA)
    # dataset_train.show_example(False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset_train,
    	num_replicas=len(args.gpu_id.split(",")),
    	rank=gpu_proc_id
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset_valid,
    	num_replicas=len(args.gpu_id.split(",")),
    	rank=gpu_proc_id
    )

    loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, sampler=train_sampler
        )
    loader_valid = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, sampler=valid_sampler
        )

    # checkpoints and weights
    # %%
    if (not args.resume) and (PRETRAIN == "imagenet"):
        model = smp.Unet(ENCODER_ARCH, encoder_weights=PRETRAIN, decoder_attention_type="scse")
        print("starting training with iamgenet weights")
    else:
        model = smp.Unet(ENCODER_ARCH, decoder_attention_type="scse")

    if PRETRAIN == "lysto":
        load_lysto_weights(model, lysto_checkpt_path, "resnet50")
    

    start_ep = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        print(f"loaded checkpoint {args.resume}")
        start_ep = checkpoint["epochs"]

    # %%
    # instance model, optimizer etc
    model.cuda(gpu_proc_id)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_proc_id], find_unused_parameters=True)

    if FREEZE_ENCODER:
        for param in model.module.encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam([
                    {'params': model.module.encoder.parameters(), 'lr':LR},
                    {'params': model.module.decoder.parameters(), 'lr':LR},
                    {'params': model.module.segmentation_head.parameters(), 'lr':LR}
    ])

    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])

    segm_criterion = torch.nn.MSELoss()
    prob_cons_criterion = torch.nn.L1Loss()


    print("Starting training..")
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
            img = img.cuda(non_blocking=True).float()
            mask = mask.cuda(non_blocking=True).unsqueeze(1).float()

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
            if gpu_proc_id == 0:
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
                img = img.cuda(non_blocking=True).float()
                mask = mask.cuda(non_blocking=True).unsqueeze(1).float()

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
                if gpu_proc_id == 0:
                    writer.add_scalar("segmentation_loss/Valid", segm_loss.item(), epoch * len(loader_valid) + j)
                    writer.add_scalar("regression_loss/Valid", conservation_loss.item(), epoch * len(loader_valid) + j)


                # log 
                print(
                    f"\rValid epoch {epoch + 1}/{EPOCHS + start_ep} ({j+1}/{len(loader_valid)}) loss:{loss.item():.4f}|segm_loss:{segm_loss.item():.2f} |cons_loss: {conservation_loss.item():.2f}",
                    end="", flush=True
                    )
            if gpu_proc_id == 0:
                cae, mae, mse = compute_reg_metrics(reg_met)
                qk, mcc, acc = compute_cls_metrics(reg_met)
                writer.add_scalar("metrics/cae", cae, epoch)
                writer.add_scalar("metrics/mae", mae, epoch)
                writer.add_scalar("metrics/mse", mse, epoch)
                writer.add_scalar("metrics/qkappa", qk, epoch)
                writer.add_scalar("metrics/mcc", mcc, epoch)
                writer.add_scalar("metrics/accuracy",acc,epoch)

        


        if gpu_proc_id == 0:
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



if __name__ == "__main__":
    main()