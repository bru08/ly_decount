
# %%
import numpy as np
import pandas as pd
from histolab_chia.tiler import GridTiler
from histolab_chia.slide import Slide
from pathlib import Path
import torch
import re
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import segmentation_models_pytorch as smp
import shutil


# %%
# folder with extracted tiles
REF_DIR = Path("/home/riccardi/neuroblastoma_project_countCD3/grid_tile_extration/1_grid_tile_extraction_for_annotations/output_extraction/CD_3")
# folder with count outputs
CNT_DIR = Path("/home/papa/ly_decount/eval_counts")
# temporary directory to store extracted tiles
TMP_EXT_DIR = Path("/datadisk/neuroblastoma_checkpoints_backup/temp_extraction")
ANNOT_PATH = Path("/home/papa/ly_decount/annotations_nb_opbg_v2")
checkpoint_path = "/datadisk/neuroblastoma_checkpoints_backup/experiments_dens_map_batch3/dens_count_efficientnet-b3_imagenet_ep_120_bs_32_resume_2020-11-22T07:53:03.704015/last.pth"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%
# wsi in datset
DS_WSI = list(set([ x.split("_")[1] for x in os.listdir(ANNOT_PATH) if x.endswith(".csv")]))
print(DS_WSI, len(DS_WSI))
svs_dir = Path("/datadisk/OBPG_NB_TIF/OBPG/CD_3")
# %%

# load model
device = torch.device("cuda")
model = smp.Unet("efficientnet-b3", decoder_attention_type="scse")
model.to(device)
checkpoint = torch.load(checkpoint_path)
print(f"Epochs trained {checkpoint['epochs']}")
model.load_state_dict(checkpoint["model"])
model.eval()

transforms_val = A.Compose([ToTensorV2()])

try:
    shutil.rmtree(TMP_EXT_DIR)
except:
    print("no folder to clean")

with open(CNT_DIR/"counts.csv", "a") as f:
        f.write(f"wsi,mean,median,std,min,max,n_tiles\n")

# do operations for each slide
for i, wsi_id in enumerate(DS_WSI):
    print(f"Processing wsi {wsi_id} {i+1}/{len(DS_WSI)}")
    # already extracted tiles
    reference_path = REF_DIR / str(wsi_id)
    # create slide object
    svs_path = svs_dir / f"{wsi_id}.bif.tif"
    os.makedirs(TMP_EXT_DIR)
    slide_obj= Slide( str(svs_path), processed_path=".", slide_filters="texture")
    # extract additional tiles
    tiler = GridTiler(
        tile_size= (512, 512),
        level = 1,
        check_tissue=True,
        pixel_overlap= 0,
        prefix=  str(TMP_EXT_DIR) + "/" + slide_obj.name + "_",
        suffix=".png",
        partial=0.05,
        maximum=None,
    )
    # extract tiles
    tiler.extract(slide_obj)

    # count and save
    files = [x for x in os.listdir(TMP_EXT_DIR) if x.endswith(".png")]
    counts = []
    with torch.no_grad():
        for j, file in enumerate(files):
            print(f"Inference... {j+1}/{len(files)}", end="", flush=True)
            img = np.array(Image.open(TMP_EXT_DIR/file).convert("RGB"))
            img_transformed = transforms_val(image=img)
            img = img_transformed["image"] / 255.
            inp = img.unsqueeze(0).to(device)
            print("aa", inp.shape)
            out = model(inp).squeeze().cpu().numpy().clip(0)
            count = out.sum()
            counts.append(count)
    
    with open(CNT_DIR/"counts.csv", "a") as f:
        f.write(f"{wsi_id},{np.mean(counts)},{np.median(counts)},{np.std(counts)},{np.min(counts)},{np.max(counts)},{len(counts)}\n")
        

    # delete directory and its content, keep counts
    shutil.rmtree(TMP_EXT_DIR)
