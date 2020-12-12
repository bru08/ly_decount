
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

wsi_id = 29

# already extracted tiles
reference_path = REF_DIR / str(wsi_id)
# create slide object
svs_path = svs_dir / f"{wsi_id}.bif.tif"

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
    maximum=170,
)

plot = tiler.extraction_plot(slide_obj)
im = Image.fromarray(plot)
im.save('test.jpeg')