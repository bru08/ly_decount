"""
Generate Yolo dataset from OPBG annotations data
- Take the folders with extracted tiles: 1 folder per slide with tiles inside
- Take the folder with the .csv files from VIA annotations tools
- match annotation files with patient id, then create annotations looking first to annotation file
- Create symbolic link of tiles present in VIA annotations inside a train/valid folder structure
- Create .txt files with annotations for the image with same stem in yolo format (coordinates are normalized with image size as yolo darknet standard)
"""

import os
from pathlib import Path
import re

import pandas as pd

IMG_SIDE = 512
BB_SIDE_PX = 13
l_class_id = [0]

IMG_DS_DIR = Path("")
RAW_ANNOTATIONS_DIR = Path("")
DS_DIR = Path("./dataset")

os.makedirs(DS_DIR / "train")
os.makedirs(DS_DIR / "valid")


# explore annotation folder
ann_files = [x for x in os.listdir(RAW_ANNOTATIONS_DIR) if x.endswith(".csv") and re.search("_(\d+)_")]

train_size = round(len(ann_files) * 0.75)
tot_imgs, tot_annot = 0, 0

# match annotated tiles and create symbolic links
for i, annf in enumerate(ann_files):
    dest_dir = "train" if i < train_size else "valid"
    
    slide_id = re.search("_(\d+)_", x).group(1)
    df = pd.read_csv(RAW_ANNOTATIONS_DIR / annf)
    
    # create symlink for images
    for tilename in df["filename"].unique():
        os.symlink(IMG_DS_DIR / slide_id / tilename, DS_DIR / dest_dir / annf)
        
        # create annotations
        tile_df = df.loc[df["filename"] == tilename, "region_shape_attributes"].values
        centers = (json.loads(x) for x in tile_df)
        norm_coords = [(elem.cx/IMG_SIDE, elem.cy/IMG_SIDE, BB_SIDE_PX/IMG_SIDE, BB_SIDE_PX/IMG_SIDE)]
        
        # write annotation file
        with open(DS_DIR / dest_dir / re.sub(".png$", ".txt$", tilename), "w+") as f:
            for coord in norm_coords:
                f.write(",".join([str(x) for x in l_class_id+coord]))
            
            
    
    
        