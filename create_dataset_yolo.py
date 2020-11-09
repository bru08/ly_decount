"""
- Take the folders with extracted tiles 1 folder per slide with tiles inside
- Take the folder with the .csv files from VIA annotations tools
- match annotation files with patient id, then create annotations looking first to annotaiton file
- Create symbolic link of tiles present in VIA annotations inside a train/valid folders
- Create .txt files with annotations for the iamge with same stem in yolo format (normalized with image size)
"""

import os
from pathlib import Path
import re
import json

import pandas as pd

IMG_SIDE = 512
BB_SIDE_PX = 18
l_class_id = 0
TRAIN_PROP = 0.75

# set folder for origin annotations, extracted tiles and destination folder
IMG_DS_DIR = Path("/home/riccardi/neuroblastoma_project_countCD3/output_extraction/CD_3")
RAW_ANNOTATIONS_DIR = Path(__file__).parent / "annotations_nb_opbg"
DS_DIR = Path(__file__).parent / "dataset_nb_yolo"

os.makedirs(DS_DIR / "train", exist_ok=True)
os.makedirs(DS_DIR / "valid", exist_ok=True)

# explore annotation folder
ann_files = [x for x in os.listdir(RAW_ANNOTATIONS_DIR) if x.endswith(".csv") and re.search(r"_(\d+)_", x)]
print(f"Found {len(ann_files)} annotation files")
train_size = round(len(ann_files) * TRAIN_PROP)
# counters
tot_imgs = {"train":0, "valid":0}
tot_annot = {"train":0, "valid":0}

# match annotated tiles and create symbolic links
for i, annf in enumerate(ann_files):
    # send items to train/valid in sequential order
    mode = "train" if i < train_size else "valid"
    
    # get slide id
    slide_id = re.search("_(\d+)_", annf).group(1)
    # read csv annotation file
    df = pd.read_csv(RAW_ANNOTATIONS_DIR / annf)
    
    # create symlink for images, iterate over tile names present in annot file
    for tilename in df["filename"].unique():
        
        # create annotations
        tile_df = df.loc[df["filename"] == tilename, "region_shape_attributes"].values
        centers = [json.loads(x) for x in tile_df if x != "{}"]
        tot_annot[mode] += len(centers)

        if centers:
            norm_coords = []
            for elem in centers:
                # load coordinate and normalize with image size
                tmp = [elem['cx']/IMG_SIDE, elem['cy']/IMG_SIDE, BB_SIDE_PX/IMG_SIDE, BB_SIDE_PX/IMG_SIDE]
                if (elem["cx"] < 0) or (elem["cy"] < 0):
                    # if annotation malformed reject it
                    print(elem)
                    continue
                norm_coords.append(tmp)
            
            tot_imgs[mode] += 1
            os.symlink(IMG_DS_DIR / slide_id / tilename, DS_DIR / mode / tilename)
            
            # write annotation file
            with open(DS_DIR / mode / re.sub(".png", ".txt", tilename), "w+") as f:
                for coord in norm_coords:
                    f.write(f"{l_class_id},")
         
                    f.write(",".join([str(x) for x in coord]))
                    f.write("\n")
                
print("Dataset created")
print(f"Tiles: tot: {tot_imgs['train']+tot_imgs['valid']} train: {tot_imgs['train']} val: {tot_imgs['valid']}")
print(f"Annotations: tot: {tot_annot['train']+tot_annot['valid']} train: {tot_annot['train']} val: {tot_annot['valid']}")
print("Enjoy your experiments :)")
        
        
        