import torch
import re
import os
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

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



class DMapDataFiltered(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, mode, transformations, slide_ids=None, point_label_init=100, lbl_sigma_gauss=1):
        super().__init__()
        tmp_path = Path(dataset_dir) / mode
        self.pt_lbl_init = point_label_init
        self.lbl_sigma_gauss = lbl_sigma_gauss

        self.img_files = []
        for elem in os.listdir(tmp_path):
            if (tmp_path/elem).suffix == ".png":
                slide_id = elem.split("_")[0]
                if slide_ids and slide_id in slide_ids:
                    self.img_files.append(tmp_path/elem)

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