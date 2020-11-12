import os
import re
import bisect
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
from skimage import io
import h5py
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class DataMy:

    def __init__(self, path):
        self.imgs_files = [x for x in os.listdir(path) if x.endswith(".png")]
        self.label_files = [re.sub(".png$", ".txt", x) for x in self.imgs_files]
        self.path = path

    def __getitem__(self, idx):
        img = io.imread(self.path + os.sep + self.imgs_files[idx])/255
        if len(img.shape) < 3:
            img = np.stack([img]*3)
        with open(self.path + os.sep + self.label_files[idx], "r+") as f:
            n_obj = len([x for x in f.readlines() if x.strip()])

        return img, n_obj

    def __len__(self):
        return len(self.imgs_files)



def grade(score, breakpoints=[0,5,10,20,50,200], grades=list(range(7))):
  """
  Convert number of lymphocytes into one of the seven classes for the challenge
  """
  i = bisect.bisect_left(breakpoints, score)
  return grades[i]


class DataLysto(Dataset):

    def __init__(self, path, mode):
      """
      Pytorch dataset wrapper for hdf5 file
      """
      super(DataLysto, self).__init__()

      self.ds = h5py.File(path, 'r')
    
      self.n_tot = self.ds["x"].shape[0]
      train_up_id = int(self.n_tot *0.75)
      self.mode = mode
      if mode == "train":
        self.low, self.up = 0, train_up_id
      elif mode == "valid":
        self.low, self.up = train_up_id, self.n_tot
      elif mode == "test":
        self.low, self.up = 0, self.n_tot
      else:
        raise ValueError

      self.elements_ids = np.arange(self.low, self.up)

    def __getitem__(self, idx):
      idx = self.elements_ids[idx]

      img = self.ds["x"][idx][16:-16,16:-16,:]
      img = self.transforms(img)
      
      if self.mode in ["train", "valid"]:
        trgt = self.ds["y"][idx]
        return img, trgt, grade(trgt)
      else:

        return img
    
    def transforms(self, img):
      trans_list = []
      if self.mode == "train":
        trans_list.extend([
          T.ToPILImage(), 
          T.RandomHorizontalFlip(),
          T.RandomVerticalFlip(),
          T.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        img = T.Compose(trans_list)(img)
      img = torch.from_numpy(np.array(img)).float() / 255
      img = img.permute(2,0,1)
      #img = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img)
      return img

    def __len__(self):
      return len(self.elements_ids)


class OPBGData(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, mode, transformations):
        super().__init__()
        tmp_path = Path(dataset_dir) / mode
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

        transformed = self.transforms(image=img)

        img = transformed["image"] / 255.

        return img, len(annots), self.grade_opbg(len(annots))
  
    def grade_opbg(self, score, breakpoints=[0,5,10,20,50,200], grades=list(range(7))):
      """
      Convert number of lymphocytes into one of the seven classes for the opbg
      """
      i = bisect.bisect_left(breakpoints, score)
      return grades[i]
    
        
    def show_example(self):
        img, n, cls_ = self.__getitem__(0)
        fig = plt.figure(figsize=(6,6))
        plt.imshow(img.permute(1,2,0).numpy())
        plt.title(f"n: {n}, class: {cls_}")
        plt.show()

    def __len__(self):
        return len(self.img_files)

