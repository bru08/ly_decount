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
from torch import nn
import time
import re
import torchvision
import segmentation_models_pytorch as smp
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_lysto_weights, dens_map_to_detection, plot_dens_detection, match_det
from metrics import compute_cls_metrics, compute_reg_metrics

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)

import staintools 
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
from matplotlib import cm
# %%

# %%
ENCODER_ARCH = "resnet50"
PRETRAIN = "lysto" # or lysto - imagenet only for resnet50
lysto_checkpt_path = "/home/papa/ly_decount/A_lysto_regression/experiments/2020-11-08T16:43:50_resnetrefb_30ep_freeze_5ep_difflr/last.pth"
FREEZE_ENCODER = True
DATASET_DIR = "/home/papa/ly_decount/dataset_nb_yolo" 
# checkpoint high augmentations
#  "/home/papa/ly_decount/C_count_dens_map/experiments/dens_count_efficientnet-b3_imagenet_ep_120_bs_32_scratch_2020-11-19T12:44:06.569185/last.pth"
# checkpoint best but long training
#  /datadisk/neuroblastoma_checkpoints_backup/experiments/dens_count_efficientnet-b3_imagenet_ep_120_bs_24_resume_2020-11-18T01:22:59.330088/last.pth
# checkpoint onesto
# 


checkpoint_path = "/datadisk/neuroblastoma_checkpoints_backup/experiments/dens_count_efficientnet-b3_imagenet_ep_120_bs_24_resume_2020-11-18T01:22:59.330088/last.pth"

LR = 1e-3
EPOCHS = 5
BATCH_SIZE = 16

# %%
class DMapData(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, mode, transformations, point_label_init=100, lbl_sigma_gauss=1):
        super().__init__()
        tmp_path = Path(dataset_dir) / mode
        self.pt_lbl_init = point_label_init
        self.lbl_sigma_gauss = lbl_sigma_gauss
        self.img_files = [tmp_path/x for x in os.listdir(tmp_path) if (tmp_path/x).suffix == ".png"]
        self.transforms = transformations
        self.setup_standardizer()

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


        # standardize before
        #img = self.normalizer.transform(img)

        transformed = self.transforms(image=img, mask=mask_lbl)

        img = transformed["image"] / 255.
        mask = transformed["mask"]

        return img, mask
    
    def getimage(self, idx):
        img = np.array(Image.open(self.img_files[idx]).convert("RGB"))
        return img

    def generate_label_mask(self, img, annotations):
        mask = np.zeros(img.shape[:2])
        s = self.lbl_sigma_gauss
        for annot in annotations:

            mask[annot[1], annot[0]] = self.pt_lbl_init
        mask_blur = gaussian_filter(mask, (s,s), order=0)
        mask_blur /= mask_blur.sum()
        mask_blur *= (100 * len(annotations))
        return mask_blur
        
    def show_example(self, with_mask=True, idx=1, raw=False):

        img, lbl = self.__getitem__(idx)
        img = img.permute(1,2,0).numpy()
        if raw:
            img = self.getimage(idx)
        fig = plt.figure(figsize=(6,6))
        plt.imshow(img)
        if with_mask:
            plt.imshow(lbl.numpy(), alpha=0.5)
        plt.title(f"Sum: {lbl.sum():.2f}")
        plt.show()

    def setup_standardizer(self):
        img = np.array(Image.open(self.img_files[1]).convert("RGB"))
        self.normalizer = staintools.StainNormalizer(method='macenko')
        self.normalizer.fit(img)

    def __len__(self):
        return len(self.img_files)




# %%
# define albumentations transpose
# execution is first to last
transforms = A.Compose([
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
])
transforms_val = A.Compose([
    ToTensorV2(),
])
dataset_train = DMapData(DATASET_DIR, "train", transforms, lbl_sigma_gauss=5)
dataset_valid = DMapData(DATASET_DIR, "valid", transforms_val, lbl_sigma_gauss=5)
dataset_train.show_example(False, raw=False)
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=True)

# %%
dataset_train.show_example(with_mask=False, idx=1)
#

# %%
# isntance model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model = smp.Unet("efficientnet-b3", decoder_attention_type="scse")
model.to(device)

# %%
checkpoint = torch.load(checkpoint_path)
model_state_dict = OrderedDict()
for k,v in checkpoint["model"].items():
    model_state_dict[k.split(".",1)[1]] = v
# %%
# TODO beware what model you are loading normal/dataparallel
print(list(checkpoint.keys()))
print(f"Epochs trained {checkpoint['epochs']}")
model.load_state_dict(checkpoint["model"])

#%%
# %%
plt.plot(checkpoint["losses_tr"]["segment"])
#plt.yscale("log")

# %%
plt.plot(checkpoint["losses_tr"]["conserv"])
#plt.yscale("log")
# %%
#### TRY INFERENCE
img , msk = dataset_valid[10]

model.eval()
with torch.no_grad():
    out = model(img.unsqueeze(0).to(device))

fig, ax = plt.subplots(ncols=2,nrows=3, figsize=(12,18))
img_d = img.cpu().permute(1,2,0).numpy()
msk_out = out.squeeze().cpu().numpy()
msk = msk.numpy()
gt = dens_map_to_detection(msk)
pred = dens_map_to_detection(msk_out)



# plotting
ax[0,0].imshow(img_d)
ax[0,0].imshow(msk_out>msk_out.max()*0.2, alpha=0.7)
ax[0,0].set_title("Image and density map overlay thresholded 0.3*max")
ax[0,1].imshow(msk_out)
ax[0,1].set_title(f"Predicted density map, (Integral/d; {msk_out.sum().item()/100:.2f})")
ax[1,0].imshow(img_d)
ax[1,0].set_title("Input image")
ax[1,1].imshow(msk)
ax[1,1].set_title(f"Target density map (integral/d: {msk.sum().item()/100:.2f})")

ax[2,0].imshow(img_d, alpha=0.8)
ax[2,0].scatter(*gt.T[::-1], s=60, edgecolor="green", facecolor="none", linewidths=2, label="gt")
ax[2,0].scatter(*pred.T[::-1], c="orange", marker="+", label="pred")
ax[2,0].legend()




plt.tight_layout()
plt.show()



# COMPUTE DETECTION METRICS
# %%
m = distance_matrix(gt, pred, p=2)
res = linear_sum_assignment(m)

matches = []
dist_rejected = 0
for i in range(len(gt)):
    try:
        gt_m = gt[res[0][i]]
        pred_m = pred[res[1][i]]
        if euclidean(gt_m, pred_m) < 20:
            group = np.array([gt_m, pred_m])
            matches.append(group)
        else:
            dist_rejected +=1
    except IndexError:
        pass


print(f"Target objects : {len(gt)}")
print(f"Detections: {len(pred)}")
print(f"Matched : {len(matches)}")
print(f"Rejected matches based on distance threshold: {dist_rejected}")
print(f"Unmatched predictions: {len(pred) - len(matches) - dist_rejected}")
print(f"Unmatched ground truths: {len(gt) - len(matches)}")

# %%
plt.figure(figsize=(7,7))
plt.imshow(img_d, alpha=0.7)

colors = cm.tab10.colors

for i, elem in enumerate(matches):
    k = i%len(colors)
    plt.scatter(*elem[0].T[::-1], s=50, alpha=0.4, color=colors[k])
    plt.scatter(*elem[1].T[::-1], s=50, marker="+", color=colors[k])

# %%
recall = len(matches) / len(gt)
precision = len(matches) / len(pred)
f_one = 2 / (1/recall + 1/precision)
print(precision, recall, f_one)

# %%
# compute metrics on the validation set for this obj det
match_thresh = 10
metrics = dict(precision=[], recall=[], f1=[])
diff = []
preds_num = []
gt_num = []
model.eval()
ti = time.time()
with torch.no_grad():

    for i, (img , msk) in enumerate(dataset_valid):
        print(f"\r{i+1}/{len(dataset_valid)}", end="", flush=True)
        msk_out = model(img.unsqueeze(0).to(device)).detach()
        # compute detections
        gt = dens_map_to_detection(msk.numpy())
        pred = dens_map_to_detection(msk_out.cpu().squeeze().numpy())
        #diff.append(len(gt)-len(pred))
        preds_num.append(len(pred))
        gt_num.append(len(gt))
        # matching
        if pred.shape[0] and gt.shape[0]:
            matches = match_det(gt, pred, d_th=match_thresh)
            # compute metrics and append
            recall = len(matches) / len(gt) if len(gt) else 0
            precision = len(matches) / len(pred) if len(pred) else 0 
            f_one = 2 / (1/(recall + 1e-6) + 1/(precision+1e-6))
            #
        else:
            precision, recall, f_one = 0,0,0

        metrics["recall"].append(recall)
        metrics["precision"].append(precision)
        metrics["f1"].append(f_one)
tt = time.time() -ti
# %%

preds_num = np.array(preds_num)
gt_num = np.array(gt_num)
cae, mae, mse = compute_reg_metrics(dict(pred=preds_num, trgt=gt_num))
qk, mcc, acc = compute_cls_metrics(dict(pred=preds_num, trgt=gt_num))
metrics["cae"] = cae
metrics["mae"] = mae
metrics["mse"] = mse
metrics["qk"] = qk
metrics["mcc"] = mcc
metrics["acc"] = acc
# %%
print(f"Completed inference on {len(dataset_valid)} tile in {tt:.2f}: {len(dataset_valid)/tt:.2f} fps")
print(Path(checkpoint_path).stem)
print("Average metrics over validation set")
for k,v in metrics.items():
    print(f"{k:10s}: {np.mean(v):.2f}")


# %%
