# %%
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import matplotlib.patches as patches
import re

# %%
train_path = Path("/home/papa/ly_decount/dataset_nb_yolo/train")
files = [x for x in os.listdir(train_path) if x.endswith(".png")]
# %%
# choose one image to visualize
idx = 3
with open(train_path/re.sub(".png",".txt", files[idx]), "r") as f:
    annots = f.read().split("\n")
annots = [[float(x)*512 for x in y.split(",")[1:]] for y in annots if y]
print(annots)
img = Image.open(train_path/files[idx])
fig, ax = plt.subplots(figsize=(7,7))
for annot in annots:
    x = annot[0]-annot[2]/2
    y = annot[1]-annot[3]/2
    rect = patches.Rectangle((x,y), annot[2]+4, annot[3]+4, linewidth=1, facecolor="none", edgecolor="green")
    ax.add_patch(rect)

ax.imshow(img)

# %%
