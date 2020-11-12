# %%
# exploring listo data
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# %%
data = h5py.File("./training.h5", 'r')
# %%
X = data["x"]
y = data["y"]
X.shape
# %%
idx = 5
plt.imshow(X[idx])
plt.title(f"N Lymph: {y[idx]}")

# %%
# exploring opbg annotations data
dataset_dir = Path("/home/riccardi/neuroblastoma_project_countCD3/try_yolo_ultralytics/dataset_nb_yolo_trail" )
train_dir = dataset_dir / "train"
annots = [train_dir/x for x in os.listdir(train_dir) if (train_dir/x).suffix==".txt"]
len(annots)
# %%
per_tile_annot = []
for ann in annots:
    with open(ann, "r") as f:
        list = [x for x in f.read().split("\n") if x != '']
    per_tile_annot.append(len(list))
len(per_tile_annot)
per_tile_annot[:10]
# %%
plt.boxplot(per_tile_annot)
plt.show()
# %%
quantiles = np.linspace(0,1,8)[:-1]
print(quantiles)
np.quantile(per_tile_annot, quantiles)
# %%
sugg_categories = [0,5,10,20,50,100,200]