import torch
from collections import OrderedDict

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
import numpy as np
import matplotlib.pyplot as plt

def load_lysto_weights(model, state_path, encoder_arch):
    assert encoder_arch == "resnet50"
    checkpoint = torch.load(state_path)
    print(list(checkpoint.keys()))
    model_state = checkpoint["model_state"]
    #
    state_keys = [x for x in model_state.keys() if x.startswith("base_modules")]
    model_keys = list(model.encoder.state_dict().keys())
    #
    state_renamed = OrderedDict()
    for i, skey in enumerate(state_keys[1:]):
        try:
            state_renamed[model_keys[i+1]] = model_state[skey]
        except Exception as e:
            print(e)
    # needed for toch segmentation models that by defualt assume resnets have fc layers at the end
    state_renamed["fc.bias"] = 1
    state_renamed["fc.weight"] = 1
    #
    model.encoder.load_state_dict(state_renamed, strict=False)
    print("Succesfully converted and loaded resnet50 weights from lysto pretraining")



def dens_map_to_detection(mask):
    # threshold mask
    t = filters.threshold_otsu(mask)
    lymph = mask > t
    # watershed transform
    distance = ndi.distance_transform_edt(lymph)
    local_maxi = peak_local_max(distance, indices=False, min_distance=7)
    markers = measure.label(local_maxi)
    segmented_cells = watershed(-distance, markers, mask=lymph)
    # center for each connected components
    centers = []
    for elem in np.unique(segmented_cells):
        if elem != 0:
            centers.append(np.argwhere(segmented_cells==elem).mean(axis=0))
    centers = np.array(centers)
    return centers

def plot_dens_detection(img, mask):
    centers = dens_map_to_detection(mask)
    plt.figure(figsize=(7,7))
    plt.imshow(img)
    plt.scatter(*centers.T[::-1], marker="+", c="green", s=50)
    plt.title(f"Detected objects: {len(centers)}")
    plt.show()
