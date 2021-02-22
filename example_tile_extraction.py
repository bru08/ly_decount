"""

"""
# %%
from histolab.tiler import GridTiler
from histolab.slide import Slide

from pathlib import Path
import os
import shutil
from PIL import Image

# %%
TMP_EXT_DIR = Path(__file__).parent / 'temp_extraction' 
SVS_PATH = Path(__file__).parent / 'PDL_6.bif.tif'

# %%
slide_obj= Slide(SVS_PATH, processed_path=".", slide_filters="texture")
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

# %%
plot = tiler.extraction_plot(slide_obj)
# %%
Image.fromarray(plot).save('tile_extraction_example.png')

# %%
