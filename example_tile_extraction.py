"""
Create a tiler object and save an image contianing the tile extraction map
"""
# %%
from pathlib import Path
from PIL import Image

from histolab.tiler import GridTiler
from histolab.slide import Slide

#TODO CHANGE PATHS HERE!
SVS_PATH = Path(__file__).parent / 'PDL_6.bif.tif'
N_TILES = 175
TILE_SIZE = 512
EXTRACTION_LVL = 1

# %%
slide_obj= Slide(str(SVS_PATH), processed_path=".", slide_filters="texture")
slide_obj
# %%
tiler = GridTiler(
    tile_size= (TILE_SIZE, TILE_SIZE),
    level = EXTRACTION_LVL,
    check_tissue=True,
    pixel_overlap= 0,
    prefix=  str(Path(__file__).parent / slide_obj.name),
    suffix=".png",
    partial=0.99,
    maximum=N_TILES,
)

# %%
# create and save the extraction map as png
plot = tiler.extraction_plot(slide_obj)
Image.fromarray(plot).save('tile_extraction_example.png')
# %%
