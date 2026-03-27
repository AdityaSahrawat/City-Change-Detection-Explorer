import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
import geopandas as gpd
from skimage.transform import resize
from scipy.ndimage import median_filter, binary_opening
import matplotlib.pyplot as plt

base_path = "data/raw/S2B_MSIL2A_20260302T053639_N0512_R005_T43REN_20260302T093017.SAFE/GRANULE/*/IMG_DATA"


def get_band(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Band not found: {pattern}")
    return files[0]


# 10m bands
b03 = get_band(f"{base_path}/R10m/*_B03_10m.jp2")
b04 = get_band(f"{base_path}/R10m/*_B04_10m.jp2")
b08 = get_band(f"{base_path}/R10m/*_B08_10m.jp2")

# 20m band
b11 = get_band(f"{base_path}/R20m/*_B11_20m.jp2")

print(b03, b04, b08, b11)


green_src = rasterio.open(b03)
red_src = rasterio.open(b04)
nir_src = rasterio.open(b08)
swir_src = rasterio.open(b11)

# crop to polygon

gdf = gpd.read_file("data/hisar.geojson").to_crs(red_src.crs)

green_crop, _ = mask(green_src, gdf.geometry, crop=True)
red_crop, _ = mask(red_src, gdf.geometry, crop=True)
nir_crop, _ = mask(nir_src, gdf.geometry, crop=True)
swir_crop_raw, _ = mask(swir_src, gdf.geometry, crop=True)

# SWIR is 20m; match to 10m grid
swir_crop = resize(
    swir_crop_raw[0],
    red_crop[0].shape,
    order=1,
    preserve_range=True,
).astype("float32")

# Use float arrays

green = green_crop[0].astype("float32")
red = red_crop[0].astype("float32")
nir = nir_crop[0].astype("float32")

ndvi = (nir - red) / (nir + red + 1e-6)
ndwi = (green - nir) / (green + nir + 1e-6)
ndbi = (swir_crop - nir) / (swir_crop + nir + 1e-6)

classification = np.zeros_like(ndvi, dtype=np.uint8)
classification[ndwi > 0.1] = 1
classification[(ndvi > 0.5) & (classification == 0)] = 2
classification[(ndbi > 0.15) & (ndvi < 0.3) & (classification == 0)] = 3
classification = median_filter(classification, size=3)

built_up = (classification == 3)
built_up = binary_opening(built_up, structure=np.ones((3,3)))
classification[built_up] = 3

from matplotlib.colors import ListedColormap, BoundaryNorm

cmap = ListedColormap(["#2d2d2d", "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"])
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

plt.figure(figsize=(12, 10))
plt.imshow(classification, cmap=cmap, norm=norm, interpolation="nearest")
plt.title("0=bg, 1=water, 2=veg, 3=built, 4=soil")

# colorbar with discrete labels
cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(["bg", "water", "veg", "built", "soil"])

# add explicit legend
from matplotlib.patches import Patch
legend_patches = [
    Patch(color="#2d2d2d", label="Background"),
    Patch(color="#1f77b4", label="Water"),
    Patch(color="#2ca02c", label="Vegetation"),
    Patch(color="#d62728", label="Built-up"),
    Patch(color="#ff7f0e", label="Soil"),
]
plt.legend(handles=legend_patches, loc="lower right", framealpha=0.9)

plt.axis("off")
plt.show()
