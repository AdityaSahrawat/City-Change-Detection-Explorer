import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
import geopandas as gpd
from skimage.transform import resize
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
classification[ndwi > 0.2] = 1
classification[(ndvi > 0.5) & (classification == 0)] = 2
classification[(ndbi > 0.1) & (classification == 0)] = 3

plt.imshow(classification, cmap="viridis")
plt.title("Classification Map")
plt.colorbar()
plt.axis("off")
plt.show()
