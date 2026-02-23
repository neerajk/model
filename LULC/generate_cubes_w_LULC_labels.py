
# This script generates Sentinel-2 image cubes with corresponding LULC labels for fine-tuning a model. 
# It uses the Open Data Cube (ODC) to load both Sentinel-2 and LULC
# data, performs necessary coordinate transformations, and prepares the data for model training.
# The generated cubes will be saved in a format suitable for training a machine learning model.
# Note: Ensure you have the necessary libraries installed and access to the Planetary Computer API.

import pandas as pd
from dask.diagnostics import ProgressBar
from pyproj import Transformer

from pystac_client import Client
import planetary_computer
from odc.stac import stac_load
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil


# Define the area of interest and time range
bbox = [77.90, 30.20, 78.20, 30.45]
month_range = "2023-11-01/2023-11-30"
year_range = "2023-01-01/2023-12-31"

# Load LULC data using ODC
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# ----------------------------------
# Search Sentinel
# ----------------------------------
print("Searching Sentinel scenes...")

# We filter for low cloud cover to ensure better quality data for LULC classification
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=month_range,
    query={"eo:cloud_cover": {"lt": 10}},
)

# Get the list of items and sign them for access
items = list(search.get_items())
print("Scenes found:", len(items))

# Sign items for access
items = [planetary_computer.sign(item) for item in items]

# Load Sentinel data using ODC
ds = stac_load(
    items,
    bands=["B02","B03","B04","B08"],
    bbox=bbox,
    resolution=10,
    chunks={"x":1024,"y":1024},
)

print("Downloading Sentinel data...")

with ProgressBar():
    ds = ds.load()

print("Sentinel loaded.")
print(ds)

# ----------------------------------
# Search LULC
# ----------------------------------
print("Loading IO LULC...")

# The IO LULC dataset provides annual land cover classifications, which we will use as labels for our model.
search_lulc = catalog.search(
    collections=["io-lulc-annual-v02"],
    bbox=bbox,
    datetime=year_range,
)

# Get the list of LULC items and sign them for access
lulc_items = list(search_lulc.get_items())
lulc_items = [planetary_computer.sign(item) for item in lulc_items]

# Load LULC data using ODC
lulc_ds = stac_load(
    lulc_items,
    bbox=bbox,
    resolution=10,
    chunks={"x":1024,"y":1024},
)

with ProgressBar():
    lulc_ds = lulc_ds.load()
# Convert to numpy array and ensure it's int64 for model compatibility
# Select 2023 LULC explicitly
lulc_2023 = lulc_ds.sel(time="2023-01-01")

# Extract the data variable safely
var_name = list(lulc_2023.data_vars)[0]
lulc = lulc_2023[var_name].values.astype(np.int64)

print('*******')
print("Final LULC shape:", lulc.shape)
print('*******')


# ----------------------------------

# Coordinate transformation

# Sentinel-2 data is in UTM, we need to convert to lat/lon for the model

transformer = Transformer.from_crs(ds.odc.crs, "EPSG:4326", always_xy=True)


def normalize_time(dt):
    week = dt.timetuple().tm_yday / 365.0
    hour = dt.hour / 24.0
    return week, hour

def normalize_latlon(lat, lon):
    lat_norm = (lat + 90) / 180.0
    lon_norm = (lon + 180) / 360.0
    return lat_norm, lon_norm

def encode_scalar(x):
    return np.array([
        np.sin(2*np.pi*x),
        np.cos(2*np.pi*x)
    ], dtype=np.float32)




#------------------

#Generate Chips

#---------------
CHIP_SIZE = 256
chips = []
masks = []
latlon_meta = []
time_meta = []


for t_idx in tqdm(range(len(ds.time)), desc="Processing timesteps"):

    dt = pd.to_datetime(ds.time.values[t_idx]).to_pydatetime()
    week_norm, hour_norm = normalize_time(dt)

    week_enc = encode_scalar(week_norm)
    hour_enc = encode_scalar(hour_norm)

    img = ds.isel(time=t_idx).to_array().values.astype(np.float32) / 10000.0
    _, H, W = img.shape

    for i in range(0, H - CHIP_SIZE, CHIP_SIZE):
        for j in range(0, W - CHIP_SIZE, CHIP_SIZE):

            chip = img[:, i:i+CHIP_SIZE, j:j+CHIP_SIZE]
            mask_chip = lulc[i:i+CHIP_SIZE, j:j+CHIP_SIZE]

            if mask_chip.ndim != 2:
                print("Mask dimension error:", mask_chip.shape)
                continue
            if chip.shape != (4, CHIP_SIZE, CHIP_SIZE):
                continue

            # center pixel coordinate
            y_coord = float(ds.y.values[i + CHIP_SIZE//2])
            x_coord = float(ds.x.values[j + CHIP_SIZE//2])

            lon, lat = transformer.transform(x_coord, y_coord)

            lat_norm, lon_norm = normalize_latlon(lat, lon)
            lat_enc = encode_scalar(lat_norm)
            lon_enc = encode_scalar(lon_norm)

            chips.append(chip)
            masks.append(mask_chip)
            latlon_meta.append(np.concatenate([lat_enc, lon_enc]))
            time_meta.append(np.concatenate([week_enc, hour_enc]))

print("Total chips created:", len(chips))

#Save the cubes

os.makedirs("cubes", exist_ok=True)

for idx in tqdm(range(len(chips))):

    np.savez(
        f"cubes/cube_{idx}.npz",
        image=chips[idx].astype(np.float32),      # (4,128,128)
        mask=masks[idx].astype(np.int64),         # (128,128)
        latlon=latlon_meta[idx],                 # (4,)
        time=time_meta[idx]                      # (4,)
    )