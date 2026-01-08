import pystac_client
import planetary_computer
import odc.stac
import matplotlib.pyplot as plt
import os

from pystac.extensions.eo import EOExtension as eo

# === User Inputs ===
# Example: bbox = [min_lon, min_lat, max_lon, max_lat]
bbox = list(map(float, input("Enter bbox as min_lon,min_lat,max_lon,max_lat: ").split(",")))
time = input("Enter time range (YYYY-MM-DD/YYYY-MM-DD): ")

# === Connect to Planetary Computer ===
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# === Search Landsat Collection ===
search = catalog.search(
    collections=['landsat-c2-l2'],
    bbox=bbox,
    datetime=time,
    query={'eo:cloud_cover': {'lt': 10}},
)

items = search.get_all_items()
print(f"Returned {len(items)} items")

# Select item with least cloud cover
selected_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
print(
    f"Choosing {selected_item.id} from {selected_item.datetime.date()} "
    f"with {selected_item.properties['eo:cloud_cover']}% cloud cover"
)

# === Load Data ===
bands_of_interest = ['nir08', 'red', 'green', 'blue']
data = odc.stac.load(
    [selected_item],
    bands=bands_of_interest,
    bbox=bbox,
).isel(time=0)

# === Create Output Folder ===
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# === Generate NCC (Natural Color Composite) ===
plt.figure(figsize=(8, 8))
plt.imshow(data[['red', 'green', 'blue']].to_array())
plt.title("Natural Color Composite (NCC)")
plt.axis("off")
plt.savefig(os.path.join(output_folder, "NCC.png"), dpi=300, bbox_inches="tight")
plt.close()

# === Generate FCC (False Color Composite) ===
# FCC usually uses NIR, Red, Green → RGB
plt.figure(figsize=(8, 8))
plt.imshow(data[['nir08', 'red', 'green']].to_array())
plt.title("False Color Composite (FCC)")
plt.axis("off")
plt.savefig(os.path.join(output_folder, "FCC.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"NCC and FCC images saved in folder: {output_folder}")
