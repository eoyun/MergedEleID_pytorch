import zarr
import numpy as np

root = zarr.open_group("/home/eoyun/data/train.zarr", mode="r")

labels = root["label_str"][:]
H = root["H_mass"][:]
A = root["A_mass"][:]

print("unique labels:")
print(np.unique(labels))

print("\nunique H_mass:")
print(np.unique(H))

print("\nunique A_mass:")
print(np.unique(A))
