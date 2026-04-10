import zarr
import numpy as np

root = zarr.open_group("/home/eoyun/data/train.zarr", mode="r")
w = root["weight"][:]

print("global weight stats")
print("min =", np.min(w))
print("max =", np.max(w))
print("sum =", np.sum(w))
print("n_neg =", np.sum(w < 0))
print("n_zero =", np.sum(w == 0))
print("n_pos =", np.sum(w > 0))
