import h5py

filename = "/home/eoyun/data/260403_v1/DYto2L_1J_MLM/DYto2L_1J_MLM_0.h5"

with h5py.File(filename, "r") as f:
    print("Top-level keys:", list(f.keys()))

    def show(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"[Group]   {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"[Dataset] {name} shape={obj.shape} dtype={obj.dtype}")

    f.visititems(show)
