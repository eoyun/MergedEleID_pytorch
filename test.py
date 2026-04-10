import os
import glob
import h5py
import zarr
import numpy as np


# =========================
# HDF5 dataset keys
# =========================
IMAGE_KEY = "images"
LABEL_KEY = "Label"
SOURCE_KEY = "source_file"

NUMERIC_KEYS = [
    "A_mass",
    "E5x5",
    "H_mass",
    "IsAddTrk",
    "IsEleCleaningID",
    "NumEle",
    "NumEleHard",
    "dEta",
    "dPhi",
    "dR",
    "pT",
    "weight",
]


# =========================
# utility
# =========================
def find_h5_files(root_dir, patterns=("*.h5", "*.hdf5")):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root_dir, "**", pat), recursive=True))
    return sorted(files)


def decode_object_array(arr):
    """
    HDF5 object/string dataset -> Python str list
    """
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        elif isinstance(x, np.bytes_):
            out.append(x.decode("utf-8"))
        elif x is None:
            out.append("")
        else:
            out.append(str(x))
    return out


def to_fixed_unicode_array(values, min_len=1):
    """
    list[str] -> numpy fixed-length unicode array
    """
    max_len = max([len(v) for v in values], default=min_len)
    max_len = max(max_len, min_len)
    return np.asarray(values, dtype=f"<U{max_len}"), max_len


def infer_dtype_for_key(key):
    """
    네 스키마 기준 명시적 dtype
    """
    if key in ["A_mass", "E5x5", "H_mass", "dEta", "dPhi", "dR", "pT", "weight"]:
        return "f4"
    if key in ["IsAddTrk", "IsEleCleaningID", "NumEle", "NumEleHard"]:
        return "i4"
    raise KeyError(f"Unknown numeric key: {key}")


def inspect_files(file_list):
    total_n = 0
    image_shape = None
    image_dtype = None

    all_label_values = set()
    max_source_len = 1
    max_label_len = 1

    for path in file_list:
        with h5py.File(path, "r") as f:
            x = f[IMAGE_KEY]
            n = x.shape[0]

            if image_shape is None:
                image_shape = x.shape[1:]
                image_dtype = x.dtype
            else:
                if image_shape != x.shape[1:]:
                    raise ValueError(
                        f"Image shape mismatch in {path}: {x.shape[1:]} vs {image_shape}"
                    )
                if image_dtype != x.dtype:
                    raise ValueError(
                        f"Image dtype mismatch in {path}: {x.dtype} vs {image_dtype}"
                    )

            # numeric 길이 검사
            for key in NUMERIC_KEYS:
                if f[key].shape[0] != n:
                    raise ValueError(f"Length mismatch: {path} key={key}")

            # label / source_file 길이 검사
            if f[LABEL_KEY].shape[0] != n:
                raise ValueError(f"Length mismatch: {path} key={LABEL_KEY}")
            if f[SOURCE_KEY].shape[0] != n:
                raise ValueError(f"Length mismatch: {path} key={SOURCE_KEY}")

            # label scan
            labels = decode_object_array(f[LABEL_KEY][:])
            sources = decode_object_array(f[SOURCE_KEY][:])

            for s in labels:
                all_label_values.add(s)
                if len(s) > max_label_len:
                    max_label_len = len(s)

            for s in sources:
                if len(s) > max_source_len:
                    max_source_len = len(s)

            total_n += n

    label_names = sorted(all_label_values)
    label_to_id = {name: i for i, name in enumerate(label_names)}

    return {
        "total_n": total_n,
        "image_shape": image_shape,
        "image_dtype": image_dtype,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "max_label_len": max_label_len,
        "max_source_len": max_source_len,
    }


def build_zarr_from_hdf5(
    input_root,
    zarr_path,
    image_chunk_n=256,
    meta_chunk_n=8192,
    compressor_level=3,
):
    file_list = find_h5_files(input_root)
    if len(file_list) == 0:
        raise RuntimeError(f"No HDF5 files found under: {input_root}")

    info = inspect_files(file_list)

    N = info["total_n"]
    image_shape = info["image_shape"]
    image_dtype = info["image_dtype"]
    label_names = info["label_names"]
    label_to_id = info["label_to_id"]
    max_label_len = info["max_label_len"]
    max_source_len = info["max_source_len"]

    print(f"[info] files        = {len(file_list)}")
    print(f"[info] total_n      = {N}")
    print(f"[info] image_shape  = {image_shape}")
    print(f"[info] image_dtype  = {image_dtype}")
    print(f"[info] n_classes    = {len(label_names)}")
    print(f"[info] label_names  = {label_names}")

    root = zarr.open_group(zarr_path, mode="w", zarr_format=3)

    comp = zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=compressor_level,
        shuffle=zarr.codecs.BloscShuffle.shuffle,
    )

    # main image
    images = root.create_array(
        name="images",
        shape=(N, *image_shape),
        chunks=(image_chunk_n, *image_shape),
        dtype=image_dtype,
        compressors=comp,
    )

    # numeric arrays
    arrays = {}
    for key in NUMERIC_KEYS:
        arrays[key] = root.create_array(
            name=key,
            shape=(N,),
            chunks=(meta_chunk_n,),
            dtype=infer_dtype_for_key(key),
            compressors=comp,
        )

    # label string / label id
    label_str = root.create_array(
        name="label_str",
        shape=(N,),
        chunks=(meta_chunk_n,),
        dtype=np.dtype(f"<U{max_label_len}"),
        compressors=comp,
    )

    label_id = root.create_array(
        name="label_id",
        shape=(N,),
        chunks=(meta_chunk_n,),
        dtype="i4",
        compressors=comp,
    )

    # source_file string
    source_file = root.create_array(
        name="source_file",
        shape=(N,),
        chunks=(meta_chunk_n,),
        dtype=np.dtype(f"<U{max_source_len}"),
        compressors=comp,
    )

    # global attrs
    root.attrs["source_root"] = os.path.abspath(input_root)
    root.attrs["num_source_files"] = len(file_list)
    root.attrs["total_samples"] = int(N)
    root.attrs["class_names"] = label_names
    root.attrs["hdf5_image_key"] = IMAGE_KEY
    root.attrs["hdf5_label_key"] = LABEL_KEY
    root.attrs["hdf5_source_key"] = SOURCE_KEY
    root.attrs["numeric_keys"] = NUMERIC_KEYS

    start = 0
    for i, path in enumerate(file_list):
        with h5py.File(path, "r") as f:
            x = f[IMAGE_KEY][:]
            n = x.shape[0]
            stop = start + n

            images[start:stop] = x

            for key in NUMERIC_KEYS:
                arrays[key][start:stop] = f[key][:]

            labels_py = decode_object_array(f[LABEL_KEY][:])
            sources_py = decode_object_array(f[SOURCE_KEY][:])

            labels_np, _ = to_fixed_unicode_array(labels_py, min_len=max_label_len)
            sources_np, _ = to_fixed_unicode_array(sources_py, min_len=max_source_len)

            label_ids_np = np.asarray([label_to_id[s] for s in labels_py], dtype=np.int32)

            label_str[start:stop] = labels_np
            label_id[start:stop] = label_ids_np
            source_file[start:stop] = sources_np

        print(f"[write] {i+1:5d}/{len(file_list):5d}  n={n:6d}  {path}")
        start = stop

    print(f"[done] wrote Zarr store: {zarr_path}")
    
if __name__ == "__main__":
    build_zarr_from_hdf5(
        input_root="/home/eoyun/data/260403_v1",
        zarr_path="/home/eoyun/data/train.zarr",
        image_chunk_n=256,
        meta_chunk_n=8192,
        compressor_level=3,
    )
