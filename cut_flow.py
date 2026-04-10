import zarr
import numpy as np


def isin_float(arr, allowed, atol=1e-6):
    mask = np.zeros(arr.shape, dtype=bool)
    for v in allowed:
        mask |= np.isclose(arr, v, atol=atol)
    return mask


def build_selected_index_with_cutflow_nan_bkg(
    zarr_path,
    out_npy,
    signal_label="mergedHard",
    background_labels=("notElectron", "notMerged"),   # 실제 label 이름으로 교체
    valid_H_masses=(250.0, 750.0, 2000.0),
    valid_A_masses=(1.0, 2.0, 5.0, 10.0),
    pt_cut=20.0,
):
    root = zarr.open_group(zarr_path, mode="r")

    label = root["label_str"][:].astype(str)
    H = root["H_mass"][:]
    A = root["A_mass"][:]
    pt = root["pT"][:]
    ele_id = root["IsEleCleaningID"][:]

    N = len(label)

    # -------------------------
    # 1. common cut
    # -------------------------
    common_mask = (ele_id == 0) & (pt > pt_cut)

    # -------------------------
    # 2. signal cut
    # -------------------------
    signal_mass_mask = isin_float(H, valid_H_masses) & isin_float(A, valid_A_masses)
    signal_mask = (label == signal_label) & signal_mass_mask

    # -------------------------
    # 3. background cut
    #    background는 H_mass, A_mass가 NaN이어야 함
    # -------------------------
    bkg_label_mask = np.isin(label, np.asarray(background_labels))
    bkg_mass_invalid_mask = np.isnan(H) & np.isnan(A)
    background_mask = bkg_label_mask & bkg_mass_invalid_mask

    # -------------------------
    # 4. final cut
    # -------------------------
    final_mask = common_mask & (signal_mask | background_mask)

    selected_idx = np.flatnonzero(final_mask).astype(np.int64)
    np.save(out_npy, selected_idx)

    # -------------------------
    # 5. summary
    # -------------------------
    print(f"[total] N = {N}")
    print(f"[common] pass = {common_mask.sum()}")
    print(f"[signal] pass = {(common_mask & signal_mask).sum()}")
    print(f"[bkg]    pass = {(common_mask & background_mask).sum()}")
    print(f"[final]  pass = {final_mask.sum()}")
    print(f"[save]   {out_npy}")

    print("\n[label summary after final cut]")
    unique_labels, counts = np.unique(label[final_mask], return_counts=True)
    for l, c in zip(unique_labels, counts):
        print(f"{l:20s} {c}")

    return selected_idx, final_mask

selected_idx, final_mask = build_selected_index_with_cutflow_nan_bkg(
    zarr_path="/home/eoyun/data/train.zarr",
    out_npy="/home/eoyun/data/selected_idx_cutflow_260403_v1.npy",
    signal_label="mergedHard",
    background_labels=("notElectron", "notMerged"),   # 실제 label 이름으로 교체
    valid_H_masses=(250, 750, 2000),
    valid_A_masses=(1, 2, 5, 10),
    pt_cut=20.0,
)
