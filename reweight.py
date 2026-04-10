import numpy as np
import zarr


BINS = np.array(
    [0, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1500, np.inf],
    dtype=float
)

SIGNAL_LABEL = "mergedHard"
BKG1_LABEL = "notElectron"
BKG2_LABEL = "notMerged"

VALID_H = (250.0, 750.0, 2000.0)
VALID_A = (1.0, 2.0, 5.0, 10.0)


def find_bin_indices(pt, bins):
    nbins = len(bins) - 1
    b = np.searchsorted(bins, pt, side="right") - 1
    return np.clip(b, 0, nbins - 1)


def build_leaf_subgroups(root, base_idx):
    label = root["label_str"][base_idx].astype(str)
    H = root["H_mass"][base_idx]
    A = root["A_mass"][base_idx]

    leaf = {}

    # signal 12개
    for h in VALID_H:
        for a in VALID_A:
            name = f"sig_H{int(h)}_A{int(a)}"
            m = (
                (label == SIGNAL_LABEL)
                & np.isclose(H, h, atol=1e-6)
                & np.isclose(A, a, atol=1e-6)
            )
            leaf[name] = base_idx[m]

    # background 2개
    leaf["bkg_notElectron"] = base_idx[(label == BKG1_LABEL)]
    leaf["bkg_notMerged"] = base_idx[(label == BKG2_LABEL)]

    return leaf


def sample_backgrounds(leaf, n_bkg=100000, seed=1234):
    rng = np.random.default_rng(seed)

    out = {}
    for name, idx in leaf.items():
        if name.startswith("sig_"):
            out[name] = idx.copy()
        else:
            if len(idx) <= n_bkg:
                out[name] = idx.copy()
                print(f"[warn] {name}: only {len(idx)} events available, use all.")
            else:
                out[name] = rng.choice(idx, size=n_bkg, replace=False)

    return out


def assign_target_leaf_total(name, parent_target=1.0):
    if name.startswith("sig_"):
        return parent_target / 12.0
    elif name in ("bkg_notElectron", "bkg_notMerged"):
        return parent_target
    else:
        raise KeyError(name)


def make_count_based_leaf_weights(idx, pt, bins, target_leaf_total):
    """
    기존 physics weight를 쓰지 않고,
    count 기반으로 pT bin equalization + leaf total normalization 수행
    """
    bin_idx = find_bin_indices(pt, bins)
    nbins = len(bins) - 1

    bin_counts = np.zeros(nbins, dtype=np.int64)
    for ib in range(nbins):
        bin_counts[ib] = np.sum(bin_idx == ib)

    nonempty = (bin_counts > 0)
    n_nonempty = int(nonempty.sum())

    if n_nonempty == 0:
        raise ValueError("Empty leaf subgroup encountered.")

    target_bin_sum = target_leaf_total / n_nonempty

    new_w = np.zeros(len(idx), dtype=np.float64)

    for ib in range(nbins):
        if not nonempty[ib]:
            continue

        m = (bin_idx == ib)
        n_bin = int(m.sum())

        # bin 안의 모든 이벤트에 동일 weight 부여
        per_event_weight = target_bin_sum / n_bin
        new_w[m] = per_event_weight

    # 검증용 bin sums
    new_bin_sums = np.zeros(nbins, dtype=np.float64)
    for ib in range(nbins):
        m = (bin_idx == ib)
        if m.any():
            new_bin_sums[ib] = new_w[m].sum()

    return new_w, {
        "n_events": len(idx),
        "bin_counts": bin_counts,
        "nonempty": nonempty,
        "n_nonempty_bins": n_nonempty,
        "target_leaf_total": float(target_leaf_total),
        "target_bin_sum": float(target_bin_sum),
        "after_total": float(new_w.sum()),
        "after_bin_sums": new_bin_sums,
        "w_min": float(new_w[new_w > 0].min()) if np.any(new_w > 0) else 0.0,
        "w_max": float(new_w.max()),
    }


def build_training_sample_and_weights_count_based(
    zarr_path,
    base_selected_idx_path,
    out_idx_path,
    out_weight_path,
    out_group_path,
    bins=BINS,
    n_bkg=100000,
    seed=1234,
    parent_target=1.0,
):
    root = zarr.open_group(zarr_path, mode="r")
    base_idx = np.load(base_selected_idx_path).astype(np.int64)

    # 1) cutflow 통과본 -> leaf subgroup
    leaf = build_leaf_subgroups(root, base_idx)

    print("[leaf counts before background sampling]")
    for name in sorted(leaf.keys()):
        print(f"{name:20s} {len(leaf[name])}")

    # 2) background 100k sampling
    sampled_leaf = sample_backgrounds(leaf, n_bkg=n_bkg, seed=seed)

    print("\n[leaf counts after background sampling]")
    for name in sorted(sampled_leaf.keys()):
        print(f"{name:20s} {len(sampled_leaf[name])}")

    # 3) count-based train weight 생성
    final_indices = []
    final_weights = []
    final_groups = []

    print("\n[reweight summary by leaf subgroup]")
    for name in sorted(sampled_leaf.keys()):
        idx = sampled_leaf[name]
        pt = root["pT"][idx].astype(np.float64)

        target_leaf_total = assign_target_leaf_total(name, parent_target=parent_target)

        new_w, info = make_count_based_leaf_weights(
            idx=idx,
            pt=pt,
            bins=bins,
            target_leaf_total=target_leaf_total,
        )

        final_indices.append(idx.astype(np.int64))
        final_weights.append(new_w.astype(np.float64))
        final_groups.append(np.array([name] * len(idx), dtype=object))

        print(
            f"{name:20s}  "
            f"N={info['n_events']:7d}  "
            f"nonempty_bins={info['n_nonempty_bins']:2d}  "
            f"target_leaf_total={info['target_leaf_total']:.8e}  "
            f"target_bin_sum={info['target_bin_sum']:.8e}  "
            f"w[min,max]=({info['w_min']:.4e}, {info['w_max']:.4e})"
        )

    train_idx = np.concatenate(final_indices)
    train_weight = np.concatenate(final_weights)
    train_group = np.concatenate(final_groups)

    np.save(out_idx_path, train_idx)
    np.save(out_weight_path, train_weight)
    np.save(out_group_path, train_group)

    # 4) 최종 검증
    g = train_group.astype(str)

    sig_mask = np.char.startswith(g, "sig_")
    b1_mask = (g == "bkg_notElectron")
    b2_mask = (g == "bkg_notMerged")

    W_sig = train_weight[sig_mask].sum()
    W_b1 = train_weight[b1_mask].sum()
    W_b2 = train_weight[b2_mask].sum()

    print("\n[final verification: parent totals]")
    print(f"signal total      = {W_sig:.8e}")
    print(f"bkg_notElectron   = {W_b1:.8e}")
    print(f"bkg_notMerged     = {W_b2:.8e}")

    print("\n[final verification: signal leaf totals]")
    for h in VALID_H:
        for a in VALID_A:
            name = f"sig_H{int(h)}_A{int(a)}"
            m = (g == name)
            print(f"{name:20s} {train_weight[m].sum():.8e}")

    print("\n[saved files]")
    print(out_idx_path)
    print(out_weight_path)
    print(out_group_path)

    return train_idx, train_weight, train_group


if __name__ == "__main__":
    build_training_sample_and_weights_count_based(
        zarr_path="/home/eoyun/data/train.zarr",
        base_selected_idx_path="/home/eoyun/data/selected_idx_cutflow_260403_v1.npy",
        out_idx_path="/home/eoyun/data/train_idx_260403_v1.npy",
        out_weight_path="/home/eoyun/data/train_weight_260403_v1.npy",
        out_group_path="/home/eoyun/data/train_group_260403_v1.npy",
        bins=BINS,
        n_bkg=100000,
        seed=1234,
        parent_target=1.0,
    )
