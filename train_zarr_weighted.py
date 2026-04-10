import os
import json
import numpy as np
import zarr
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms

from transformers import AutoImageProcessor, AutoModelForImageClassification

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--out",type=str,required=True)

args = parser.parse_args()
OUTNAME = args.out

# =========================================================
# 0. paths / config
# =========================================================
ZARR_PATH = "/home/eoyun/data/train.zarr"
IDX_PATH = "/home/eoyun/data/train_idx_260403_v1.npy"
WEIGHT_PATH = "/home/eoyun/data/train_weight_260403_v1.npy"
GROUP_PATH = "/home/eoyun/data/train_group_260403_v1.npy"

# MODEL_NAME = "microsoft/resnet-50"
# MODEL_NAME = "microsoft/resnet-101"
# MODEL_NAME = "microsoft/cvt-21"
MODEL_NAME = "microsoft/swin-base-patch4-window7-224-in22k"

OUTPUT_DIR = f"./ckpts/pytorch/{OUTNAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 100
LR = 1e-5
EARLY_STOPPING_PATIENCE = 10

USE_WEIGHTED_VAL_LOSS = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["TRANSFORMERS_CACHE"] = os.getenv("SCRATCH", "/tmp") + "/hf/transformers"
os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
CACHE_PATH = os.environ["HF_HOME"]


# =========================================================
# 1. metadata / class names
# =========================================================
root_meta = zarr.open_group(ZARR_PATH, mode="r")

if "class_names" in root_meta.attrs:
    class_names = list(root_meta.attrs["class_names"])
else:
    class_names = sorted(np.unique(root_meta["label_str"][:].astype(str)).tolist())

label_to_idx = {label: i for i, label in enumerate(class_names)}
idx_to_label = {i: label for i, label in enumerate(class_names)}
num_classes = len(class_names)

print("Classes:", class_names)
print("Num classes:", num_classes)


# =========================================================
# 2. load train_idx / train_weight / train_group
# =========================================================
all_idx = np.load(IDX_PATH).astype(np.int64)
all_weight = np.load(WEIGHT_PATH).astype(np.float64)
all_group = np.load(GROUP_PATH, allow_pickle=True).astype(str)

assert len(all_idx) == len(all_weight) == len(all_group), "idx/weight/group length mismatch"

print("Total selected events:", len(all_idx))
print("Weight stats:")
print("  min =", float(all_weight.min()))
print("  max =", float(all_weight.max()))
print("  sum =", float(all_weight.sum()))

print("\nGroup counts:")
unique_groups, group_counts = np.unique(all_group, return_counts=True)
for g, c in zip(unique_groups, group_counts):
    print(f"  {g:20s} {c}")


# =========================================================
# 3. split
#    stratify by subgroup, not just by class
# =========================================================
idx_train, idx_temp, w_train, w_temp, g_train, g_temp = train_test_split(
    all_idx,
    all_weight,
    all_group,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=all_group,
)

idx_val, idx_test, w_val, w_test, g_val, g_test = train_test_split(
    idx_temp,
    w_temp,
    g_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=g_temp,
)

print(f"\nTrain size: {len(idx_train)}")
print(f"Val size:   {len(idx_val)}")
print(f"Test size:  {len(idx_test)}")


# =========================================================
# 4. image processor -> model-compatible resize / norm
# =========================================================
processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_PATH,
    local_files_only=False,
)

IMG_H = 98
IMG_W = 98
mean = processor.image_mean
std = processor.image_std

print(type(processor.size))
print(processor.size)
print(f"\nProcessor size: {IMG_H} x {IMG_W}")
print("Mean:", mean)
print("Std:", std)

train_transforms = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W), interpolation=transforms.InterpolationMode.BICUBIC),
    #transforms.RandomApply([
    #    transforms.RandomChoice([
    #        transforms.RandomAffine(degrees=0, scale=(0.6, 0.9), fill=0),
    #        transforms.RandomAffine(degrees=0, scale=(1.1, 1.4), fill=0),
    #    ])
    #], p=0.5),
    #transforms.RandomApply([transforms.RandomRotation(degrees=72)], p=0.5),
    #transforms.RandomApply([transforms.ColorJitter(brightness=0.05)], p=0.3),
    #transforms.RandomApply([transforms.ColorJitter(contrast=0.05)], p=0.5),
    #transforms.RandomApply([transforms.RandomAffine(degrees=0, fill=0)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# =========================================================
# 5. dataset
# =========================================================
class ZarrWeightedDataset(Dataset):
    def __init__(self, zarr_path, indices, sample_weights, transform=None):
        self.zarr_path = zarr_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float64)
        self.transform = transform

        self._root = None
        self._images = None
        self._label_id = None
        self._label_str = None
        self._pt = None
        self._h_mass = None

    def _ensure_open(self):
        if self._root is None:
            self._root = zarr.open_group(self.zarr_path, mode="r")
            self._images = self._root["images"]
            self._label_id = self._root["label_id"]
            self._label_str = self._root["label_str"]
            self._pt = self._root["pT"]
            self._h_mass = self._root["H_mass"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()

        real_idx = int(self.indices[idx])
        sample_weight = float(self.sample_weights[idx])

        img = np.asarray(self._images[real_idx])   # HWC, uint8
        label = int(self._label_id[real_idx])
        pt = float(self._pt[real_idx])
        h_mass = float(self._h_mass[real_idx])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, sample_weight, pt, real_idx, h_mass


train_dataset = ZarrWeightedDataset(ZARR_PATH, idx_train, w_train, transform=train_transforms)
val_dataset = ZarrWeightedDataset(ZARR_PATH, idx_val, w_val, transform=eval_transforms)
test_dataset = ZarrWeightedDataset(ZARR_PATH, idx_test, w_test, transform=eval_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)


# =========================================================
# 6. model
# =========================================================
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_PATH,
    local_files_only=False,
    num_labels=num_classes,
    ignore_mismatched_sizes=True,
    id2label=idx_to_label,
    label2id=label_to_idx,
)

model.to(DEVICE)


# =========================================================
# 7. weighted focal loss
# =========================================================
def weighted_focal_loss(
    logits,
    targets,
    sample_weights=None,
    alpha=0.25,
    gamma=2.0,
):
    """
    반환:
      loss_reduced: scalar
      loss_num: numerator for epoch aggregation
      loss_den: denominator for epoch aggregation
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    p_t = torch.exp(-ce)
    loss_per_sample = alpha * ((1.0 - p_t) ** gamma) * ce

    if sample_weights is None:
        loss_num = loss_per_sample.sum()
        loss_den = torch.tensor(loss_per_sample.numel(), device=logits.device, dtype=loss_num.dtype)
    else:
        sw = sample_weights.to(logits.device).float()
        loss_num = (loss_per_sample * sw).sum()
        loss_den = sw.sum().clamp_min(1e-12)

    loss_reduced = loss_num / loss_den
    return loss_reduced, loss_num.detach(), loss_den.detach()


# =========================================================
# 8. optimizer / scheduler / scaler
# =========================================================
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.8,
    patience=3,
)
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))


# =========================================================
# 9. train / eval functions
# =========================================================
def run_one_epoch(model, loader, optimizer=None, use_weights=True):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss_num = 0.0
    epoch_loss_den = 0.0

    all_preds = []
    all_labels = []

    for images, labels, sample_weights, pt, real_idx, h_mass in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        sample_weights = sample_weights.to(DEVICE, non_blocking=True).float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(images).logits
                loss, loss_num, loss_den = weighted_focal_loss(
                    logits,
                    labels,
                    sample_weights=sample_weights if use_weights else None,
                )

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        epoch_loss_num += float(loss_num.cpu().item())
        epoch_loss_den += float(loss_den.cpu().item())

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    epoch_loss = epoch_loss_num / max(epoch_loss_den, 1e-12)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_f1, all_labels, all_preds


# =========================================================
# 10. train loop
# =========================================================
metrics_history = {
    "train_loss": [],
    "val_loss": [],
    "train_f1": [],
    "val_f1": [],
}

best_val_f1 = -np.inf
epochs_no_improve = 0

best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
config_path = os.path.join(OUTPUT_DIR, "run_config.json")

with open(config_path, "w") as f:
    json.dump({
        "zarr_path": ZARR_PATH,
        "idx_path": IDX_PATH,
        "weight_path": WEIGHT_PATH,
        "group_path": GROUP_PATH,
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": EPOCHS,
        "lr": LR,
        "random_state": RANDOM_STATE,
        "class_names": class_names,
    }, f, indent=2)

print("\nStart training...\n")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_f1, _, _ = run_one_epoch(
        model,
        train_loader,
        optimizer=optimizer,
        use_weights=True,
    )

    val_loss, val_f1, _, _ = run_one_epoch(
        model,
        val_loader,
        optimizer=None,
        use_weights=USE_WEIGHTED_VAL_LOSS,
    )

    metrics_history["train_loss"].append(train_loss)
    metrics_history["val_loss"].append(val_loss)
    metrics_history["train_f1"].append(train_f1)
    metrics_history["val_f1"].append(val_f1)

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss={train_loss:.6f}, Train F1={train_f1:.4f} | "
        f"Val Loss={val_loss:.6f}, Val F1={val_f1:.4f}"
    )

    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "class_names": class_names,
            "model_name": MODEL_NAME,
        }, best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

print(f"\nBest Val F1 = {best_val_f1:.4f}")


# =========================================================
# 11. test evaluation
# =========================================================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize


PT_BINS = np.array(
    [0, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1500, np.inf],
    dtype=float
)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def predict_on_loader(model, loader):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []
    y_pt = []
    y_h_mass = []
    y_weight = []
    y_real_idx = []

    with torch.no_grad():
        for images, labels, sample_weights, pt, real_idx, h_mass in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(images).logits
                probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())
            y_prob.append(probs.detach().cpu().numpy())
            y_pt.append(to_numpy(pt))
            y_h_mass.append(to_numpy(h_mass))
            y_weight.append(to_numpy(sample_weights))
            y_real_idx.append(to_numpy(real_idx))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    y_pt = np.concatenate(y_pt).astype(np.float64)
    y_h_mass = np.concatenate(y_h_mass).astype(np.float64)
    y_weight = np.concatenate(y_weight).astype(np.float64)
    y_real_idx = np.concatenate(y_real_idx).astype(np.int64)

    return y_true, y_pred, y_prob, y_pt, y_weight, y_real_idx, y_h_mass


def plot_multiclass_roc(y_true, y_prob, class_names, save_path):
    num_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    plt.figure(figsize=(7, 7))
    any_plotted = False

    for i, name in enumerate(class_names):
        y_true_i = y_true_bin[:, i]

        # positive 또는 negative 한쪽만 있으면 ROC 불가
        if y_true_i.max() == 0 or y_true_i.min() == 1:
            print(f"[ROC] skip '{name}': only one class present.")
            continue

        fpr, tpr, _ = roc_curve(y_true_i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        any_plotted = True

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve")
    if any_plotted:
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def compute_binned_fraction(values, denom_mask, numer_mask, bins):
    nbins = len(bins) - 1
    frac = np.full(nbins, np.nan, dtype=np.float64)
    numer = np.zeros(nbins, dtype=np.int64)
    denom = np.zeros(nbins, dtype=np.int64)

    b = np.searchsorted(bins, values, side="right") - 1
    b = np.clip(b, 0, nbins - 1)

    for ib in range(nbins):
        m_bin = (b == ib)

        d = denom_mask & m_bin
        n = numer_mask & m_bin

        denom[ib] = int(d.sum())
        numer[ib] = int(n.sum())

        if denom[ib] > 0:
            frac[ib] = numer[ib] / denom[ib]

    return frac, numer, denom


def plot_pt_eff_and_fake_rate(
    y_true,
    y_pred,
    y_pt,
    y_h_mass,
    signal_idx,
    class_names,
    bins,
    save_path,
):
    centers = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if np.isinf(hi):
            centers.append(lo + 0.5 * (lo - bins[i - 1]))
        else:
            centers.append(0.5 * (lo + hi))
    centers = np.asarray(centers, dtype=np.float64)

    true_signal = (y_true == signal_idx)
    pred_signal = (y_pred == signal_idx)
    true_background = (y_true != signal_idx)

    # signal efficiency = true signal 중 signal로 맞춘 비율
    sig_eff, sig_num, sig_den = compute_binned_fraction(
        y_pt,
        denom_mask=true_signal,
        numer_mask=(true_signal & pred_signal),
        bins=bins,
    )

    # background fake rate = true background 중 signal로 잘못 예측한 비율
    bkg_fake, bkg_num, bkg_den = compute_binned_fraction(
        y_pt,
        denom_mask=true_background,
        numer_mask=(true_background & pred_signal),
        bins=bins,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    sig_err = np.sqrt(sig_eff * (1 - sig_eff) / sig_den)
    bkg_err = np.sqrt(bkg_fake * (1 - bkg_fake) / bkg_den)

    axes[0].errorbar(centers, sig_eff, yerr=sig_err, fmt="o", linestyle="none", capsize=4)
    axes[1].errorbar(centers, bkg_fake, yerr=bkg_err, fmt="o", linestyle="none", capsize=4)

    axes[0].set_ylabel("Signal Efficiency")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"pT vs Signal Efficiency ({class_names[signal_idx]})")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("pT [GeV]")
    axes[1].set_ylabel("Background Fake Rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"pT vs Background Fake Rate (predicted as {class_names[signal_idx]})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return {
        "sig_eff": sig_eff,
        "sig_num": sig_num,
        "sig_den": sig_den,
        "bkg_fake": bkg_fake,
        "bkg_num": bkg_num,
        "bkg_den": bkg_den,
        "centers": centers,
    }


def plot_confusion_matrix_figure(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(cm, aspect="auto")
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xticks(np.arange(len(class_names)))
    axes[0].set_yticks(np.arange(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, f"{cm[i, j]}", ha="center", va="center")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cm_norm, aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_title("Confusion Matrix (Row-normalized)")
    axes[1].set_xticks(np.arange(len(class_names)))
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_history(metrics_history, save_path):
    epochs_range = range(1, len(metrics_history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, metrics_history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, metrics_history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, metrics_history["train_f1"], label="Train F1")
    plt.plot(epochs_range, metrics_history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# =========================================================
# 11. test evaluation + plots
# =========================================================
ckpt = torch.load(best_model_path, map_location="cpu")

best_model = AutoModelForImageClassification.from_pretrained(
    ckpt["model_name"],
    cache_dir=CACHE_PATH,
    local_files_only=False,
    num_labels=num_classes,
    ignore_mismatched_sizes=True,
    id2label=idx_to_label,
    label2id=label_to_idx,
)
best_model.load_state_dict(ckpt["model_state_dict"])
best_model.to(DEVICE)
best_model.eval()

test_loss, test_f1, y_true_epoch, y_pred_epoch = run_one_epoch(
    best_model,
    test_loader,
    optimizer=None,
    use_weights=USE_WEIGHTED_VAL_LOSS,
)

y_true, y_pred, y_prob, y_pt, y_weight, y_real_idx, y_h_mass = predict_on_loader(best_model, test_loader)

print("\n[Test]")
print(f"Test Loss = {test_loss:.6f}")
print(f"Test F1   = {test_f1:.4f}")

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

signal_class_name = "mergedHard"
signal_idx = label_to_idx[signal_class_name]

# save raw arrays
np.save(os.path.join(RESULTS_DIR, "y_true_test.npy"), y_true)
np.save(os.path.join(RESULTS_DIR, "y_pred_test.npy"), y_pred)
np.save(os.path.join(RESULTS_DIR, "y_prob_test.npy"), y_prob)
np.save(os.path.join(RESULTS_DIR, "y_pt_test.npy"), y_pt)
np.save(os.path.join(RESULTS_DIR, "y_h_mass_test.npy"), y_h_mass)
np.save(os.path.join(RESULTS_DIR, "y_weight_test.npy"), y_weight)
np.save(os.path.join(RESULTS_DIR, "y_real_idx_test.npy"), y_real_idx)

# save per-event prediction table
import pandas as pd
df_pred = pd.DataFrame({
    "real_idx": y_real_idx,
    "pT": y_pt,
    "h_mass": y_h_mass,
    "sample_weight": y_weight,
    "y_true_idx": y_true,
    "y_pred_idx": y_pred,
    "y_true_label": [idx_to_label[i] for i in y_true],
    "y_pred_label": [idx_to_label[i] for i in y_pred],
    "prob_signal": y_prob[:, signal_idx],
})
df_pred.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

# plots
plot_training_history(
    metrics_history,
    save_path=os.path.join(RESULTS_DIR, "training_history.png"),
)

plot_multiclass_roc(
    y_true=y_true,
    y_prob=y_prob,
    class_names=class_names,
    save_path=os.path.join(RESULTS_DIR, "roc_curve.png"),
)

pt_summary = plot_pt_eff_and_fake_rate(
    y_true=y_true,
    y_pred=y_pred,
    y_pt=y_pt,
    y_h_mass = y_h_mass,
    signal_idx=signal_idx,
    class_names=class_names,
    bins=PT_BINS,
    save_path=os.path.join(RESULTS_DIR, "pt_efficiency_fake_rate.png"),
)

plot_confusion_matrix_figure(
    y_true=y_true,
    y_pred=y_pred,
    class_names=class_names,
    save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
)

# save pT-binned summary
bin_labels = []
for i in range(len(PT_BINS) - 1):
    lo, hi = PT_BINS[i], PT_BINS[i + 1]
    if np.isinf(hi):
        bin_labels.append(f"[{lo}, inf)")
    else:
        bin_labels.append(f"[{lo}, {hi})")

df_pt = pd.DataFrame({
    "pt_bin": bin_labels,
    "pt_center": pt_summary["centers"],
    "signal_numer": pt_summary["sig_num"],
    "signal_denom": pt_summary["sig_den"],
    "signal_efficiency": pt_summary["sig_eff"],
    "bkg_numer": pt_summary["bkg_num"],
    "bkg_denom": pt_summary["bkg_den"],
    "bkg_fake_rate": pt_summary["bkg_fake"],
})
df_pt.to_csv(os.path.join(RESULTS_DIR, "pt_efficiency_fake_rate.csv"), index=False)

print(f"\nSaved plots and tables to: {RESULTS_DIR}")


# =========================================================
# 12. save metrics / predictions
# =========================================================
np.save(os.path.join(OUTPUT_DIR, "train_loss.npy"), np.array(metrics_history["train_loss"]))
np.save(os.path.join(OUTPUT_DIR, "val_loss.npy"), np.array(metrics_history["val_loss"]))
np.save(os.path.join(OUTPUT_DIR, "train_f1.npy"), np.array(metrics_history["train_f1"]))
np.save(os.path.join(OUTPUT_DIR, "val_f1.npy"), np.array(metrics_history["val_f1"]))

np.save(os.path.join(OUTPUT_DIR, "y_true_test.npy"), y_true)
np.save(os.path.join(OUTPUT_DIR, "y_pred_test.npy"), y_pred)

print(f"\nSaved outputs to: {OUTPUT_DIR}")
