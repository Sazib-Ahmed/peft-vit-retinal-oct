#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeiT (ViT) + LoRA (FROZEN BACKBONE) Pipeline for Retinal OCT Classification
==========================================================================
Backbone:
  - deit_small_distilled_patch16_224 (timm)

Adaptation:
  - LoRA injected into attention qkv ("qkv" in timm DeiT blocks)
  - Frozen backbone: only LoRA params + classification head(s) train

Datasets:
  - Folder splits: root/{train,val,test}/{class_name}/*.(png|jpg|jpeg|bmp|tif|tiff|webp)

Geometry:
  - Optional strict Pad-to-Square (recommended for wide-field scans like C8)

Paper-grade evaluation (saved to run folder):
  * Accuracy, Macro-F1
  * ROC curves + per-class AUC + Macro-AUC (OVR)
  * Confusion matrix (normalized)
  * Classification report heatmap
  * Sensitivity & Specificity per class
  * Calibration: ECE, NLL, Brier + Reliability diagram
  * Optional Temperature scaling (val-set) for publication-grade calibration
  * t-SNE of DeiT features (CLS token)
  * Efficiency: trainable params, % trainable, adapter-only checkpoint size,
               inference latency/throughput, peak VRAM (train + infer)

Class imbalance (paper-minimum handling):
  - Supports extreme imbalance (e.g., one class ~13 images vs others thousands)
  - Implements BOTH (configurable):
      (1) WeightedRandomSampler for training loader (oversample rare classes)
      (2) Class-weighted CrossEntropyLoss (inverse-frequency weights, clipped)
    Default is "sampler+loss". You can switch to only one via CONFIG["imbalance"]["strategy"].

Requirements:
  pip install timm peft scikit-learn pandas matplotlib seaborn opencv-python

Author: (fill in)
Paper:  "Parameter-Efficient Adaptation of Vision Transformers for Retinal OCT Classification"

Notes on logging/progress bars:
  - If you redirect stdout/stderr to a file, classic tqdm progress bars will look "broken"
    (many lines) because files are not TTYs.
  - This script auto-disables tqdm bars when not running in a real terminal (isatty==False)
    and prints clean periodic progress lines instead.
"""

# ==========================================
# 0. IMPORTS
# ==========================================
import os
import time
import math
import json
import random
import warnings
import gc
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Iterable, Iterator

import cv2
cv2.setNumThreads(0)  # IMPORTANT for Windows + DataLoader workers

import numpy as np
import pandas as pd
import torch
import timm

import matplotlib
matplotlib.use("Agg")  # must be BEFORE pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    f1_score,
    roc_auc_score,
)
from sklearn.manifold import TSNE

from peft import get_peft_model, LoraConfig

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ==========================================
# 1. USER SETTINGS (EDIT THESE)
# ==========================================
DATASET_CONFIGS: Dict[str, Dict] = {
    # Example (edit roots):
    "C8": {
        "root": r"D:\AIUB\DSP\Code\Datasets\C8\RetinalOCT_Dataset_CLEAN",
        "pad_to_square": True,
        "cache_in_ram": False,
    },
    "NEH_UT_2021": {
        "root": r"D:\AIUB\DSP\Code\Datasets\NEH_UT_2021RetinalOCTDataset\NEH_UT_2021_splits_clean",
        "pad_to_square": True,
        "cache_in_ram": False,  # recommended if very large
    },
    "OCTDL": {
        "root": r"D:\AIUB\DSP\Code\Datasets\OCTDL\OCTDL_SPLIT_BY_PATIENT",
        "pad_to_square": True,
        "cache_in_ram": False,
    },
}

# Choose which datasets to run (order matters)
RUN_DATASETS = ["NEH_UT_2021", "C8"]

# ==========================================
# 2. GLOBAL CONFIG
# ==========================================
CONFIG = {
    # --- Repro / device ---
    "seed": 42,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "amp_enabled": True,

    # DataLoader
    "num_workers": 4,          # Windows: start with 2–4
    "pin_memory": True,

    # --- Model ---
    "model_name": "deit_small_distilled_patch16_224",
    "img_size": 224,

    # --- Training ---
    "batch_size": 256,
    "epochs": 2,
    "lr": 5e-4,
    "weight_decay": 1e-2,
    "warmup_epochs": 1,
    "patience": 10,
    "label_smoothing": 0.1,

    # --- LoRA (Frozen backbone) ---
    "freeze_backbone": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,

    # --- Class imbalance handling (paper-minimum) ---
    "imbalance": {
        "enabled": True,
        # "none" | "loss" | "sampler" | "sampler+loss"
        "strategy": "sampler+loss",
        # inverse-frequency weights can explode; clip to keep training stable
        "max_class_weight": 20.0,
        # for sampler
        "sampler_replacement": True,
    },

    # --- Calibration ---
    "use_temperature_scaling": True,
    "ece_bins": 15,

    # --- Inference benchmark ---
    "bench_merge_lora": True,        # merge LoRA before timing (fairness)
    "bench_warmup_iters": 50,
    "bench_timed_iters": 300,
    "bench_latency_batch": 1,
    "bench_throughput_batch": 64,

    # --- Logging / progress ---
    "log_progress_every": 50,  # when not TTY, print every N steps
}

# ==========================================
# 2.0 LOGGING + PROGRESS (TTY-SAFE)
# ==========================================
def is_tty() -> bool:
    try:
        return sys.stdout.isatty() and sys.stderr.isatty()
    except Exception:
        return False

def log(msg: str):
    print(msg, flush=True)

def progress_iter(iterable: Iterable, desc: str, total: Optional[int] = None) -> Iterator:
    """
    - If interactive TTY: use tqdm progress bar.
    - If redirected to file: print clean periodic progress lines.
    """
    if is_tty():
        yield from tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False)
        return

    # Non-TTY (e.g., stdout redirected): periodic logs instead of tqdm spam
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None

    t0 = time.time()
    for i, item in enumerate(iterable, start=1):
        if (i == 1) or (CONFIG["log_progress_every"] > 0 and (i % CONFIG["log_progress_every"] == 0)) or (total is not None and i == total):
            dt = time.time() - t0
            if total is not None and i > 0:
                rate = i / max(dt, 1e-9)
                eta = (total - i) / max(rate, 1e-9)
                log(f"[{desc}] {i}/{total} ({100.0*i/total:.1f}%) | {rate:.2f} it/s | ETA {eta/60:.1f} min")
            else:
                log(f"[{desc}] {i} it | {i/max(dt,1e-9):.2f} it/s")
        yield item

# ==========================================
# 2.1 REPRODUCIBILITY (SPAWN-SAFE)
# ==========================================
def set_seeds():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    if CONFIG["device"].type == "cuda":
        torch.cuda.manual_seed_all(CONFIG["seed"])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def worker_init_fn(worker_id: int):
    # Ensure each worker has a distinct but deterministic seed
    base_seed = CONFIG["seed"]
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)

# ==========================================
# 2.2 GPU AUGMENTATION + GPU NORMALIZATION  (SPAWN-SAFE)
# ==========================================
# IMPORTANT:
# - Do NOT create CUDA modules at import time.
# - Create lazily inside main process.

_data_augmentation_gpu = None

normalize_gpu = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

def get_data_augmentation_gpu():
    global _data_augmentation_gpu
    if _data_augmentation_gpu is None:
        # NOTE: In many torchvision versions transforms are nn.Module-compatible and can be in nn.Sequential.
        # If your torchvision build does not support that, switch to CPU-side transforms or kornia.
        _data_augmentation_gpu = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.uint8),
            transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0),
        ).to(CONFIG["device"])
    return _data_augmentation_gpu

def apply_train_gpu_aug_and_norm(x: torch.Tensor) -> torch.Tensor:
    # x: (B,C,H,W) uint8 on device
    with torch.no_grad():
        aug = get_data_augmentation_gpu()
        x = aug(x)              # float32 in [0,1]
        x = normalize_gpu(x)
    return x

def apply_eval_gpu_norm(x: torch.Tensor) -> torch.Tensor:
    # x: (B,C,H,W) uint8 on device
    x = x.float().div_(255.0)
    return normalize_gpu(x)

# ==========================================
# 3. UTILS: PAD-TO-SQUARE + DATASET
# ==========================================
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def pad_to_square_np(image: np.ndarray) -> np.ndarray:
    """Pads an HxWxC image to square with black borders."""
    h, w = image.shape[:2]
    if h == w:
        return image
    diff = abs(h - w)
    pad_1 = diff // 2
    pad_2 = diff - pad_1
    if h > w:
        padding = ((0, 0), (pad_1, pad_2), (0, 0))
    else:
        padding = ((pad_1, pad_2), (0, 0), (0, 0))
    return np.pad(image, padding, mode="constant", constant_values=0)

def list_images_by_class(split_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """Returns (paths, labels, class_names) where label indices follow sorted class_names."""
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths, labels = [], []
    for c in class_names:
        cdir = os.path.join(split_dir, c)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            p = os.path.join(cdir, fn)
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in VALID_EXTS:
                continue
            paths.append(p)
            labels.append(class_to_idx[c])
    return paths, labels, class_names

class OCTFolderDataset(Dataset):
    """
    Folder-based dataset with optional RAM caching.
    Expected structure:
      root/train/<class>/*.png|jpg|...
      root/val/<class>/*.png|jpg|...
      root/test/<class>/*.png|jpg|...
    """
    def __init__(
        self,
        root_dir: str,
        split: str,
        img_size: int,
        transform=None,
        pad_to_square: bool = False,
        cache_in_ram: bool = False,
        class_names: Optional[List[str]] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform  # kept for compatibility (not used now)
        self.pad_to_square = pad_to_square
        self.cache_in_ram = cache_in_ram

        split_dir = os.path.join(root_dir, split)

        # Ensure consistent class order across splits (use train split as source of truth)
        if class_names is None:
            _, _, class_names = list_images_by_class(split_dir)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Build path list (in class order, sorted filenames)
        self.paths = []
        self.labels = []
        for c in self.class_names:
            cdir = os.path.join(split_dir, c)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                p = os.path.join(cdir, fn)
                if not os.path.isfile(p):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext not in VALID_EXTS:
                    continue
                self.paths.append(p)
                self.labels.append(self.class_to_idx[c])

        self.cached_images = None
        if self.cache_in_ram:
            self.cached_images = []
            log(f"[INFO] RAM-caching: loading {len(self.paths)} images for split='{split}' ...")
            for p in progress_iter(self.paths, desc=f"Caching {split}", total=len(self.paths)):
                img = cv2.imread(p)
                if img is None:
                    self.cached_images.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.pad_to_square:
                    img = pad_to_square_np(img)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                self.cached_images.append(img)
            log(f"[SUCCESS] RAM-cached {len(self.cached_images)} images for '{split}'.")

    def __len__(self):
        return len(self.paths)

    def _load_image(self, idx: int) -> np.ndarray:
        if self.cache_in_ram and self.cached_images is not None:
            return self.cached_images[idx]

        p = self.paths[idx]
        img = cv2.imread(p)
        if img is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.pad_to_square:
            img = pad_to_square_np(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx: int):
        # Faster + spawn-safe: no PIL, no ToTensor per sample
        img = self._load_image(idx)  # np.uint8 HWC
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint8 CHW
        label = int(self.labels[idx])
        return x, label

def build_transforms():
    # Kept for structure compatibility; transforms not needed now
    return None, None

def compute_class_weights_from_labels(labels: List[int], num_classes: int, max_w: float) -> np.ndarray:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    # Standard inverse frequency weight normalized to mean=1
    inv = counts.sum() / (num_classes * counts)
    inv = np.clip(inv, 0.0, max_w)
    return inv.astype(np.float32)

def build_dataloaders(dataset_name: str, ds_cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], torch.Tensor]:
    root = ds_cfg["root"]
    pad_sq = ds_cfg.get("pad_to_square", False)
    cache = ds_cfg.get("cache_in_ram", False)

    train_tf, eval_tf = build_transforms()

    # Derive class order from train split
    train_split_dir = os.path.join(root, "train")
    _, _, class_names = list_images_by_class(train_split_dir)

    train_ds = OCTFolderDataset(root, "train", CONFIG["img_size"], train_tf, pad_sq, cache, class_names)
    val_ds   = OCTFolderDataset(root, "val",   CONFIG["img_size"], eval_tf,  pad_sq, cache, class_names)
    test_ds  = OCTFolderDataset(root, "test",  CONFIG["img_size"], eval_tf,  pad_sq, cache, class_names)

    # Class weights (from TRAIN split) for imbalance handling
    num_classes = len(class_names)
    class_w_np = compute_class_weights_from_labels(
        train_ds.labels,
        num_classes=num_classes,
        max_w=float(CONFIG["imbalance"]["max_class_weight"]),
    )
    class_w_t = torch.tensor(class_w_np, dtype=torch.float32, device=CONFIG["device"])

    # Print dataset summary + imbalance stats
    counts = np.bincount(np.array(train_ds.labels, dtype=np.int64), minlength=num_classes)
    log(f"\n[DATASET] {dataset_name}")
    log(f" - root: {root}")
    log(f" - pad_to_square: {pad_sq}")
    log(f" - cache_in_ram: {cache}")
    log(f" - classes ({len(class_names)}): {class_names}")
    log(f" - split sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    log(f" - train class counts: {counts.tolist()}")
    if CONFIG["imbalance"]["enabled"]:
        log(f" - imbalance strategy: {CONFIG['imbalance']['strategy']}")
        log(f" - class weights (inv-freq, clipped@{CONFIG['imbalance']['max_class_weight']}): {class_w_np.tolist()}")

    # Windows optimization: persistent workers + prefetch only when num_workers > 0
    nw = int(CONFIG["num_workers"])
    dl_kwargs = dict(
        num_workers=nw,
        pin_memory=bool(CONFIG["pin_memory"]),
        worker_init_fn=worker_init_fn,
    )
    if nw > 0:
        dl_kwargs.update(dict(
            persistent_workers=True,
            prefetch_factor=4,
        ))

    # WeightedRandomSampler (oversample minority classes)
    sampler = None
    use_sampler = CONFIG["imbalance"]["enabled"] and ("sampler" in CONFIG["imbalance"]["strategy"])
    if use_sampler:
        sample_weights = class_w_np[np.array(train_ds.labels, dtype=np.int64)]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=bool(CONFIG["imbalance"]["sampler_replacement"]),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=(sampler is None),   # don't shuffle if using sampler
        sampler=sampler,
        **dl_kwargs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        **dl_kwargs
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        **dl_kwargs
    )

    return train_loader, val_loader, test_loader, class_names, class_w_t

# ==========================================
# 4. MODEL: TIMM + PEFT LoRA (FROZEN)
# ==========================================
class TimmVisionWrapper(nn.Module):
    """Bridge wrapper for PEFT -> timm vision model."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        if input_ids is not None:
            x = input_ids
        elif pixel_values is not None:
            x = pixel_values
        else:
            x = kwargs.get("x", None)
            if x is None:
                raise ValueError("No input tensor provided.")
        return self.model(x)

    def forward_features(self, x):
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(x)
        raise AttributeError("Underlying timm model has no forward_features()")

def freeze_backbone_keep_lora_and_heads(model: nn.Module):
    """
    Enforce 'LoRA frozen backbone' strictly:
      - freeze everything
      - unfreeze LoRA params + classification head(s)
    """
    for _, p in model.named_parameters():
        p.requires_grad = False

    # Unfreeze LoRA
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    # Unfreeze heads (distilled DeiT has head and head_dist)
    head_keywords = [".head.", ".head_dist.", "head.", "head_dist.", "modules_to_save"]
    for n, p in model.named_parameters():
        if any(k in n for k in head_keywords):
            p.requires_grad = True

def count_params(model: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    return total, trainable, pct

def create_deit_lora_model(num_classes: int) -> nn.Module:
    log(f"\n[MODEL] Creating {CONFIG['model_name']} + LoRA (freeze_backbone={CONFIG['freeze_backbone']})")
    base_model = timm.create_model(
        CONFIG["model_name"],
        pretrained=True,
        num_classes=num_classes,
    )
    wrapped = TimmVisionWrapper(base_model)

    modules_to_save = ["head"]
    if hasattr(base_model, "head_dist"):
        modules_to_save.append("head_dist")

    lora_cfg = LoraConfig(
        r=int(CONFIG["lora_r"]),
        lora_alpha=int(CONFIG["lora_alpha"]),
        target_modules=["qkv"],
        lora_dropout=float(CONFIG["lora_dropout"]),
        bias="none",
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(wrapped, lora_cfg).to(CONFIG["device"])

    if CONFIG["freeze_backbone"]:
        freeze_backbone_keep_lora_and_heads(model)

    total, trainable, pct = count_params(model)
    log(f"[PARAMS] total={total:,} trainable={trainable:,} ({pct:.3f}%)")
    return model

def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Handle distilled DeiT outputs safely:
      - if tuple/list -> average logits for prediction
      - else -> direct logits
    """
    out = model(x)
    if isinstance(out, (tuple, list)):
        return (out[0] + out[1]) / 2.0
    return out

# ==========================================
# 5. METRICS: CALIBRATION + MEDICAL
# ==========================================
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi)
        if not np.any(m):
            continue
        ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)

def compute_brier(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def compute_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1.0)
    return float((-np.log(p[np.arange(len(labels)), labels])).mean())

def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int, title: str, save_path: str):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    accs, confs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi)
        if np.any(m):
            accs.append(correct[m].mean())
            confs.append(conf[m].mean())
        else:
            accs.append(0.0)
            confs.append((lo + hi) / 2.0)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.bar(centers, accs, width=1.0 / n_bins, alpha=0.7, edgecolor="black", label="Empirical")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def sensitivity_specificity_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = cm.shape[0]
    sens, spec = [], []
    for i in range(k):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sens.append(tp / (tp + fn + 1e-12))
        spec.append(tn / (tn + fp + 1e-12))
    return np.array(sens), np.array(spec)

def plot_sens_spec(cm: np.ndarray, class_names: List[str], title: str, save_path: str):
    sens, spec = sensitivity_specificity_from_cm(cm)
    x = np.arange(len(class_names))
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, sens, width=0.4, label="Sensitivity (Recall)")
    plt.bar(x + 0.2, spec, width=0.4, label="Specificity")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion(cm: np.ndarray, class_names: List[str], title: str, save_path: str):
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype(float), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Normalized Frequency"})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_classification_report_heatmap(report_str: str, title: str, save_path: str):
    lines = report_str.split("\n")
    rows = []
    for line in lines[2:-5]:
        parts = [p for p in line.split(" ") if p]
        if len(parts) >= 4:
            rows.append([parts[0], float(parts[1]), float(parts[2]), float(parts[3])])
    if not rows:
        log("[WARN] Could not parse classification report for heatmap.")
        return
    df = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1"]).set_index("class")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".3f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(labels: np.ndarray, probs: np.ndarray, class_names: List[str], title: str, save_path: str):
    plt.figure(figsize=(10, 8))
    k = len(class_names)
    colors = plt.colormaps.get_cmap("viridis")(np.linspace(0, 1, k))
    for i, cname in enumerate(class_names):
        if i in np.unique(labels):
            fpr, tpr, _ = roc_curve((labels == i).astype(int), probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{cname} (AUC={roc_auc:.3f})")
        else:
            plt.plot([], [], color=colors[i], lw=2, label=f"{cname} (no samples)")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 6. TEMPERATURE SCALING (VAL -> TEST)
# ==========================================
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)

    @torch.no_grad()
    def get_T(self) -> float:
        return float(self.temperature.detach().cpu().item())

@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for x, y in progress_iter(loader, desc="Collect logits", total=len(loader)):
            x = x.to(CONFIG["device"], non_blocking=True)
            y = y.to(CONFIG["device"], non_blocking=True)
            x = apply_eval_gpu_norm(x)
            with torch.autocast(device_type=CONFIG["device"].type, enabled=CONFIG["amp_enabled"]):
                logits = forward_logits(model, x)
            all_logits.append(logits.detach())
            all_labels.append(y.detach())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def fit_temperature_on_val(model: nn.Module, val_loader: DataLoader) -> TemperatureScaler:
    log("[CAL] Fitting temperature scaling on VAL set...")
    logits, labels = collect_logits_labels(model, val_loader)
    scaler = TemperatureScaler().to(CONFIG["device"])
    nll_criterion = nn.CrossEntropyLoss()

    optimizer = optim.LBFGS([scaler.temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = nll_criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    log(f"[CAL] Learned temperature T = {scaler.get_T():.4f}")
    return scaler

# ==========================================
# 7. TRAINING ENGINE (FROZEN LoRA)
# ==========================================
@dataclass
class RunPaths:
    out_dir: str
    best_model_path: str
    last_model_path: str
    history_csv: str
    summary_json: str
    adapter_dir: str

def make_run_paths(dataset_name: str, num_classes: int) -> RunPaths:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("runs", f"{dataset_name}", f"deit_small_lora_frozen_{num_classes}c", ts)
    os.makedirs(out_dir, exist_ok=True)
    return RunPaths(
        out_dir=out_dir,
        best_model_path=os.path.join(out_dir, "best_val_loss.pth"),
        last_model_path=os.path.join(out_dir, "last_epoch.pth"),
        history_csv=os.path.join(out_dir, "history.csv"),
        summary_json=os.path.join(out_dir, "summary.json"),
        adapter_dir=os.path.join(out_dir, "adapter_only"),
    )

class EarlyStopping:
    def __init__(self, patience: int, path: str):
        self.patience = patience
        self.path = path
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best is None or score > self.best:
            self.best = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            log(f"[ES] Saved best checkpoint -> {self.path}")
        else:
            self.counter += 1
            log(f"[ES] no improve: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

def train_one_epoch(model, loader, criterion, optimizer) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    if CONFIG["device"].type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    scaler = torch.amp.GradScaler(enabled=CONFIG["amp_enabled"])
    t0 = time.time()

    for x, y in progress_iter(loader, desc="Train", total=len(loader)):
        x = x.to(CONFIG["device"], non_blocking=True)
        y = y.to(CONFIG["device"], non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        x = apply_train_gpu_aug_and_norm(x)

        with torch.autocast(device_type=CONFIG["device"].type, enabled=CONFIG["amp_enabled"]):
            out = model(x)
            if isinstance(out, (tuple, list)):
                loss = 0.5 * (criterion(out[0], y) + criterion(out[1], y))
                logits = (out[0] + out[1]) / 2.0
            else:
                loss = criterion(out, y)
                logits = out

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    epoch_time = time.time() - t0
    peak_mem_gb = 0.0
    if CONFIG["device"].type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return total_loss / max(total, 1), correct / max(total, 1), epoch_time, peak_mem_gb

@torch.no_grad()
def validate_one_epoch(model, loader, criterion) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_y, all_p = [], []

    for x, y in progress_iter(loader, desc="Val", total=len(loader)):
        x = x.to(CONFIG["device"], non_blocking=True)
        y = y.to(CONFIG["device"], non_blocking=True)

        x = apply_eval_gpu_norm(x)

        with torch.autocast(device_type=CONFIG["device"].type, enabled=CONFIG["amp_enabled"]):
            out = model(x)
            if isinstance(out, (tuple, list)):
                loss = 0.5 * (criterion(out[0], y) + criterion(out[1], y))
                logits = (out[0] + out[1]) / 2.0
            else:
                loss = criterion(out, y)
                logits = out

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

        all_y.append(y.detach().cpu().numpy())
        all_p.append(pred.detach().cpu().numpy())

    y_np = np.concatenate(all_y)
    p_np = np.concatenate(all_p)
    macro_f1 = float(f1_score(y_np, p_np, average="macro"))
    return total_loss / max(total, 1), correct / max(total, 1), macro_f1

def save_adapter_only(model: nn.Module, out_dir: str) -> int:
    """
    Save adapter-only weights (PEFT) for storage-size reporting.
    Returns size in bytes of adapter weights file(s).
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        model.save_pretrained(out_dir)  # PEFT adapters
        size = 0
        for r, _, files in os.walk(out_dir):
            for f in files:
                size += os.path.getsize(os.path.join(r, f))
        return size
    except Exception as e:
        log(f"[WARN] Could not save adapter-only checkpoint: {e}")
        return 0

def train_engine(dataset_name: str, train_loader, val_loader, class_names: List[str], class_weights: torch.Tensor, run_paths: RunPaths):
    num_classes = len(class_names)
    model = create_deit_lora_model(num_classes)

    # Class-weighted loss (optional via imbalance.strategy)
    use_weighted_loss = CONFIG["imbalance"]["enabled"] and ("loss" in CONFIG["imbalance"]["strategy"])
    ce_weight = class_weights if use_weighted_loss else None

    criterion = nn.CrossEntropyLoss(
        weight=ce_weight,
        label_smoothing=float(CONFIG["label_smoothing"])
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=float(CONFIG["lr"]), weight_decay=float(CONFIG["weight_decay"]))

    def lr_lambda(epoch):
        if epoch < int(CONFIG["warmup_epochs"]):
            return float(epoch + 1) / float(CONFIG["warmup_epochs"])
        return 0.5 * (1 + math.cos(math.pi *
                                   (epoch - CONFIG["warmup_epochs"]) /
                                   max(1, (CONFIG["epochs"] - CONFIG["warmup_epochs"]))))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    early_stop = EarlyStopping(int(CONFIG["patience"]), run_paths.best_model_path)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
        "epoch_time_sec": [],
        "train_peak_vram_gb": [],
    }

    total_train_t0 = time.time()

    for ep in range(1, int(CONFIG["epochs"]) + 1):
        train_loss, train_acc, ep_time, peak_vram = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_macro_f1 = validate_one_epoch(model, val_loader, criterion)

        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["lr"].append(lr_now)
        history["epoch_time_sec"].append(ep_time)
        history["train_peak_vram_gb"].append(peak_vram)

        log(
            f"[{dataset_name}] Epoch {ep:03d}/{int(CONFIG['epochs']):03d} | "
            f"lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% val_macroF1={val_macro_f1*100:.2f}% | "
            f"ep_time={ep_time:.1f}s peakVRAM={peak_vram:.2f}GB"
        )

        early_stop(val_loss, model)
        if early_stop.stop:
            log(f"[{dataset_name}] Early stopping triggered.")
            break

    total_train_time = time.time() - total_train_t0

    torch.save(model.state_dict(), run_paths.last_model_path)
    pd.DataFrame(history).to_csv(run_paths.history_csv, index=False)

    if os.path.exists(run_paths.best_model_path):
        model.load_state_dict(torch.load(run_paths.best_model_path, map_location=CONFIG["device"]))

    adapter_bytes = save_adapter_only(model, run_paths.adapter_dir)
    total_p, trainable_p, trainable_pct = count_params(model)

    return model, history, total_train_time, (total_p, trainable_p, trainable_pct), adapter_bytes

# ==========================================
# 8. PLOTS: TRAIN CURVES + LR
# ==========================================
def plot_training_curves(history: Dict, title: str, out_dir: str):
    epochs = history["epoch"]
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, np.array(history["train_acc"]) * 100, label="train_acc")
    plt.plot(epochs, np.array(history["val_acc"]) * 100, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["lr"])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title(f"{title} - LR schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_schedule.png"), dpi=300)
    plt.close()

# ==========================================
# 9. EVALUATION (TEST): METRICS + FIGURES + TSNE
# ==========================================
def unwrap_timm_backbone(model: nn.Module) -> nn.Module:
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model
    if hasattr(base, "model"):
        base = base.model
    return base

@torch.no_grad()
def evaluate_on_loader(model: nn.Module, loader: DataLoader, class_names: List[str]) -> Dict:
    model.eval()
    all_y, all_pred, all_probs, all_logits = [], [], [], []

    for x, y in progress_iter(loader, desc="Test eval", total=len(loader)):
        x = x.to(CONFIG["device"], non_blocking=True)
        y = y.to(CONFIG["device"], non_blocking=True)

        x = apply_eval_gpu_norm(x)

        with torch.autocast(device_type=CONFIG["device"].type, enabled=CONFIG["amp_enabled"]):
            logits = forward_logits(model, x)
            probs = torch.softmax(logits, dim=1)

        pred = probs.argmax(dim=1)

        all_y.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu())

    y_np = np.concatenate(all_y)
    p_np = np.concatenate(all_pred)
    probs_np = np.concatenate(all_probs)
    logits_t = torch.cat(all_logits, dim=0)

    acc = float((p_np == y_np).mean())
    macro_f1 = float(f1_score(y_np, p_np, average="macro"))

    macro_auc = None
    try:
        y_onehot = np.eye(len(class_names))[y_np]
        macro_auc = float(roc_auc_score(y_onehot, probs_np, average="macro", multi_class="ovr"))
    except Exception as e:
        log(f"[WARN] Could not compute macro-AUC: {e}")

    ece = compute_ece(probs_np, y_np, n_bins=int(CONFIG["ece_bins"]))
    nll = compute_nll(probs_np, y_np)
    brier = compute_brier(probs_np, y_np, num_classes=len(class_names))

    return {
        "labels": y_np,
        "preds": p_np,
        "probs": probs_np,
        "logits": logits_t,
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "ece": ece,
        "nll": nll,
        "brier": brier,
    }

def plot_tsne_stratified(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    title: str,
    save_path: str,
    samples_per_class: int = 200,
):
    backbone = unwrap_timm_backbone(model)
    if not hasattr(backbone, "forward_features"):
        log("[WARN] No forward_features(); skipping t-SNE.")
        return

    model.eval()
    num_classes = len(class_names)

    feats_by_class = {i: [] for i in range(num_classes)}
    labels_by_class = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for x, y in progress_iter(loader, desc="t-SNE feats", total=len(loader)):
            x = x.to(CONFIG["device"], non_blocking=True)
            x = apply_eval_gpu_norm(x)

            with torch.autocast(device_type=CONFIG["device"].type, enabled=CONFIG["amp_enabled"]):
                f = backbone.forward_features(x)
            if f.ndim == 3:
                f = f[:, 0]  # CLS token

            f = f.detach().cpu().numpy()
            y = y.numpy()

            for i in range(len(y)):
                c = int(y[i])
                if len(feats_by_class[c]) < samples_per_class:
                    feats_by_class[c].append(f[i])
                    labels_by_class[c].append(c)

            if all(len(feats_by_class[c]) >= samples_per_class for c in range(num_classes)):
                break

    X_list, y_list = [], []
    for c in range(num_classes):
        X_list.extend(feats_by_class[c])
        y_list.extend(labels_by_class[c])

    if len(X_list) < 30:
        log("[WARN] Not enough samples for t-SNE.")
        return

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    log(f"[INFO] t-SNE using {len(X)} samples ({samples_per_class} per class x {num_classes} classes).")

    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    log("[INFO] Running t-SNE ...")
    tsne = TSNE(
        n_components=2,
        random_state=int(CONFIG["seed"]),
        perplexity=min(30, len(X) - 1),
    )
    Z = tsne.fit_transform(X)

    palette = sns.color_palette("viridis", num_classes)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=Z[:, 0], y=Z[:, 1],
        hue=[class_names[i] for i in y],
        hue_order=class_names,
        palette=palette,
        alpha=0.85,
        s=35
    )
    plt.title(title)
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 10. INFERENCE BENCHMARK (LATENCY/THROUGHPUT + VRAM)
# ==========================================
def maybe_merge_lora_for_timing(model: nn.Module) -> nn.Module:
    if not bool(CONFIG["bench_merge_lora"]):
        return model
    if hasattr(model, "merge_and_unload"):
        try:
            merged = model.merge_and_unload()
            log("[BENCH] LoRA merged (merge_and_unload).")
            return merged.to(CONFIG["device"])
        except Exception as e:
            log(f"[BENCH] merge_and_unload failed; benchmarking unmerged LoRA. Reason: {e}")
    return model

@torch.no_grad()
def benchmark_inference(model: nn.Module, sample: torch.Tensor, batch_size: int) -> Dict:
    model.eval()

    x = sample[:batch_size].contiguous()
    x = x.to(CONFIG["device"], non_blocking=True)
    x = apply_eval_gpu_norm(x)

    if CONFIG["device"].type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # warmup
    for _ in range(int(CONFIG["bench_warmup_iters"])):
        with torch.autocast(device_type=CONFIG["device"].type, enabled=CONFIG["amp_enabled"]):
            _ = forward_logits(model, x)
    if CONFIG["device"].type == "cuda":
        torch.cuda.synchronize()

    # timed
    times_ms = []
    for _ in range(int(CONFIG["bench_timed_iters"])):
        t0 = time.perf_counter()
        with torch.autocast(device_type=CONFIG["device"].type, enabled=CONFIG["amp_enabled"]):
            _ = forward_logits(model, x)
        if CONFIG["device"].type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms, dtype=np.float64)
    mean_ms = float(times_ms.mean())
    std_ms = float(times_ms.std())

    ms_per_img = mean_ms / batch_size
    imgs_per_s = 1000.0 / ms_per_img

    peak_vram_gb = 0.0
    if CONFIG["device"].type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "batch": int(batch_size),
        "mean_batch_ms": mean_ms,
        "std_batch_ms": std_ms,
        "ms_per_img": float(ms_per_img),
        "imgs_per_s": float(imgs_per_s),
        "peak_vram_gb": float(peak_vram_gb),
    }

# ==========================================
# 11. MAIN: RUN DATASETS
# ==========================================
def main():
    set_seeds()

    log("\n" + "=" * 90)
    log("DeiT + LoRA (Frozen backbone) OCT Pipeline (paper-grade)")
    log("=" * 90)
    log(f"[ENV] torch={torch.__version__} cuda={torch.cuda.is_available()} device={CONFIG['device']}")
    log(f"[CFG] model={CONFIG['model_name']} img_size={CONFIG['img_size']} batch={CONFIG['batch_size']} amp={CONFIG['amp_enabled']}")
    log(f"[CFG] num_workers={CONFIG['num_workers']} (Windows spawn-safe)")
    log(f"[CFG] imbalance={CONFIG['imbalance']}")
    log("=" * 90)

    for ds_name in RUN_DATASETS:
        if ds_name not in DATASET_CONFIGS:
            log(f"[SKIP] Unknown dataset '{ds_name}'")
            continue

        train_loader, val_loader, test_loader, class_names, class_w = build_dataloaders(ds_name, DATASET_CONFIGS[ds_name])
        run_paths = make_run_paths(ds_name, num_classes=len(class_names))

        model, history, total_train_time, param_stats, adapter_bytes = train_engine(
            ds_name, train_loader, val_loader, class_names, class_w, run_paths
        )

        plot_training_curves(history, title=f"{ds_name} DeiT-S LoRA Frozen", out_dir=run_paths.out_dir)

        test_metrics = evaluate_on_loader(model, test_loader, class_names)

        # Calibration (temp scaling)
        T = None
        test_metrics_cal = None
        if bool(CONFIG["use_temperature_scaling"]):
            ts_model = fit_temperature_on_val(model, val_loader)

            logits = test_metrics["logits"].to(CONFIG["device"])
            with torch.no_grad():
                scaled_logits = ts_model(logits)
                probs = torch.softmax(scaled_logits, dim=1).detach().cpu().numpy()

            y = test_metrics["labels"]
            ece = compute_ece(probs, y, n_bins=int(CONFIG["ece_bins"]))
            nll = compute_nll(probs, y)
            brier = compute_brier(probs, y, num_classes=len(class_names))
            T = ts_model.get_T()
            test_metrics_cal = {"ece": ece, "nll": nll, "brier": brier}

            plot_reliability_diagram(
                probs, y, n_bins=int(CONFIG["ece_bins"]),
                title=f"{ds_name} Reliability (Temp-scaled, T={T:.3f})",
                save_path=os.path.join(run_paths.out_dir, "reliability_temp_scaled.png")
            )

        # Uncalibrated plots
        y = test_metrics["labels"]
        p = test_metrics["preds"]
        probs = test_metrics["probs"]

        rep = classification_report(y, p, target_names=class_names, digits=4, zero_division=0)
        log("\n" + "-" * 80)
        log(f"[{ds_name}] TEST classification report (uncalibrated)")
        log("-" * 80)
        log(rep)

        cm = confusion_matrix(y, p)
        plot_confusion(cm, class_names, f"{ds_name} Confusion (normalized)", os.path.join(run_paths.out_dir, "confusion.png"))
        plot_classification_report_heatmap(rep, f"{ds_name} Class report heatmap", os.path.join(run_paths.out_dir, "cls_report_heatmap.png"))
        plot_sens_spec(cm, class_names, f"{ds_name} Sensitivity/Specificity", os.path.join(run_paths.out_dir, "sens_spec.png"))
        plot_roc_curves(y, probs, class_names, f"{ds_name} ROC (OVR)", os.path.join(run_paths.out_dir, "roc.png"))
        plot_reliability_diagram(
            probs, y, n_bins=int(CONFIG["ece_bins"]),
            title=f"{ds_name} Reliability (uncalibrated)",
            save_path=os.path.join(run_paths.out_dir, "reliability_uncalibrated.png")
        )
        plot_tsne_stratified(
            model,
            test_loader,
            class_names,
            title=f"{ds_name} t-SNE (CLS features, stratified)",
            save_path=os.path.join(run_paths.out_dir, "tsne_stratified.png"),
            samples_per_class=200
        )

        # Benchmarks
        x0, _ = next(iter(test_loader))
        model_for_bench = maybe_merge_lora_for_timing(model)

        bench_lat = benchmark_inference(model_for_bench, x0, int(CONFIG["bench_latency_batch"]))
        bench_thr = benchmark_inference(model_for_bench, x0, min(int(CONFIG["bench_throughput_batch"]), int(x0.size(0))))

        total_p, trainable_p, trainable_pct = param_stats
        best_val_loss = float(np.min(history["val_loss"]))
        best_val_f1 = float(np.max(history["val_macro_f1"]))
        mean_epoch_time = float(np.mean(history["epoch_time_sec"]))
        peak_train_vram = float(np.max(history["train_peak_vram_gb"]))

        summary = {
            "dataset": ds_name,
            "classes": class_names,
            "model": CONFIG["model_name"],
            "peft": "LoRA (frozen backbone)",
            "imbalance": CONFIG["imbalance"],
            "lora": {
                "r": CONFIG["lora_r"],
                "alpha": CONFIG["lora_alpha"],
                "dropout": CONFIG["lora_dropout"],
                "merged_for_timing": bool(CONFIG["bench_merge_lora"]),
            },
            "params": {
                "total": int(total_p),
                "trainable": int(trainable_p),
                "trainable_pct": float(trainable_pct),
            },
            "storage": {
                "adapter_only_bytes": int(adapter_bytes),
                "adapter_only_mb": float(adapter_bytes / (1024**2)) if adapter_bytes else None,
            },
            "training": {
                "epochs_ran": int(len(history["epoch"])),
                "best_val_loss": best_val_loss,
                "best_val_macro_f1": best_val_f1,
                "total_train_time_sec": float(total_train_time),
                "mean_epoch_time_sec": mean_epoch_time,
                "peak_train_vram_gb": peak_train_vram,
            },
            "test_uncalibrated": {
                "acc": float(test_metrics["acc"]),
                "macro_f1": float(test_metrics["macro_f1"]),
                "macro_auc": test_metrics["macro_auc"],
                "ece": float(test_metrics["ece"]),
                "nll": float(test_metrics["nll"]),
                "brier": float(test_metrics["brier"]),
            },
            "test_temp_scaled": {
                "enabled": bool(CONFIG["use_temperature_scaling"]),
                "T": T,
                **(test_metrics_cal if test_metrics_cal is not None else {}),
            },
            "inference_benchmark": {
                "latency_batch1": bench_lat,
                "throughput_batchN": bench_thr,
            }
        }

        with open(run_paths.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        log("\n" + "=" * 80)
        log(f"[{ds_name}] TEST summary (uncalibrated): "
              f"Acc={test_metrics['acc']*100:.2f}% MacroF1={test_metrics['macro_f1']*100:.2f}% "
              f"ECE={test_metrics['ece']:.4f} NLL={test_metrics['nll']:.4f} Brier={test_metrics['brier']:.4f}")
        if test_metrics["macro_auc"] is not None:
            log(f"[{ds_name}] Macro-AUC={test_metrics['macro_auc']:.4f}")
        if test_metrics_cal is not None:
            log(f"[{ds_name}] TEMP-SCALED (T={T:.3f}): ECE={test_metrics_cal['ece']:.4f} "
                  f"NLL={test_metrics_cal['nll']:.4f} Brier={test_metrics_cal['brier']:.4f}")
        log(f"[{ds_name}] Params trainable: {trainable_p:,} ({trainable_pct:.3f}%)")
        if adapter_bytes:
            log(f"[{ds_name}] Adapter-only size: {adapter_bytes/(1024**2):.2f} MB")
        log(f"[{ds_name}] Inference latency (B=1): {bench_lat['ms_per_img']:.3f} ms/img "
              f"({bench_lat['imgs_per_s']:.1f} imgs/s), peakVRAM={bench_lat['peak_vram_gb']:.2f}GB")
        log(f"[{ds_name}] Inference throughput (B={bench_thr['batch']}): {bench_thr['imgs_per_s']:.1f} imgs/s, "
              f"peakVRAM={bench_thr['peak_vram_gb']:.2f}GB")
        log(f"[{ds_name}] Outputs saved in: {run_paths.out_dir}")
        log("=" * 80 + "\n")

        # Cleanup
        del model
        del model_for_bench
        del train_loader, val_loader, test_loader
        del history, test_metrics, cm, rep
        del x0
        if "ts_model" in locals():
            del ts_model
        if "logits" in locals():
            del logits
        if "scaled_logits" in locals():
            del scaled_logits

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
