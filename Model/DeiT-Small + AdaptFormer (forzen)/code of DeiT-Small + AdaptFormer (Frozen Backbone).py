"""
DeiT-Small + AdaptFormer (Frozen Backbone) for Retinal OCT Classification
=======================================================================
AdaptFormer-style adapters inserted into each Transformer block's MLP (AdaptMLP):
  - Frozen original MLP branch
  - Trainable bottleneck branch: down -> ReLU -> up, scaled by s (parallel insertion)

Backbone frozen; train ONLY adapters + heads

Multi-dataset runner (each dataset has train/val/test splits)
Dynamic class discovery from train/<class>/
RAM caching per split (stores uint8 tensors to save RAM)
GPU batch augmentations (torchvision.transforms.v2 preferred; Kornia fallback)

Strong imbalance handling:
  - Class-Balanced weights (Effective Number of Samples)
  - Optional WeightedRandomSampler for severe imbalance

Research-grade evaluation:
  - Accuracy, Macro-F1, Macro-AUC (OVR)
  - Confusion matrix (raw + normalized)
  - Classification report (print + csv + heatmap)
  - Sensitivity/Specificity per class + plot
  - Calibration: ECE, NLL, Brier + Reliability diagram
  - Optional Temperature Scaling (fit on VAL, apply on TEST)
  - t-SNE (CLS features)
  - Inference benchmark latency/throughput + peak VRAM

Saves ONLY best model (best val loss) + full results in dataset folder
Clears RAM + VRAM after each dataset
"""

# =========================
# 0. IMPORTS & DEP CHECKS
# =========================
import os
import sys
import time
import math
import json
import random
import warnings
import gc
import subprocess
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt


def _pip_install(pkg: str):
    """Robust pip install that works in both scripts and notebooks."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


# tqdm
try:
    from tqdm import tqdm
except Exception:
    _pip_install("tqdm")
    from tqdm import tqdm

# sklearn
try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_curve,
        auc,
        f1_score,
        roc_auc_score,
    )
    from sklearn.manifold import TSNE
except Exception:
    _pip_install("scikit-learn")
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_curve,
        auc,
        f1_score,
        roc_auc_score,
    )
    from sklearn.manifold import TSNE

# timm
try:
    import timm
except Exception:
    _pip_install("timm")
    import timm

# seaborn optional (nice heatmaps)
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# PIL
try:
    from PIL import Image
except Exception:
    _pip_install("pillow")
    from PIL import Image

# torchvision (for v2 GPU augment)
try:
    import torchvision
    from torchvision.transforms import v2 as T2
    from torchvision.transforms.v2 import InterpolationMode
    _HAS_TV2 = True
except Exception:
    _HAS_TV2 = False
    T2 = None
    InterpolationMode = None

# Kornia fallback for GPU aug if v2 not available
try:
    import kornia
    import kornia.augmentation as K
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


# =========================
# 1. USER SETTINGS (EDIT)
# =========================
DATASET_CONFIGS = {
  "C8": {"root": r"D:\AIUB\DSP\Code\Datasets\C8\RetinalOCT_Dataset_CLEAN_SHAONLY",
         "pad_to_square": True,
         "cache_in_ram": True},

  "NEH_UT_2021": {"root": r"D:\AIUB\DSP\Code\Datasets\NEH_UT_2021RetinalOCTDataset\NEH_UT_2021RetinalOCTDataset_V2_CLEAN_SHAONLY",
          "pad_to_square": True,
          "cache_in_ram": True},

  "THOCT": {"root": r"D:\AIUB\DSP\Code\Datasets\THOCT1800\THOCT1800_CLEAN_SHAONLY",
          "pad_to_square": True,
          "cache_in_ram": True},

  "Srinivasan_2014": {"root": r"D:\AIUB\DSP\Code\Datasets\Srinivasan_2014\Srinivasan_2014_CLEAN_SHAONLY",
          "pad_to_square": True,
          "cache_in_ram": True},

  "OCTDL": {"root": r"D:\AIUB\DSP\Code\Datasets\OCTDL\OCTDL_CLEAN_SHAONLY",
          "pad_to_square": True,
          "cache_in_ram": True},
}

# Choose which datasets to run (keys from DATASET_CONFIGS)
RUN_DATASETS = ["THOCT", "Srinivasan_2014", "OCTDL", "NEH_UT_2021", "C8"]


# =========================
# 2. GLOBAL CONFIG
# =========================
CONFIG = {
    # --- reproducibility / device ---
    "seed": 42,

    # NOTE: torch.device is NOT JSON serializable. We keep it as torch.device for runtime,
    # and later use a json "default" function to serialize it as a string (e.g., "cuda").
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "deterministic": False,   # True = more reproducible but slower
    "amp_enabled": True,      # mixed precision on CUDA
    "num_workers": 0,         # Windows-safe; avoid RAM duplication with caching
    "pin_memory": True,

    # --- model ---
    "model_name": "deit_small_distilled_patch16_224",
    "img_size": 224,

    # --- training ---
    "batch_size": 64,         # safe default for 16GB VRAM; increase carefully
    "epochs": 100,
    "lr": 5e-4,
    "weight_decay": 1e-2,
    "warmup_epochs": 10,
    "patience": 10,
    "label_smoothing": 0.1,
    "grad_clip_norm": 1.0,

    # --- AdaptFormer (frozen backbone) ---
    "freeze_backbone": True,

    # AdaptFormer defaults (paper suggests mid_dim=64, scale s=0.1, parallel insertion)
    "adapt_mid_dim": 64,
    "adapt_scale": 0.10,
    "adapt_dropout": 0.0,         # paper doesn't rely on dropout in the adapter; keep 0 by default

    # Optionally restrict which blocks get adapters:
    # - start_layer=0, end_layer=None => all blocks
    # - e.g. start_layer=6, end_layer=None => top half
    "adapt_start_layer": 0,
    "adapt_end_layer": None,

    # --- imbalance handling ---
    "use_class_balanced_weights": True,
    "cb_beta": 0.9999,                 # effective number beta
    "use_weighted_sampler": True,      # good for extreme imbalance
    "sampler_min_count": 20,           # if any class < this => allow sampler
    "sampler_imbalance_ratio": 10.0,   # if max/min >= this => allow sampler

    # --- calibration ---
    "use_temperature_scaling": True,

    # --- inference benchmark ---
    "bench_warmup_iters": 30,
    "bench_timed_iters": 150,
    "bench_latency_batch": 1,
    "bench_throughput_batch": 64,

    # --- outputs ---
    "runs_root": "runs",
    "save_adapter_only": True,         # also save adapter-only for size reporting (optional)
}


# =========================
# 2.0 JSON SERIALIZATION HELPERS
# =========================
def _json_default(o):
    """
    JSON fallback converter for non-serializable objects used in configs/summaries.

    Handles:
      - torch.device -> "cuda" / "cpu"
      - torch.dtype  -> "torch.float16", etc.
      - torch.Tensor -> Python list (CPU)
      - numpy scalar -> Python scalar
      - numpy array  -> list
    Anything unknown -> string representation (still valid JSON).
    """
    # torch
    if isinstance(o, torch.device):
        return str(o)
    if isinstance(o, torch.dtype):
        return str(o)
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()

    # numpy
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()

    # fallback
    return str(o)


# =========================
# 2.1 SEEDING
# =========================
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

set_seed(CONFIG["seed"], CONFIG["deterministic"])


# =========================
# 3. LOGGING (PRINT + SAVE)
# =========================
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def log_print(log_path: str, msg: str):
    print(msg)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# =========================
# 4. DATA UTILS & STATS
# =========================
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_class_names(train_dir: str) -> List[str]:
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    if len(classes) < 2:
        raise RuntimeError(f"Found <2 classes in: {train_dir}. Classes={classes}")
    return classes

def scan_split_counts(split_dir: str, class_names: List[str]) -> Dict[str, int]:
    counts = {}
    for c in class_names:
        cdir = os.path.join(split_dir, c)
        if not os.path.isdir(cdir):
            counts[c] = 0
            continue
        n = 0
        for fn in os.listdir(cdir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in VALID_EXT:
                n += 1
        counts[c] = n
    return counts

def dataset_stats_table(root: str, class_names: List[str]) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Missing split folder: {split_dir}")
        counts = scan_split_counts(split_dir, class_names)
        for c in class_names:
            rows.append({"split": split, "class": c, "count": counts[c]})
    df = pd.DataFrame(rows)
    return df

def print_and_save_dataset_stats(df: pd.DataFrame, out_dir: str, log_path: str):
    safe_mkdir(out_dir)
    df.to_csv(os.path.join(out_dir, "dataset_split_class_counts.csv"), index=False)

    # Summary pivot: split rows, class columns.
    pivot = df.pivot_table(index="split", columns="class", values="count", aggfunc="sum", fill_value=0)
    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot_path = os.path.join(out_dir, "dataset_split_summary.csv")
    pivot.to_csv(pivot_path)

    log_print(log_path, "\n[DATA] Split/Class counts:")
    log_print(log_path, str(pivot))

    # Basic imbalance diagnostics on train split.
    train_counts = df[df["split"] == "train"].set_index("class")["count"].to_dict()
    nonzero = [v for v in train_counts.values() if v > 0]
    if nonzero:
        mn, mx = min(nonzero), max(nonzero)
        ratio = (mx / mn) if mn > 0 else float("inf")
        log_print(log_path, f"[DATA] Train imbalance: min={mn}, max={mx}, ratio={ratio:.2f}")
    else:
        log_print(log_path, "[DATA] WARNING: no train images??")


# =========================
# 5. PAD-TO-SQUARE (PIL)
# =========================
def pad_to_square_pil(img: Image.Image, fill: int = 0) -> Image.Image:
    """
    Optional preprocessing:
    - Many OCT datasets contain rectangular images.
    - ViT/DeiT likes square inputs, so we can pad before resizing to avoid distortion.
    """
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    new_img = Image.new("RGB", (size, size), (fill, fill, fill))
    new_img.paste(img, ((size - w)//2, (size - h)//2))
    return new_img


# =========================
# 6. RAM-CACHED DATASET (uint8 tensors)
# =========================
class OCTCachedFolderDataset(Dataset):
    """
    Expects:
      root/train/<class>/*.png|jpg
      root/val/<class>/*.png|jpg
      root/test/<class>/*.png|jpg

    RAM caching:
      - Stores uint8 tensors (C,H,W) to reduce memory (1 byte/channel instead of float32)
      - Augmentation + normalization happens later on GPU
    """
    def __init__(
        self,
        root_dir: str,
        split: str,
        class_names: List[str],
        img_size: int,
        pad_to_square: bool,
        cache_in_ram: bool,
        log_path: str,
    ):
        self.root_dir = root_dir
        self.split = split
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.img_size = img_size
        self.pad_to_square = pad_to_square
        self.cache_in_ram = cache_in_ram
        self.log_path = log_path

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Missing split folder: {split_dir}")

        self.paths: List[str] = []
        self.labels: List[int] = []

        # Classic ImageFolder-style scan, but we keep it custom to support caching.
        for c in class_names:
            cdir = os.path.join(split_dir, c)
            if not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in VALID_EXT:
                    continue
                p = os.path.join(cdir, fn)
                if os.path.isfile(p):
                    self.paths.append(p)
                    self.labels.append(self.class_to_idx[c])

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

        self.cache: Optional[List[torch.Tensor]] = None

        if cache_in_ram:
            # Rough estimate for RAM usage:
            # Each image stored as uint8: H*W*3 bytes.
            est_bytes = len(self.paths) * (img_size * img_size * 3)
            est_gb = est_bytes / (1024**3)
            log_print(log_path, f"[RAM] Caching split='{split}' images={len(self.paths)} "
                                f"as uint8 tensors. Estimated RAM ~ {est_gb:.2f} GB")

            self.cache = []
            for p in tqdm(self.paths, desc=f"[CACHE] {split}", leave=False):
                t = self._load_and_preprocess_uint8(p)
                self.cache.append(t)

            log_print(log_path, f"[RAM] Cached {len(self.cache)} images for split='{split}'")

    def _load_and_preprocess_uint8(self, path: str) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
    
        if self.pad_to_square:
            img = pad_to_square_pil(img, fill=0)
    
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
    
        # Ensure NumPy array is writable (avoids torch warning)
        arr = np.array(img, dtype=np.uint8, copy=True)  # HWC
    
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8
        return t

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        y = self.labels[idx]
        if self.cache_in_ram and self.cache is not None:
            x = self.cache[idx]
        else:
            x = self._load_and_preprocess_uint8(self.paths[idx])
        return x, y


# =========================
# 7. GPU AUGMENTATION + NORMALIZATION
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class GPUAugmentAndNormalize(nn.Module):
    """
    Performs augmentation AND normalization *on GPU*.

    Input:
      - uint8 tensor image(s): (B,C,H,W) or (C,H,W) on GPU
    Output:
      - float32 normalized: (B,C,H,W)
    """
    def __init__(self, mode: str, device: torch.device):
        super().__init__()
        assert mode in ["train", "eval"]
        self.mode = mode
        self.device = device

        self.use_tv2 = _HAS_TV2
        self.use_kornia = (not _HAS_TV2) and _HAS_KORNIA

        if not (self.use_tv2 or self.use_kornia):
            raise RuntimeError(
                "Need torchvision.transforms.v2 OR kornia for GPU batch augmentation.\n"
                "Fix: `pip install --upgrade torchvision` (recommended) OR `pip install kornia`."
            )

        if self.use_tv2:
            if mode == "train":
                self.pipe = nn.Sequential(
                    T2.RandomHorizontalFlip(p=0.5),
                    T2.RandomAffine(
                        degrees=5,
                        translate=(0.02, 0.02),
                        scale=(0.95, 1.05),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    T2.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
                    T2.ToDtype(torch.float32, scale=True),
                    T2.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0),
                    T2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                )
            else:
                self.pipe = nn.Sequential(
                    T2.ToDtype(torch.float32, scale=True),
                    T2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                )

        else:
            if mode == "train":
                self.pipe = K.AugmentationSequential(
                    K.RandomHorizontalFlip(p=0.5),
                    K.RandomAffine(degrees=5, translate=0.02, scale=(0.95, 1.05), p=0.7),
                    K.RandomGaussianBlur((3, 3), (0.1, 1.0), p=0.15),
                    data_keys=["input"],
                    same_on_batch=False,
                )
            else:
                self.pipe = nn.Identity()

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if self.use_tv2:
            return self.pipe(x)

        x = x.float().div(255.0)
        x = self.pipe(x)

        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x


# =========================
# 8. ADAPTFORMER (AdaptMLP) INJECTION
# =========================
class AdaptMLP(nn.Module):
    """
    AdaptFormer AdaptMLP (parallel insertion):
      y = MLP_frozen(x) + s * Up(ReLU(Down(x)))

    Notes:
      - In timm DeiT/ViT, the Block already applies LayerNorm before calling mlp:
          mlp_out = blk.mlp( blk.norm2(x) )
        so the input x here is already LN(x), matching the paper formula.

    Initialization (paper):
      - down weight: Kaiming Normal
      - up weight: zeros
      - biases: zeros
    """
    def __init__(self, orig_mlp: nn.Module, dim: int, mid_dim: int, scale: float, dropout: float = 0.0):
        super().__init__()
        self.orig_mlp = orig_mlp
        self.scale = float(scale)
        self.drop = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()

        self.adapt_down = nn.Linear(dim, mid_dim, bias=True)
        self.adapt_act = nn.ReLU()
        self.adapt_up = nn.Linear(mid_dim, dim, bias=True)

        # Freeze original MLP branch
        for p in self.orig_mlp.parameters():
            p.requires_grad_(False)

        # Init per AdaptFormer paper
        nn.init.kaiming_normal_(self.adapt_down.weight, mode="fan_in", nonlinearity="relu")
        if self.adapt_down.bias is not None:
            nn.init.zeros_(self.adapt_down.bias)

        nn.init.zeros_(self.adapt_up.weight)
        if self.adapt_up.bias is not None:
            nn.init.zeros_(self.adapt_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.orig_mlp(x)
        a = self.drop(x)
        adapt_out = self.adapt_up(self.adapt_act(self.adapt_down(a)))
        return frozen_out + (self.scale * adapt_out)


def inject_adaptformer_into_deit_mlp(
    model: nn.Module,
    mid_dim: int,
    scale: float,
    dropout: float,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
):
    """
    Replace each Transformer block's .mlp with AdaptMLP wrapper (parallel adapter).
    Compatible with timm DeiT/ViT blocks: model.blocks[i].mlp and model.blocks[i].norm2.
    """
    if not hasattr(model, "blocks"):
        raise RuntimeError("Model has no .blocks; not a timm DeiT/ViT style model?")

    n_blocks = len(model.blocks)
    if end_layer is None:
        end_layer = n_blocks - 1

    start_layer = max(0, int(start_layer))
    end_layer = min(n_blocks - 1, int(end_layer))

    for i, blk in enumerate(model.blocks):
        if i < start_layer or i > end_layer:
            continue

        # Infer token embedding dim
        dim = None
        if hasattr(blk, "norm2") and hasattr(blk.norm2, "normalized_shape"):
            ns = blk.norm2.normalized_shape
            if isinstance(ns, (tuple, list)) and len(ns) >= 1:
                dim = int(ns[0])
        if dim is None and hasattr(blk, "mlp") and hasattr(blk.mlp, "fc1"):
            dim = int(blk.mlp.fc1.in_features)

        if dim is None:
            raise RuntimeError(f"Could not infer embed dim for block {i}.")

        blk.mlp = AdaptMLP(
            orig_mlp=blk.mlp,
            dim=dim,
            mid_dim=int(mid_dim),
            scale=float(scale),
            dropout=float(dropout),
        )


def freeze_backbone_keep_adapters_and_heads(model: nn.Module):
    """
    Freeze all params, then unfreeze:
      - AdaptFormer adapter params (adapt_down/adapt_up)
      - DeiT classification heads: head and head_dist
    """
    for p in model.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if "adapt_" in n:
            p.requires_grad = True

    for n, p in model.named_parameters():
        if n.startswith("head.") or n.startswith("head_dist."):
            p.requires_grad = True


def count_params(model: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    return total, trainable, pct


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    DeiT-distilled returns (logits, logits_dist).
    For evaluation/benchmark, average them.
    """
    out = model(x)
    if isinstance(out, (tuple, list)):
        return (out[0] + out[1]) / 2.0
    return out


def loss_for_distilled(model_out, y, criterion):
    """
    DeiT-distilled training often uses two heads: (logits, logits_dist)
    We compute average loss: 0.5*(CE(logits)+CE(logits_dist))
    """
    if isinstance(model_out, (tuple, list)):
        return 0.5 * (criterion(model_out[0], y) + criterion(model_out[1], y))
    return criterion(model_out, y)


def get_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Adapter-only state dict = AdaptFormer params + heads.

    We detach and move to CPU for portability.
    """
    sd = {}
    for name, tensor in model.state_dict().items():
        if ("adapt_" in name) or name.startswith("head.") or name.startswith("head_dist."):
            sd[name] = tensor.detach().cpu()
    return sd


# =========================
# 9. IMBALANCE HANDLING
# =========================
def class_balanced_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """
    Effective number of samples (Cui et al.):
      w_i = (1 - beta) / (1 - beta^{n_i})
    Then normalize weights to mean ~ 1.
    """
    counts = counts.astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    for i, n in enumerate(counts):
        if n <= 0:
            weights[i] = 0.0
        else:
            weights[i] = (1.0 - beta) / (1.0 - (beta ** n))
    if weights.sum() > 0:
        weights = weights / (weights.mean() + 1e-12)
    return weights


def build_sampler_if_needed(train_labels: List[int], num_classes: int, log_path: str):
    """
    Optional WeightedRandomSampler:
    - Enable if (min class < sampler_min_count) OR (max/min >= sampler_imbalance_ratio)
    """
    counts = np.bincount(np.array(train_labels), minlength=num_classes)
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return None

    mn = int(nonzero.min())
    mx = int(nonzero.max())
    ratio = (mx / mn) if mn > 0 else float("inf")

    use_sampler = CONFIG["use_weighted_sampler"] and (
        (mn < CONFIG["sampler_min_count"]) or (ratio >= CONFIG["sampler_imbalance_ratio"])
    )

    if not use_sampler:
        log_print(log_path, f"[IMB] Sampler disabled. min={mn} max={mx} ratio={ratio:.2f}")
        return None

    inv = np.zeros_like(counts, dtype=np.float64)
    for i in range(num_classes):
        inv[i] = (1.0 / counts[i]) if counts[i] > 0 else 0.0

    sample_w = np.array([inv[y] for y in train_labels], dtype=np.float64)
    sample_w = torch.from_numpy(sample_w).double()
    sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(train_labels), replacement=True)

    log_print(log_path, f"[IMB] Using WeightedRandomSampler. min={mn} max={mx} ratio={ratio:.2f}")
    return sampler


# =========================
# 10. METRICS + PLOTS
# =========================
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

def plot_confusion(cm: np.ndarray, class_names: List[str], title: str, save_path: str):
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype(float), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

    plt.figure(figsize=(10, 8))
    if _HAS_SNS:
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={"label": "Normalized Frequency"})
    else:
        plt.imshow(cm_norm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_sens_spec(cm: np.ndarray, class_names: List[str], title: str, save_path: str):
    sens, spec = sensitivity_specificity_from_cm(cm)
    x = np.arange(len(class_names))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, sens, width=0.4, label="Sensitivity (Recall)")
    plt.bar(x + 0.2, spec, width=0.4, label="Specificity")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

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
    plt.show()
    plt.close()

def plot_roc_curves(labels: np.ndarray, probs: np.ndarray, class_names: List[str], title: str, save_path: str):
    plt.figure(figsize=(10, 8))
    for i, cname in enumerate(class_names):
        if not np.any(labels == i):
            plt.plot([], [], label=f"{cname} (no samples)")
            continue
        fpr, tpr, _ = roc_curve((labels == i).astype(int), probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_classification_report_heatmap(report_dict: Dict, title: str, save_path: str):
    rows = []
    for k, v in report_dict.items():
        if isinstance(v, dict) and ("precision" in v):
            rows.append([k, v["precision"], v["recall"], v["f1-score"]])
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1"]).set_index("class")

    plt.figure(figsize=(10, 6))
    if _HAS_SNS:
        sns.heatmap(df, annot=True, cmap="Blues", fmt=".3f")
    else:
        plt.imshow(df.values, aspect="auto")
        plt.colorbar()
        plt.xticks(range(3), ["precision", "recall", "f1"], rotation=45, ha="right")
        plt.yticks(range(len(df.index)), df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                plt.text(j, i, f"{df.values[i,j]:.2f}", ha="center", va="center")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

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
    plt.show()
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
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["lr"])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title(f"{title} - LR schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_schedule.png"), dpi=300)
    plt.show()
    plt.close()


# =========================
# 11. TEMPERATURE SCALING
# =========================
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
def collect_logits_labels(model: nn.Module, loader: DataLoader, aug_eval: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for x_u8, y in tqdm(loader, desc="Collect logits", leave=False):
        x_u8 = x_u8.to(CONFIG["device"], non_blocking=True)
        y = y.to(CONFIG["device"], non_blocking=True)
        with torch.no_grad():
            x = aug_eval(x_u8)
        with torch.autocast(device_type=CONFIG["device"].type,
                            enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda")):
            logits = forward_logits(model, x)
        all_logits.append(logits.detach())
        all_labels.append(y.detach())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def fit_temperature_on_val(model: nn.Module, val_loader: DataLoader, aug_eval: nn.Module, log_path: str) -> TemperatureScaler:
    log_print(log_path, "[CAL] Fitting temperature scaling on VAL set...")
    logits, labels = collect_logits_labels(model, val_loader, aug_eval)
    scaler = TemperatureScaler().to(CONFIG["device"])
    nll_criterion = nn.CrossEntropyLoss()

    optimizer = optim.LBFGS([scaler.temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = nll_criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    log_print(log_path, f"[CAL] Learned temperature T = {scaler.get_T():.4f}")
    return scaler


# =========================
# 12. TRAINING ENGINE
# =========================
@dataclass
class RunPaths:
    out_dir: str
    log_path: str
    best_model_path: str
    history_csv: str
    summary_json: str
    adapter_path: str

def make_run_paths(dataset_name: str, num_classes: int) -> RunPaths:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(CONFIG["runs_root"], dataset_name, f"deit_small_adaptformer_frozen_{num_classes}c", ts)
    safe_mkdir(out_dir)
    return RunPaths(
        out_dir=out_dir,
        log_path=os.path.join(out_dir, "run.log"),
        best_model_path=os.path.join(out_dir, "best_model.pth"),   # ONLY BEST
        history_csv=os.path.join(out_dir, "history.csv"),
        summary_json=os.path.join(out_dir, "summary.json"),
        adapter_path=os.path.join(out_dir, "adapter_only.pth"),
    )

def create_deit_adaptformer_model(num_classes: int, log_path: str) -> nn.Module:
    log_print(log_path, f"\n[MODEL] Creating {CONFIG['model_name']} + AdaptFormer (frozen={CONFIG['freeze_backbone']})")
    model = timm.create_model(CONFIG["model_name"], pretrained=True, num_classes=num_classes)

    inject_adaptformer_into_deit_mlp(
        model,
        mid_dim=CONFIG["adapt_mid_dim"],
        scale=CONFIG["adapt_scale"],
        dropout=CONFIG["adapt_dropout"],
        start_layer=CONFIG["adapt_start_layer"],
        end_layer=CONFIG["adapt_end_layer"],
    )

    if CONFIG["freeze_backbone"]:
        freeze_backbone_keep_adapters_and_heads(model)
    else:
        # if not frozen, allow normal finetune + adapters (not your target setup)
        for p in model.parameters():
            p.requires_grad_(True)

    total, trainable, pct = count_params(model)
    log_print(log_path, f"[PARAMS] total={total:,} trainable={trainable:,} ({pct:.4f}%)")
    log_print(log_path, f"[ADAPT] mid_dim={CONFIG['adapt_mid_dim']} scale={CONFIG['adapt_scale']} "
                        f"layers={CONFIG['adapt_start_layer']}..{CONFIG['adapt_end_layer']}")
    return model.to(CONFIG["device"])

class EarlyStoppingBestOnly:
    def __init__(self, patience: int, best_path: str, log_path: str):
        self.patience = patience
        self.best_path = best_path
        self.log_path = log_path
        self.best_val = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_val is None or val_loss < self.best_val:
            self.best_val = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.best_path)
            log_print(self.log_path, f"[ES] New best val_loss={val_loss:.6f} -> saved {self.best_path}")
        else:
            self.counter += 1
            log_print(self.log_path, f"[ES] No improve {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

def build_loss_and_sampler(train_labels: List[int], num_classes: int, log_path: str):
    counts = np.bincount(np.array(train_labels), minlength=num_classes)
    log_print(log_path, f"[IMB] Train class counts: {counts.tolist()}")

    weights = None
    if CONFIG["use_class_balanced_weights"]:
        cbw = class_balanced_weights(counts, beta=CONFIG["cb_beta"])
        weights = torch.tensor(cbw, dtype=torch.float32, device=CONFIG["device"])
        log_print(log_path, f"[IMB] Class-Balanced weights (mean~1): {cbw.round(4).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=CONFIG["label_smoothing"])

    sampler = build_sampler_if_needed(train_labels, num_classes, log_path)
    return criterion, sampler

def train_one_epoch(model, loader, criterion, optimizer, aug_train: nn.Module, log_path: str):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    if CONFIG["device"].type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        from torch.amp import GradScaler
        scaler = GradScaler(enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda"))
    except Exception:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda"))

    t0 = time.time()

    for x_u8, y in tqdm(loader, desc="Train", leave=False):
        x_u8 = x_u8.to(CONFIG["device"], non_blocking=True)
        y = y.to(CONFIG["device"], non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            x = aug_train(x_u8)

        with torch.autocast(device_type=CONFIG["device"].type,
                            enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda")):
            out = model(x)
            loss = loss_for_distilled(out, y, criterion)

            if isinstance(out, (tuple, list)):
                logits = (out[0] + out[1]) / 2.0
            else:
                logits = out

        scaler.scale(loss).backward()

        if CONFIG["grad_clip_norm"] and CONFIG["grad_clip_norm"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)

    epoch_time = time.time() - t0
    peak_mem_gb = 0.0
    if CONFIG["device"].type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return total_loss / max(total, 1), correct / max(total, 1), epoch_time, peak_mem_gb

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, aug_eval: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_y, all_p = [], []

    for x_u8, y in tqdm(loader, desc="Val", leave=False):
        x_u8 = x_u8.to(CONFIG["device"], non_blocking=True)
        y = y.to(CONFIG["device"], non_blocking=True)

        with torch.no_grad():
            x = aug_eval(x_u8)

        with torch.autocast(device_type=CONFIG["device"].type,
                            enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda")):
            out = model(x)
            loss = loss_for_distilled(out, y, criterion)

            if isinstance(out, (tuple, list)):
                logits = (out[0] + out[1]) / 2.0
            else:
                logits = out

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)

        all_y.append(y.detach().cpu().numpy())
        all_p.append(pred.detach().cpu().numpy())

    y_np = np.concatenate(all_y)
    p_np = np.concatenate(all_p)
    macro_f1 = float(f1_score(y_np, p_np, average="macro"))
    return total_loss / max(total, 1), correct / max(total, 1), macro_f1

def train_engine(dataset_name: str, train_loader, val_loader, class_names: List[str], run_paths: RunPaths):
    log_path = run_paths.log_path
    num_classes = len(class_names)

    model = create_deit_adaptformer_model(num_classes, log_path)

    train_labels = train_loader.dataset.labels
    criterion, _ = build_loss_and_sampler(train_labels, num_classes, log_path)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    def lr_lambda(epoch):
        if epoch < CONFIG["warmup_epochs"]:
            return float(epoch + 1) / float(max(1, CONFIG["warmup_epochs"]))
        denom = max(1, (CONFIG["epochs"] - CONFIG["warmup_epochs"]))
        return 0.5 * (1 + math.cos(math.pi * (epoch - CONFIG["warmup_epochs"]) / denom))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    early_stop = EarlyStoppingBestOnly(CONFIG["patience"], run_paths.best_model_path, log_path)

    aug_train = GPUAugmentAndNormalize("train", CONFIG["device"])
    aug_eval  = GPUAugmentAndNormalize("eval",  CONFIG["device"])

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

    for ep in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc, ep_time, peak_vram = train_one_epoch(
            model, train_loader, criterion, optimizer, aug_train, log_path
        )
        val_loss, val_acc, val_macro_f1 = validate_one_epoch(
            model, val_loader, criterion, aug_eval
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["lr"].append(lr_now)
        history["epoch_time_sec"].append(ep_time)
        history["train_peak_vram_gb"].append(peak_vram)

        log_print(
            log_path,
            f"[{dataset_name}] Epoch {ep:03d}/{CONFIG['epochs']:03d} | "
            f"lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% val_macroF1={val_macro_f1*100:.2f}% | "
            f"ep_time={ep_time:.1f}s peakVRAM={peak_vram:.2f}GB"
        )

        early_stop(val_loss, model)
        if early_stop.stop:
            log_print(log_path, f"[{dataset_name}] Early stopping triggered.")
            break

    total_train_time = time.time() - total_train_t0

    pd.DataFrame(history).to_csv(run_paths.history_csv, index=False)

    if os.path.exists(run_paths.best_model_path):
        model.load_state_dict(torch.load(run_paths.best_model_path, map_location=CONFIG["device"]))

    adapter_bytes = None
    if CONFIG["save_adapter_only"]:
        adapter_sd = get_adapter_state_dict(model)
        torch.save(adapter_sd, run_paths.adapter_path)
        adapter_bytes = os.path.getsize(run_paths.adapter_path)

    total_p, trainable_p, trainable_pct = count_params(model)
    return model, history, total_train_time, (total_p, trainable_p, trainable_pct), adapter_bytes


# =========================
# 13. EVALUATION + TSNE + BENCH
# =========================
@torch.no_grad()
def evaluate_on_loader(model: nn.Module, loader: DataLoader, class_names: List[str], aug_eval: nn.Module) -> Dict:
    model.eval()
    all_y, all_pred, all_probs, all_logits = [], [], [], []

    for x_u8, y in tqdm(loader, desc="Test eval", leave=False):
        x_u8 = x_u8.to(CONFIG["device"], non_blocking=True)
        y = y.to(CONFIG["device"], non_blocking=True)

        with torch.no_grad():
            x = aug_eval(x_u8)

        with torch.autocast(device_type=CONFIG["device"].type,
                            enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda")):
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
    except Exception:
        macro_auc = None

    ece = compute_ece(probs_np, y_np, n_bins=15)
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
    aug_eval: nn.Module,
    samples_per_class: int = 200,
):
    if not hasattr(model, "forward_features"):
        print("[WARN] model has no forward_features(); skipping t-SNE.")
        return

    model.eval()
    num_classes = len(class_names)

    feats_by_class = {i: [] for i in range(num_classes)}
    labels_by_class = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for x_u8, y in tqdm(loader, desc="t-SNE feats", leave=False):
            x_u8 = x_u8.to(CONFIG["device"], non_blocking=True)
            x = aug_eval(x_u8)

            with torch.autocast(device_type=CONFIG["device"].type,
                                enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda")):
                f = model.forward_features(x)

            if f.ndim == 3:
                f = f[:, 0]  # CLS token feature

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
        print("[WARN] Not enough samples for t-SNE.")
        return

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    print(f"[INFO] t-SNE using {len(X)} samples total.")
    tsne = TSNE(n_components=2, random_state=CONFIG["seed"], perplexity=min(30, len(X) - 1))
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for c in range(num_classes):
        mask = (y == c)
        if not np.any(mask):
            continue
        plt.scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.8, label=class_names[c])
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

@torch.no_grad()
def benchmark_inference(model: nn.Module, sample_u8: torch.Tensor, batch_size: int, aug_eval: nn.Module) -> Dict:
    model.eval()
    x_u8 = sample_u8[:batch_size].contiguous().to(CONFIG["device"], non_blocking=True)
    x = aug_eval(x_u8)

    if CONFIG["device"].type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(CONFIG["bench_warmup_iters"]):
        with torch.autocast(device_type=CONFIG["device"].type,
                            enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda")):
            _ = forward_logits(model, x)
    if CONFIG["device"].type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(CONFIG["bench_timed_iters"]):
        t0 = time.perf_counter()
        with torch.autocast(device_type=CONFIG["device"].type,
                            enabled=(CONFIG["amp_enabled"] and CONFIG["device"].type == "cuda")):
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


# =========================
# 14. DATALOADER BUILDER
# =========================
def build_dataloaders(dataset_name: str, ds_cfg: Dict, run_paths: RunPaths) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    root = ds_cfg["root"]
    pad_sq = bool(ds_cfg.get("pad_to_square", False))
    cache = bool(ds_cfg.get("cache_in_ram", True))

    train_dir = os.path.join(root, "train")
    class_names = list_class_names(train_dir)

    df_stats = dataset_stats_table(root, class_names)
    print_and_save_dataset_stats(df_stats, run_paths.out_dir, run_paths.log_path)

    train_ds = OCTCachedFolderDataset(root, "train", class_names, CONFIG["img_size"], pad_sq, cache, run_paths.log_path)
    val_ds   = OCTCachedFolderDataset(root, "val",   class_names, CONFIG["img_size"], pad_sq, cache, run_paths.log_path)
    test_ds  = OCTCachedFolderDataset(root, "test",  class_names, CONFIG["img_size"], pad_sq, cache, run_paths.log_path)

    _, sampler = build_loss_and_sampler(train_ds.labels, len(class_names), run_paths.log_path)

    shuffle = (sampler is None)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(CONFIG["batch_size"], len(train_ds)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(CONFIG["batch_size"], len(val_ds)),
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=min(CONFIG["batch_size"], len(test_ds)),
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        drop_last=False,
    )

    log_print(run_paths.log_path, f"\n[DATASET] {dataset_name}")
    log_print(run_paths.log_path, f" - root: {root}")
    log_print(run_paths.log_path, f" - pad_to_square: {pad_sq}")
    log_print(run_paths.log_path, f" - cache_in_ram: {cache}")
    log_print(run_paths.log_path, f" - classes ({len(class_names)}): {class_names}")
    log_print(run_paths.log_path, f" - split sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names


# =========================
# 15. MAIN (MULTI-DATASET)
# =========================
def main():
    print("\n" + "=" * 100)
    print("DeiT-Small + AdaptFormer (Frozen Backbone) OCT Pipeline")
    print("=" * 100)
    print(f"[ENV] torch={torch.__version__} cuda={torch.cuda.is_available()} device={CONFIG['device']}")
    if 'torchvision' in globals() and hasattr(torchvision, "__version__"):
        print(f"[ENV] torchvision={torchvision.__version__} tv2={_HAS_TV2} kornia={_HAS_KORNIA}")
    print(f"[CFG] model={CONFIG['model_name']} img_size={CONFIG['img_size']} batch={CONFIG['batch_size']} "
          f"epochs={CONFIG['epochs']} amp={CONFIG['amp_enabled']}")
    print(f"[CFG] AdaptFormer mid_dim={CONFIG['adapt_mid_dim']} scale={CONFIG['adapt_scale']} "
          f"layers={CONFIG['adapt_start_layer']}..{CONFIG['adapt_end_layer']}")
    print("=" * 100)

    if len(RUN_DATASETS) == 0:
        raise RuntimeError("RUN_DATASETS is empty. Fill DATASET_CONFIGS first.")

    for ds_name in RUN_DATASETS:
        if ds_name not in DATASET_CONFIGS:
            print(f"[SKIP] Unknown dataset '{ds_name}'")
            continue

        tmp_train_dir = os.path.join(DATASET_CONFIGS[ds_name]["root"], "train")
        tmp_classes = list_class_names(tmp_train_dir)
        run_paths = make_run_paths(ds_name, num_classes=len(tmp_classes))

        with open(os.path.join(run_paths.out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"CONFIG": CONFIG, "DATASET_CFG": DATASET_CONFIGS[ds_name]},
                f,
                indent=2,
                default=_json_default,
            )

        log_print(run_paths.log_path, f"\n=== DATASET RUN START: {ds_name} ===")

        train_loader, val_loader, test_loader, class_names = build_dataloaders(ds_name, DATASET_CONFIGS[ds_name], run_paths)

        model, history, total_train_time, param_stats, adapter_bytes = train_engine(
            ds_name, train_loader, val_loader, class_names, run_paths
        )

        plot_training_curves(history, title=f"{ds_name} DeiT-S AdaptFormer Frozen", out_dir=run_paths.out_dir)

        aug_eval = GPUAugmentAndNormalize("eval", CONFIG["device"])

        test_metrics = evaluate_on_loader(model, test_loader, class_names, aug_eval=aug_eval)

        T = None
        test_metrics_cal = None
        if CONFIG["use_temperature_scaling"]:
            ts_model = fit_temperature_on_val(model, val_loader, aug_eval=aug_eval, log_path=run_paths.log_path)

            logits = test_metrics["logits"].to(CONFIG["device"])
            with torch.no_grad():
                scaled_logits = ts_model(logits)
                probs_scaled = torch.softmax(scaled_logits, dim=1).detach().cpu().numpy()

            y = test_metrics["labels"]
            ece = compute_ece(probs_scaled, y, n_bins=15)
            nll = compute_nll(probs_scaled, y)
            brier = compute_brier(probs_scaled, y, num_classes=len(class_names))
            T = ts_model.get_T()
            test_metrics_cal = {"ece": ece, "nll": nll, "brier": brier}

            plot_reliability_diagram(
                probs_scaled, y, n_bins=15,
                title=f"{ds_name} Reliability (Temp-scaled, T={T:.3f})",
                save_path=os.path.join(run_paths.out_dir, "reliability_temp_scaled.png")
            )

        y = test_metrics["labels"]
        p = test_metrics["preds"]
        probs = test_metrics["probs"]

        rep_str = classification_report(y, p, target_names=class_names, digits=4, zero_division=0)
        rep_dict = classification_report(y, p, target_names=class_names, digits=4, zero_division=0, output_dict=True)

        log_print(run_paths.log_path, "\n" + "-" * 90)
        log_print(run_paths.log_path, f"[{ds_name}] TEST classification report (uncalibrated)")
        log_print(run_paths.log_path, "-" * 90)
        log_print(run_paths.log_path, rep_str)

        pd.DataFrame(rep_dict).transpose().to_csv(os.path.join(run_paths.out_dir, "classification_report.csv"))

        cm = confusion_matrix(y, p)
        np.savetxt(os.path.join(run_paths.out_dir, "confusion_matrix_raw.txt"), cm, fmt="%d")

        plot_confusion(cm, class_names, f"{ds_name} Confusion (normalized)", os.path.join(run_paths.out_dir, "confusion_norm.png"))
        plot_classification_report_heatmap(rep_dict, f"{ds_name} Class report heatmap", os.path.join(run_paths.out_dir, "cls_report_heatmap.png"))
        plot_sens_spec(cm, class_names, f"{ds_name} Sensitivity/Specificity", os.path.join(run_paths.out_dir, "sens_spec.png"))
        plot_roc_curves(y, probs, class_names, f"{ds_name} ROC (OVR)", os.path.join(run_paths.out_dir, "roc.png"))

        plot_reliability_diagram(
            probs, y, n_bins=15,
            title=f"{ds_name} Reliability (uncalibrated)",
            save_path=os.path.join(run_paths.out_dir, "reliability_uncalibrated.png")
        )

        plot_tsne_stratified(
            model,
            test_loader,
            class_names,
            title=f"{ds_name} t-SNE (CLS features)",
            save_path=os.path.join(run_paths.out_dir, "tsne.png"),
            aug_eval=aug_eval,
            samples_per_class=200
        )

        # AdaptFormer has no "merge like LoRA"; benchmark as-is
        x0_u8, _ = next(iter(test_loader))
        bench_lat = benchmark_inference(model, x0_u8, batch_size=CONFIG["bench_latency_batch"], aug_eval=aug_eval)
        bench_thr = benchmark_inference(model, x0_u8, batch_size=min(CONFIG["bench_throughput_batch"], x0_u8.size(0)), aug_eval=aug_eval)

        total_p, trainable_p, trainable_pct = param_stats
        best_val_loss = float(np.min(history["val_loss"]))
        best_val_f1 = float(np.max(history["val_macro_f1"]))
        mean_epoch_time = float(np.mean(history["epoch_time_sec"]))
        peak_train_vram = float(np.max(history["train_peak_vram_gb"]))

        summary = {
            "dataset": ds_name,
            "root": DATASET_CONFIGS[ds_name]["root"],
            "classes": class_names,
            "model": CONFIG["model_name"],
            "peft": "AdaptFormer (AdaptMLP in MLP, frozen backbone)",
            "adaptformer": {
                "mid_dim": int(CONFIG["adapt_mid_dim"]),
                "scale": float(CONFIG["adapt_scale"]),
                "dropout": float(CONFIG["adapt_dropout"]),
                "layers": [int(CONFIG["adapt_start_layer"]),
                           (None if CONFIG["adapt_end_layer"] is None else int(CONFIG["adapt_end_layer"]))],
            },
            "params": {
                "total": int(total_p),
                "trainable": int(trainable_p),
                "trainable_pct": float(trainable_pct),
            },
            "storage": {
                "best_model_path": run_paths.best_model_path,
                "adapter_only_enabled": bool(CONFIG["save_adapter_only"]),
                "adapter_only_bytes": int(adapter_bytes) if adapter_bytes is not None else None,
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
            json.dump(summary, f, indent=2, default=_json_default)

        log_print(run_paths.log_path, "\n" + "=" * 90)
        log_print(run_paths.log_path, f"[{ds_name}] TEST summary (uncalibrated): "
                                     f"Acc={test_metrics['acc']*100:.2f}% "
                                     f"MacroF1={test_metrics['macro_f1']*100:.2f}% "
                                     f"ECE={test_metrics['ece']:.4f} NLL={test_metrics['nll']:.4f} "
                                     f"Brier={test_metrics['brier']:.4f}")
        if test_metrics["macro_auc"] is not None:
            log_print(run_paths.log_path, f"[{ds_name}] Macro-AUC={test_metrics['macro_auc']:.4f}")
        if test_metrics_cal is not None:
            log_print(run_paths.log_path, f"[{ds_name}] TEMP-SCALED (T={T:.3f}): "
                                         f"ECE={test_metrics_cal['ece']:.4f} "
                                         f"NLL={test_metrics_cal['nll']:.4f} "
                                         f"Brier={test_metrics_cal['brier']:.4f}")
        log_print(run_paths.log_path, f"[{ds_name}] Params trainable: {trainable_p:,} ({trainable_pct:.4f}%)")
        if adapter_bytes:
            log_print(run_paths.log_path, f"[{ds_name}] Adapter-only size: {adapter_bytes/(1024**2):.2f} MB")
        log_print(run_paths.log_path, f"[{ds_name}] Inference latency (B=1): {bench_lat['ms_per_img']:.3f} ms/img "
                                     f"({bench_lat['imgs_per_s']:.1f} imgs/s), peakVRAM={bench_lat['peak_vram_gb']:.2f}GB")
        log_print(run_paths.log_path, f"[{ds_name}] Inference throughput (B={bench_thr['batch']}): "
                                     f"{bench_thr['imgs_per_s']:.1f} imgs/s, peakVRAM={bench_thr['peak_vram_gb']:.2f}GB")
        log_print(run_paths.log_path, f"[{ds_name}] Outputs saved in: {run_paths.out_dir}")
        log_print(run_paths.log_path, "=" * 90 + "\n")

        # ===== memory cleanup =====
        del model
        del train_loader, val_loader, test_loader
        del history, test_metrics, cm, rep_str, rep_dict
        del x0_u8

        if "ts_model" in locals():
            del ts_model
        if "logits" in locals():
            del logits
        if "scaled_logits" in locals():
            del scaled_logits

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        print(f"[CLEANUP] Finished dataset '{ds_name}'. RAM/VRAM cleared.\n")

if __name__ == "__main__":
    main()
