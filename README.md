# Parameter-Efficient ViT for Retinal OCT Classification (PEFT Benchmark)

Official implementation and experiment artifacts for:

**Parameter-Efficient Adaptation of Vision Transformers for Retinal OCT Classification**  
Md. Sazib Ahmed, Firoz Ahmed — *2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)*

This repository benchmarks **parameter-efficient fine-tuning (PEFT)** methods for adapting a **DeiT-Small Vision Transformer** to retinal **Optical Coherence Tomography (OCT)** classification across **five public datasets**:
- **OCT2017**, **NEH-UT**, **Srinivasan2014**, **THOCT1800**, and **OCT-C8**

We compare:
- **Full fine-tuning**
- **Linear head-only tuning (linear probe)**
- **VPT-Deep (Visual Prompt Tuning)**
- **AdaptFormer (FFN adapters)**
- **LoRA (Low-Rank Adaptation)**

> ⚠️ **Disclaimer:** This code is for research and reproducibility. It is **not** a certified medical device and should **not** be used for clinical decision-making.

---

## Key Findings (from the paper)

- PEFT methods reduce trainable parameters to **<3%** while remaining competitive with full fine-tuning.
- On the **8-class OCT-C8** benchmark:
  - **AdaptFormer** achieves **97.6% Macro-F1**
  - **LoRA (frozen backbone)** achieves **96.9% Macro-F1** while training ~**0.7%** of parameters
  - Full fine-tuning achieves **95.6% Macro-F1**
- Calibration is preserved or improved using PEFT (ECE typically ~7–9% range in the paper).

---

## Repository Structure

This repo is organized around **dataset cleaning + reporting** and **model training notebooks/scripts**.

```
Dataset/
  <DATASET_NAME>/
    dataset_cleaning.ipynb
    dataset_duplicate_check.ipynb
    final_manifest.csv
    reports/              # before/after counts, split summaries, manifests
    dup_reports/          # duplicate detection outputs (sha256/phash, etc.)

Model/
  <EXPERIMENT_FOLDER>/
    *.ipynb               # experiment notebooks
    *.py                  # (some folders include a standalone python script)
    runs/
      <DATASET>/<RUN_NAME>/<TIMESTAMP>/
        config.json
        history.csv
        summary.json
        classification_report.csv
        confusion_norm.png
        roc.png
        reliability_*.png
        ...
```

> Notes:
> - The repo contains many **experiment outputs** under `runs/`. If you want a lighter repo, consider excluding `runs/` from version control or using Git LFS.

---

## Datasets

### Datasets used (cleaned totals and final splits)

| Dataset | Classes | Original Total | Cleaned Total | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|
| OCT2017 | 4 | 109,309 | 101,174 | 90,158 | 10,016 | 1,000 |
| NEH-UT | 3 | 16,810 | 16,663 | 11,504 | 2,636 | 2,523 |
| Srinivasan2014 | 3 | 3,231 | 3,231 | 2,261 | 485 | 485 |
| THOCT1800 | 3 | 1,800 | 1,730 | 1,210 | 260 | 260 |
| OCT-C8 | 8 | 24,000 | 23,850 | 18,260 | 2,793 | 2,797 |

### Cleaning & leakage control (high level)

- Exact duplicates removed via **SHA-256 hashing**.
- Conflicting-label duplicates discarded.
- Cross-split duplicates resolved by prioritizing **Test > Val > Train** (for pre-partitioned datasets).
- **NEH-UT** uses **patient-wise splits** (when patient IDs are available).
- Other datasets without patient IDs use **image-level stratified splits**.

### Expected data format

Training assumes an **ImageFolder-style** dataset layout (class subfolders). You will likely need to point notebooks/scripts to the location where you store raw dataset images (not included in this repository).

Example (illustrative):
```
data/OCT2017/train/<class_name>/*.jpg
data/OCT2017/val/<class_name>/*.jpg
data/OCT2017/test/<class_name>/*.jpg
```

---

## Methods

### Backbone

- **DeiT-Small (distilled)**, pretrained on ImageNet-1k  
  Model id used in the paper: `deit_small_distilled_patch16_224`
- Logits from **CLS** and **distillation head** are averaged at inference.

### PEFT configurations (paper settings)

- **LoRA**: applied to attention **QKV**, `r=8`, `α=16`, `dropout=0.10` (backbone frozen; merged for inference)
- **VPT-Deep**: `20` prompt tokens per Transformer block, init `σ=0.02` (backbone frozen)
- **AdaptFormer**: FFN parallel adapters, bottleneck dim `64`, scale `s=0.10` (backbone frozen)

---

## Training & Evaluation (paper settings)

- Optimizer: **AdamW** (`lr=5e-4`, `weight_decay=1e-2`)
- Scheduler: **cosine** with **10-epoch warmup**
- Batch size: `64`
- Max epochs: `100`
- Early stopping: patience `10` (based on validation loss)
- Regularization:
  - label smoothing `ε=0.1`
  - gradient clipping `||g||2 <= 1.0`
  - AMP enabled
- Imbalance handling:
  - class-balanced loss weighting (`β=0.9999`)
  - weighted sampling when needed
- Metrics:
  - Accuracy, **Macro-F1**, Macro-AUC (OvR)
  - Calibration: **ECE (15 bins)**, NLL, Brier
  - Temperature scaling fit on validation set (minimizing NLL)

---

## Quickstart

### 1) Clone
```bash
git clone <your-repo-url>.git
cd peft-vit-retinal-oct
```

### 2) Create an environment

This repo includes environment exports that reflect the author’s machine.

**Option A (explicit spec, Windows-friendly)**
```bash
conda create -n peft-oct --file hlab_update.explicit.txt
conda activate peft-oct
```

**Option B (history-based YAML)**
```bash
conda env create -n peft-oct -f hlab_update.history.yml
conda activate peft-oct
```

**Option C (recommended: create your own minimal env)**
Create a clean environment and install the essentials (PyTorch + CUDA, torchvision, timm, numpy/pandas, scikit-learn, matplotlib, jupyter, etc.).  
This option is often more reliable than recreating a large “base” environment.

> If you install PyTorch manually, use the correct CUDA build for your GPU/driver.

### 3) Download datasets

Raw datasets are **not** distributed here. Please download each dataset from its official/public source and follow the dataset’s license/terms.

### 4) Dataset cleaning + manifests

For each dataset under `Dataset/<NAME>/`:

1. Run:
   - `dataset_duplicate_check.ipynb`
   - `dataset_cleaning.ipynb`

2. Outputs:
   - `final_manifest.csv`
   - `reports/` and `dup_reports/` summaries

These artifacts capture how duplicates/conflicts were handled and how splits were produced.

### 5) Run experiments (training / evaluation)

Experiments are organized by method under `Model/`.

Typical workflow:
1. Open Jupyter:
   ```bash
   jupyter lab
   ```
2. Navigate to a method folder (example):
   - `Model/ViT + LoRA (frozen)/ViT + LoRA (frozen).ipynb`
3. Set dataset paths and run the notebook.

> Windows tip: many folders include spaces and parentheses—use quotes when referencing paths in terminals/scripts.

### 6) Outputs

Each run writes to a timestamped directory under:
```
Model/<METHOD>/runs/<DATASET>/<RUN_NAME>/<TIMESTAMP>/
```

Common outputs include:
- `config.json` (run settings)
- `history.csv` (training curves)
- `summary.json` (key metrics)
- `classification_report.csv`
- confusion matrix, ROC, calibration/reliability plots, t-SNE plots, logs

---

## Reproducing Paper Tables / Figures

- **Table I (dataset stats & cleaning):** derived from `Dataset/*/reports/` and `final_manifest.csv`
- **Table III (main results):** see each run’s `summary.json` / `classification_report.csv`
- **Combined summaries:** folders include notebooks like “combine results of json summary.ipynb” which aggregate run metrics into a single JSON summary.
- **Efficiency metrics (Table IV):** reported in the paper; some runs include throughput/latency logs and config snapshots.

---

## Hardware / Platform Used (author’s run)

Experiments were run on a single-GPU workstation (Windows 11 Pro) with:
- Intel Core i7-10700 CPU @ 2.90GHz
- NVIDIA GeForce RTX 5060 Ti (~16GB)
- ~32 GB RAM

(See `System Information Report.html` for full details.)

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ahmed2026peftoct,
  title     = {Parameter-Efficient Adaptation of Vision Transformers for Retinal OCT Classification},
  author    = {Ahmed, Md. Sazib and Ahmed, Firoz},
  booktitle = {2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)},
  year      = {2026},
  address   = {Chittagong, Bangladesh},
  month     = apr
}
```

---

## License

Add your intended license here (e.g., MIT, Apache-2.0).  
If you plan to distribute trained weights/adapters, ensure dataset licensing allows it.

---

## Acknowledgements

This research was facilitated through the MScCS program at **American International University-Bangladesh (AIUB)**.

---

## Contact

- Md. Sazib Ahmed: 25-93664-1@student.aiub.edu  
- Firoz Ahmed: fahmed@aiub.edu
