# SAM3 Improving on PASCAL VOC 2012

**CS331 – Advanced Computer Vision | University of Information Technology (UIT)**

This repository contains the code and results for improving semantic segmentation quality on PASCAL VOC 2012 by targeting 5 weak classes of SAM3:
`diningtable`, `sofa`, `chair`, `bicycle`, `pottedplant`.

Two independent approaches are implemented:

| Approach | Method | Author |
|---|---|---|
| **LoRA fine-tuning** (this folder) | Direct LoRA adaptation of SAM3's Text Encoder + Mask Decoder on 4 target classes | Đào Mạnh Dũng |
| **UNet+ASPP refinement** (`Adjust_SAM3_on_5weakclass_pascalvoc2012-main/`) | External refinement module (ResNet-34 + ASPP + UNet decoder) + DINOv2 features + dual-prompt for pottedplant | Mai Xuân Tuấn |

Both methods are evaluated on the full VOC 2012 val set (1449 images, 20 foreground classes).

---

## Results (VOC 2012 val — mIoU 20-class)

| Method | mIoU 20fg | Δ vs SAM3 base |
|---|---|---|
| SAM3 zero-shot baseline | 0.8112 | — |
| UNet+ASPP (OLD) | 0.8414 | +0.0302 |
| UNet+ASPP v2 (multi-class) | 0.8345 | +0.0233 |
| UNet+ASPP v3 + DINOv2 | 0.8426 | +0.0314 |
| **LoRA fine-tune SAM3** | **0.8470** | **+0.0358** |

LoRA achieves 4/4 target classes above SAM3 baseline (no class regresses).

---

## Repository Structure

```
sam3_pascalvoc2012/
├── finetuningsam3.ipynb                  # LoRA fine-tuning pipeline (v19, main training notebook)
├── eval.ipynb                            # Evaluation: SAM3 base vs LoRA FT on full val (1449 images)
├── test_all_onkaggle.ipynb               # SAM3 zero-shot baseline evaluation on VOC2012 val (Kaggle)
├── Adjust_SAM3_on_5weakclass_pascalvoc2012-main/
│   ├── train_unet_aspp_just4.ipynb       # UNet+ASPP OLD training
│   ├── train_unet_aspp_new_v2.ipynb      # UNet+ASPP v2 training
│   ├── train-unetaspp-dino.ipynb         # UNet+ASPP v3 + DINOv2 training
│   ├── test-all-with-unetaspp-just4.ipynb
│   ├── test-all-with-unetaspp-v2-new.ipynb
│   ├── test-all-unetaspp-dino.ipynb      # Full val evaluation (all 3 UNet methods)
│   ├── streamlit_app/                    # Interactive demo (SAM3 + UNet/LoRA inference)
│   ├── Dockerfile                        # Docker environment for the demo app
│   └── README.md                         # Docker/demo setup instructions
└── README.md                             # This file
```

---

## 1. Dataset — PASCAL VOC 2012

Download from the official source:

```
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
```

Download **VOCtrainval_11-May-2012.tar** (training/validation data, ~2 GB) and extract it.

Or from Kaggle:

```
https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012
```

Expected structure after extraction:

```
VOCdevkit/
└── VOC2012/
    ├── JPEGImages/          # RGB images (.jpg)
    ├── SegmentationClass/   # Ground-truth masks (.png, 21 labels: 0=bg, 1-20=classes, 255=ignore)
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

Set `VOC_ROOT` in each notebook to the path of `VOCdevkit/VOC2012`.

For the LoRA training notebook, it filters images containing at least one of the 4 target classes (`bicycle`, `chair`, `diningtable`, `sofa`) — resulting in **459 training samples** (388 positive + 71 hard-negatives for chair/sofa).

---

## 2. SAM3 Model Weights

Install SAM3 from source (no pip package):

```bash
pip install --no-deps git+https://github.com/facebookresearch/sam3.git
pip install timm ftfy iopath portalocker peft albumentations safetensors
```

Download the SAM3 checkpoint (`sam3.pt`, ~1.6 GB) from Hugging Face (access request required):

```
https://huggingface.co/facebook/sam3
```

Or from Kaggle (if available):

```
https://www.kaggle.com/datasets/dngomnh/modelsam3
```

Place `sam3.pt` at the project root or update the `SAM3_CKPT` path in the notebooks.

---

## 3. LoRA Adapter Weights

The fine-tuned LoRA adapter weights (epoch 14, best mini-val mIoU = 0.6601) are **not included in this repo** due to file size.

Download from Kaggle:

```
https://www.kaggle.com/datasets/trungkienksd/sam3-finetune
```

Expected structure after download:

```
model_finetune/
├── adapter_config.json
└── adapter_model.safetensors
```

---

## 4. Running the LoRA Fine-tuning

Open `finetuningsam3.ipynb` on Kaggle (GPU required — tested on P100/T4).

**Key paths to configure (Cell 2):**

```python
VOC_ROOT      = "/kaggle/input/.../VOCdevkit/VOC2012"
SAM3_CKPT     = "/kaggle/input/.../sam3.pt"
ADAPTER_SAVE  = "/kaggle/working/model_finetune"
```

**Training configuration (v19):**

| Setting | Value |
|---|---|
| LoRA — Text Encoder | rank=4, α=8, dropout=0.15 |
| LoRA — Mask Decoder | rank=16, α=32, dropout=0.05 |
| Image Encoder | Frozen (not updated) |
| Optimizer | AdamW (Decoder lr=1e-4, Encoder lr=2e-5) |
| Scheduler | CosineAnnealingLR |
| Effective batch size | 1 × grad_accum 8 = BS 8 |
| Precision | float32 |
| Max epochs | 25, early stop patience=5 |
| Best checkpoint | epoch 14 |
| Trainable params | 2,561,440 / 843,071,190 (0.30%) |

**Loss function:**
- Positive samples: `0.50 × Dice + 0.35 × FocalBCE (γ=2) + λ_iou × IoUScore`
- Negative samples (hard-neg for chair/sofa only): `1.5 × FocalBCE` — no Dice, no score-head supervision

---

## 5. Running the Evaluation

Open `eval.ipynb` on Kaggle.

**Key paths to configure:**

```python
VOC_ROOT       = "/kaggle/input/.../VOCdevkit/VOC2012"
SAM3_CKPT      = "/kaggle/input/.../sam3.pt"
FT_ADAPTER_DIR = "/kaggle/input/.../model_finetune"
```

The notebook evaluates two pipelines side-by-side on all 1449 val images:

- **BASE**: SAM3 zero-shot, 20 classes, raw class-name prompts.
- **FT**: 4 target classes → SAM3 LoRA fine-tuned (conf=0.3); `pottedplant` → dual-prompt union (`'potted plant with its pot or container'` + `'flowerpot'`); 15 other classes → identical to BASE.

Output: per-class IoU table + bar chart.

> Note: Full evaluation takes ~8.5 hours on a Kaggle P100 (1449 images × 20 classes × 2 model passes).

---

## 6. SAM3 Baseline Evaluation (Zero-shot)

Open `test_all_onkaggle.ipynb` on Kaggle to reproduce the SAM3 zero-shot baseline result.

**Key paths to configure (Cell 5):**

```python
VOC_ROOT = Path("/kaggle/input/.../VOC2012")
```

The notebook evaluates native SAM3 (no fine-tuning) on all 1449 val images using raw class-name text prompts for all 20 VOC classes. Score-based conflict resolution is applied when multiple classes overlap on the same pixel.

Output: per-class IoU table + bar chart + per-class visualization (Original | GT | Prediction | Error Overlay).

> Note: Full evaluation takes ~8.5 hours on Kaggle P100 (same as `eval.ipynb`).

---

## 7. UNet+ASPP Methods (Partner's Approach)

See `Adjust_SAM3_on_5weakclass_pascalvoc2012-main/README.md` for full Docker-based setup and demo instructions.

**Training notebooks:**

| Notebook | Method |
|---|---|
| `train_unet_aspp_just4.ipynb` | OLD — binary sigmoid, 9-channel input (RGB + coarse mask + co-occurrence + 4×one-hot) |
| `train_unet_aspp_new_v2.ipynb` | v2 — multi-class softmax, 8-channel input (RGB + 4×coarse masks + co-occurrence) |
| `train-unetaspp-dino.ipynb` | v3 — v2 + DINOv2 frozen ViT-S/14 features (40-channel input) |

**Evaluation notebooks:**

| Notebook | Evaluates |
|---|---|
| `test-all-with-unetaspp-just4.ipynb` | OLD |
| `test-all-with-unetaspp-v2-new.ipynb` | v2 |
| `test-all-unetaspp-dino.ipynb` | v3 (DINOv2) — all 3 UNet methods |

UNet+ASPP checkpoints available at:

```
https://drive.google.com/drive/folders/1xPDGTGXI1LbXVc6BKm4re0X-RnsPGxB9?usp=sharing
```

---

## 8. Interactive Demo (Streamlit)

`Adjust_SAM3_on_5weakclass_pascalvoc2012-main/streamlit_app/` provides a Streamlit UI supporting all inference modes.

**Quick start (Docker):**

```bash
cd Adjust_SAM3_on_5weakclass_pascalvoc2012-main

# CPU mode
docker run --rm -p 8501:8501 -v "${PWD}/weights:/weights" sam3-voc2012

# GPU mode
docker run --rm --gpus all -p 8501:8501 -v "${PWD}/weights:/weights" sam3-voc2012
```

Open `http://localhost:8501`. See partner README for weight paths inside the container.

---

## Dependencies

```bash
pip install --no-deps git+https://github.com/facebookresearch/sam3.git
pip install timm ftfy iopath portalocker peft albumentations safetensors
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pillow numpy pandas matplotlib tqdm
```

Python 3.10+, PyTorch 2.x, CUDA 12.x recommended.

---

## Team

| Student ID | Name | Task |
|---|---|---|
| 23520325 | Đào Mạnh Dũng | LoRA fine-tuning SAM3 (training + evaluation) |
| 23521714 | Mai Xuân Tuấn | UNet+ASPP refinement + DINOv2 + Streamlit demo |
