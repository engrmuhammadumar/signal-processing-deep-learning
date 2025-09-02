# CWT Fault Classification — Conference-Ready Mini Project

End‑to‑end pipeline for centrifugal pump fault classification using **CWT images**:
- Efficient transfer learning (ResNet‑18 by default) with **label smoothing**, optional **MixUp/CutMix**, cosine LR, **early stopping**.
- Stratified train/val/test split (reproducible).
- Test‑time translation augmentation (TTA) for the final score.
- Saves: best checkpoint, predictions CSV, confusion matrix PNG, training curves PNG, logs CSV.
- **Grad‑CAM** script to generate heatmaps for paper figures.
- **Paper tables** script to convert metrics CSV → LaTeX tables.

> Works on Windows (CPU) and CUDA GPUs. On Windows CPU, if you see multiprocessing issues, add `--num_workers 0` to commands.

## 1) Environment

Install Python 3.10+ and then:
```bash
pip install -r requirements.txt
```

## 2) Dataset Layout

Expect a folder like:
```
E:\CP Dataset\cwt\
├─ Impeller (3.0BAR)\
├─ Mechanical seal Hole (3BAR)\
├─ Mechanical seal Scratch (3.0BAR)\
└─ Normal (3BAR)\
```

Each subfolder contains CWT images (png/jpg).

## 3) Quick Start (CPU, Windows)

```powershell
# Train + evaluate + save artifacts to .\runs\exp1
python train_resnet_cwt.py --data_root "E:\CP Dataset\cwt" --pretrained --epochs 25 --img_size 224 --batch_size 32 --tta --save_dir ".\runs\exp1" --num_workers 0
```

If you have a CUDA GPU:
```powershell
python train_resnet_cwt.py --data_root "E:\CP Dataset\cwt" --pretrained --epochs 25 --img_size 224 --batch_size 64 --tta --amp --save_dir ".\runs\exp1"
```

Useful knobs to try if **Hole vs Scratch** is tricky:
```powershell
# add mixing augs (often boosts recall on confusing classes)
--mixup_alpha 0.2 --cutmix_alpha 0.2
# slightly larger input
--img_size 256
# more training if stable
--epochs 40
```

## 4) Grad‑CAM for Paper Figures

Pick a few test images (or let the script auto‑sample) and run:
```powershell
python gradcam_cwt.py --data_root "E:\CP Dataset\cwt" --ckpt ".\runs\exp1\best.pth" --save_dir ".\runs\exp1\gradcam" --num_samples 3 --num_workers 0
```

Outputs heatmaps and overlays per class.

## 5) Export Paper Tables

Convert the metrics log to a LaTeX table:
```powershell
python paper_tables.py --log_csv ".\runs\exp1\metrics_log.csv" --out_tex ".\runs\exp1\metrics_table.tex"
```

## 6) What to Report in Your Paper

**Dataset & Preprocessing**: class names, counts, CWT generation description, image size, normalization.**Augmentations**: brightness/contrast jitter, small time/frequency masks, MixUp/CutMix (if used).**Backbone**: ResNet‑18 (pretrained ImageNet), classifier replaced with 4‑way FC.**Training**: optimizer (AdamW), lr=3e‑4, cosine decay, label smoothing 0.05, early stopping on val macro‑F1 (patience=7).**Split**: stratified 70/15/15 (report exact numbers).**Metrics**: accuracy, macro‑F1, per‑class P/R/F1; confusion matrix.**Ablations**: pretrained vs scratch, +MixUp/CutMix, image size 224 vs 256, with/without TTA.**Interpretability**: Grad‑CAM examples per class showing discriminative CWT regions.**Limitations**: Hole vs Scratch ambiguity; overfitting risks; single‑site data.**Reproducibility**: seed=1337; code + exact command lines above.

Good luck — this setup is designed to be simple, fast, and publishable.