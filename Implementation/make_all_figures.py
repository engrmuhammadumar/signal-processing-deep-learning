# make_all_figures.py
# All-in-one: confusion matrix, ROC (per-class), t-SNE, training curves, and a full report.
# UMAR style: bold fonts; CM/TSNE no grid; ROC + training have grids.
#
# Run:
#   python make_all_figures.py --data_dir "E:\CP Dataset\WCA3" --checkpoints checkpoints --img_size 256 --tta 8 --outdir figures
#   # add --grayscale if images are single-channel
#   # optionally choose which model history to plot: --history_model cnn2

import os, json, argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Global style (bold, paper-like)
# ---------------------------
def set_base_style():
    sns.set_style("white")       # clean background, no grid by default
    sns.set_context("talk")
    plt.rcParams.update({
        "savefig.dpi": 1000,     # export HiDPI
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.weight": "bold",
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.grid": False
    })

# ---------------------------
# Models (same as training)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, pool=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class CNN1_Local(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, drop=0.2):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 32, k=3)
        self.b2 = ConvBlock(32, 64, k=3)
        self.b3 = ConvBlock(64, 128, k=3)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop), nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.head(self.b3(self.b2(self.b1(x))))

class CNN2_Global(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, drop=0.3):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 32, k=5)
        self.b2 = ConvBlock(32, 64, k=5)
        self.b3 = ConvBlock(64, 128, k=5)
        self.b4 = ConvBlock(128, 256, k=5)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.head(self.b4(self.b3(self.b2(self.b1(x)))))

class DWSeparableConv(nn.Module):
    # NOTE: underscore names to match your training checkpoints
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw    = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
                               groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw    = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

class CNN3_Compact(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            DWSeparableConv(32, 64),  nn.MaxPool2d(2),
            DWSeparableConv(64, 128), nn.MaxPool2d(2),
            DWSeparableConv(128, 256),nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(256, num_classes))
    def forward(self, x): return self.head(self.blocks(self.stem(x)))

# ---------------------------
# Data helpers
# ---------------------------
def make_eval_loader(data_dir, img_size, indices, grayscale=False, batch_size=32):
    tf = []
    if grayscale: tf.append(transforms.Grayscale(num_output_channels=1))
    tf = transforms.Compose(tf + [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*(1 if grayscale else 3), [0.5]*(1 if grayscale else 3)),
    ])
    base = datasets.ImageFolder(root=data_dir, transform=tf)
    ds = Subset(base, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False), base

def load_class_names(checkpoints_dir: Path, base: datasets.ImageFolder) -> List[str]:
    idx_map_path = checkpoints_dir / "idx_to_class.json"
    if idx_map_path.exists():
        mapping = json.loads(idx_map_path.read_text(encoding="utf-8"))
        class_names = [mapping[str(i)] if str(i) in mapping else mapping[i] for i in range(len(mapping))]
    else:
        idx_to_class = {v:k for k,v in base.class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return class_names

# ---- UMAR's requested short labels: IF, MSH, MSS, N ----
def to_short_labels(class_names: List[str]) -> List[str]:
    """
    Map verbose dataset names to UMAR's abbreviations:
      IF  -> Impeller (3.0BAR)
      MSH -> Mechanical seal Hole (3BAR)
      MSS -> Mechanical seal Scratch (3.0BAR)
      N   -> Normal (3BAR)
    Falls back gracefully if names differ.
    """
    out = []
    for name in class_names:
        low = name.lower()
        if "impeller" in low:
            out.append("IF")
        elif "hole" in low:
            out.append("MSH")
        elif "scratch" in low:
            out.append("MSS")
        elif "normal" in low:
            out.append("N")
        else:
            out.append(name)  # fallback to whatever it is
    return out

# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def predict_logits_and_probs(model, loader, device, tta=1):
    model.eval()
    all_logits=[]; all_probs=[]; all_labels=[]
    for x,y in loader:
        x = x.to(device)
        if tta <= 1:
            logits = model(x)
        else:
            acc = 0
            for i in range(tta):
                x_aug = x
                if i % 2 == 1: x_aug = torch.flip(x_aug, dims=[3])  # H flip
                if i % 4 == 2: x_aug = torch.flip(x_aug, dims=[2])  # V flip
                acc += model(x_aug)
            logits = acc / tta
        probs = F.softmax(logits, dim=1)
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    return np.vstack(all_logits), np.vstack(all_probs), np.concatenate(all_labels)

# ---------------------------
# Plots — STYLED LIKE YOUR SNIPPETS
# ---------------------------
def save_confusion_matrix(figpath, cm, labels):
    set_base_style()
    plt.figure(figsize=(8.8, 6.8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=False,
                     annot_kws={"size": 22, "fontweight": "bold"})
    ax.set_xlabel('Predicted Label', fontsize=18, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=18, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize=15, rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize=15)
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000)
    plt.close()
    print(f"Saved confusion matrix -> {figpath}")

def save_roc_curves(figpath, y_true_idx, probs, labels):
    set_base_style()
    K = len(labels)
    y_bin = label_binarize(y_true_idx, classes=list(range(K)))
    plt.figure(figsize=(10.5, 8.5))

    for i in range(K):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    #plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20)
    plt.legend(loc='lower right', fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000)
    plt.close()
    print(f"Saved ROC curves -> {figpath}")

def save_tsne(figpath, feats, y_true_idx, labels):
    set_base_style()
    # t-SNE params
    N = feats.shape[0]
    perplexity = max(5, min(30, (N - 1) // 3))
    xy = TSNE(n_components=2, init="pca", random_state=42,
              perplexity=perplexity, learning_rate="auto").fit_transform(feats)

    # Bigger, bolder TSNE: larger markers, black edges, higher alpha, bigger fonts
    markers = ['o', 's', '^', 'v']
    colors  = ['blue', 'red', 'green', 'orange']
    plt.figure(figsize=(12.5, 10.5))
    for i, cname in enumerate(labels):
        sel = (y_true_idx == i)
        plt.scatter(
            xy[sel, 0], xy[sel, 1],
            s=80, marker=markers[i % len(markers)],
            edgecolors='black', linewidths=0.8,
            color=colors[i % len(colors)],
            label=cname, alpha=0.85
        )

    plt.legend(title="Classes", loc='upper right',
               prop={'weight': 'bold', 'size': 13}, title_fontsize=14, frameon=True)
    #plt.title('t-SNE of Hybrid Model Features (2D)', fontsize=20, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=18, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    # no grid for t-SNE
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000)
    plt.close()
    print(f"Saved t-SNE -> {figpath}")

def save_training_curves(history_json_path: Path, outdir: str, model_key: str = "auto"):
    set_base_style()
    if not history_json_path.exists():
        print("No history.json found; skipping training curves.")
        return
    with open(history_json_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    if model_key == "auto":
        model_key = "cnn2" if "cnn2" in hist else (list(hist.keys())[0] if hist else None)
    if not model_key or model_key not in hist:
        print("history.json present but no valid model key; skipping curves.")
        return
    h = hist[model_key]

    plt.figure(figsize=(10.5, 5.5))
    plt.plot(h["epoch"], h["train_acc"], label='Training Accuracy')
    plt.plot(h["epoch"], h["val_acc"],   label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
    plt.title('Training and Validation Accuracy', fontsize=18, fontweight='bold')
    plt.legend(fontsize=13, prop={'weight':'bold'})
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13, fontweight='bold')
    plt.grid(True, which='both', linewidth=0.3)
    plt.minorticks_on()
    acc_path = os.path.join(outdir, "Comparison_model_training_val_accuracy.png")
    plt.tight_layout(); plt.savefig(acc_path, dpi=400); plt.close()
    print(f"Saved training accuracy -> {acc_path}")

    plt.figure(figsize=(10.5, 5.5))
    plt.plot(h["epoch"], h["train_loss"], label='Training Loss')
    plt.plot(h["epoch"], h["val_loss"],   label='Validation Loss')
    plt.xlabel('Epochs', fontsize=16, fontweight='bold')
    plt.ylabel('Loss', fontsize=16, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=18, fontweight='bold')
    plt.legend(fontsize=13, prop={'weight':'bold'})
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13, fontweight='bold')
    plt.grid(True, which='both', linewidth=0.3)
    plt.minorticks_on()
    loss_path = os.path.join(outdir, "Comparison_model_training_val_loss.png")
    plt.tight_layout(); plt.savefig(loss_path, dpi=400); plt.close()
    print(f"Saved training loss -> {loss_path}")

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="All-in-one figures (UMAR style).")
    ap.add_argument("--data_dir", required=True, help="Dataset root with class folders")
    ap.add_argument("--checkpoints", default="checkpoints", help="Folder with .pt files and split_indices.json")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--history_model", default="auto", help="Which model history to plot (cnn1/cnn2/cnn3/auto)")
    args = ap.parse_args()

    set_base_style()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoints)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 1 if args.grayscale else 3

    # Load split
    split_path = ckpt_dir / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Re-run training to save the split.")
    split = json.loads(split_path.read_text())
    test_idx = split["test"]

    # Data
    loader, base = make_eval_loader(args.data_dir, args.img_size, test_idx,
                                    grayscale=args.grayscale, batch_size=args.batch_size)
    class_names_verbose = load_class_names(ckpt_dir, base)   # original names from dataset
    class_names_display = to_short_labels(class_names_verbose)  # -> ['IF','MSH','MSS','N'] for your data

    true_idx  = np.array([base.samples[i][1] for i in test_idx], dtype=int)
    all_paths = [base.samples[i][0] for i in test_idx]

    # Models & weights
    m1 = CNN1_Local(num_classes=len(class_names_display), in_ch=in_ch)
    m2 = CNN2_Global(num_classes=len(class_names_display), in_ch=in_ch)
    m3 = CNN3_Compact(num_classes=len(class_names_display), in_ch=in_ch)

    def load_w(m, fname):
        p = ckpt_dir / fname
        if not p.exists(): raise FileNotFoundError(f"Missing checkpoint: {p}")
        state = torch.load(p, map_location="cpu")
        m.load_state_dict(state)
        m.to(device).eval()

    load_w(m1, "cnn1_local.pt")
    load_w(m2, "cnn2_global.pt")
    load_w(m3, "cnn3_compact.pt")

    # Inference
    logits1, probs1, y1 = predict_logits_and_probs(m1, loader, device, tta=args.tta)
    logits2, probs2, y2 = predict_logits_and_probs(m2, loader, device, tta=args.tta)
    logits3, probs3, y3 = predict_logits_and_probs(m3, loader, device, tta=args.tta)

    # Safety check
    if not (np.array_equal(y1, true_idx) and np.array_equal(y2, true_idx) and np.array_equal(y3, true_idx)):
        print("Warning: label order mismatch between models and loader; continuing.")

    # Soft-vote ensemble
    probs_ens = (probs1 + probs2 + probs3) / 3.0
    pred_idx  = probs_ens.argmax(axis=1)
    pred_prob = probs_ens.max(axis=1)

    # Hybrid logits for t-SNE
    feats_hybrid = np.concatenate([logits1, logits2, logits3], axis=1)

    # Metrics & text report (use short labels)
    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(class_names_display))))
    report_txt = classification_report(true_idx, pred_idx, target_names=class_names_display, digits=4, zero_division=0)
    overall_acc = float(np.trace(cm)) / float(np.sum(cm))

    with open(os.path.join(args.outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
        f.write("Classification Report (precision / recall / F1 per class):\n")
        f.write(report_txt)
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print(report_txt)

    # Save predictions CSV (paths + per-class probs) using short labels
    rows=[]
    for i, p in enumerate(all_paths):
        row = {"path": p,
               "true": class_names_display[true_idx[i]],
               "pred": class_names_display[pred_idx[i]],
               "pred_prob": float(pred_prob[i])}
        for j, cname in enumerate(class_names_display):
            row[f"p_{cname}"] = float(probs_ens[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "preds_test.csv"), index=False, encoding="utf-8")

    # Figures (short labels everywhere)
    save_confusion_matrix(os.path.join(args.outdir, "ensemble_confusion_matrix.png"), cm, class_names_display)
    save_roc_curves(os.path.join(args.outdir, "ensemble_ROC_curve.png"), true_idx, probs_ens, class_names_display)
    save_tsne(os.path.join(args.outdir, "tsne_2d_hybrid_model.png"), feats_hybrid, true_idx, class_names_display)
    save_training_curves(ckpt_dir / "history.json", args.outdir, model_key=args.history_model)

if __name__ == "__main__":
    main()
