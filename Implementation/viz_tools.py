# viz_tools.py
# Paper-style plots: ROC, t-SNE, and training curves
# Usage example is at the bottom of this message.

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

# ---------------------------
# Global style (bold, paper-like)
# ---------------------------
def set_paper_style():
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": 100,         # screen view; we'll export at high dpi per figure
        "savefig.dpi": 1000,       # exported files are high-res
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.weight": "bold",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

# ---------------------------
# Evaluation helpers
# ---------------------------
@torch.no_grad()
def evaluate_logits(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass to collect true labels (ints), predicted labels (ints), and probabilities [N, K].
    Works with DataParallel models too.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)              # [B, K]
        probs = F.softmax(logits, dim=1)    # [B, K]
        preds = logits.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    probs  = np.concatenate(all_probs)      # [N, K]
    return y_true, y_pred, probs

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> None:
    rep = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm) / cm.sum()
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n", rep)

# ---------------------------
# ROC (One-vs-Rest), per class + optional micro/macro
# ---------------------------
def plot_roc_ovr(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    outpath: str = "roc_curves.png",
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    include_micro_macro: bool = True,
):
    set_paper_style()
    K = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(K)))  # [N, K]

    plt.figure(figsize=(10, 8))

    # Per-class ROC
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC = {roc_auc:.3f})")

    if include_micro_macro:
        # micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, lw=2, linestyle="--", label=f"micro-average (AUC = {auc_micro:.3f})")

        # macro-average
        aucs = []
        mean_fpr = np.linspace(0, 1, 1000)
        for i in range(K):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            aucs.append(auc(fpr, tpr))
        auc_macro = np.mean(aucs)
        # Plot a reference mean line (optional)—comment out if you don’t want it
        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=1000)
    print(f"Saved ROC -> {outpath}")
    plt.close()

# ---------------------------
# t-SNE (using logits or probabilities)
# ---------------------------
@torch.no_grad()
def extract_features_logits(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (features, y_true). Here features are the final logits.
    If you want penultimate features, replace this with a forward hook.
    """
    model.eval()
    feats, labels_all = [], []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)  # [B, K]
        feats.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labels_all)

def plot_tsne(
    feats: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    outpath: str = "tsne.png",
    title: str = "t-SNE of Model Features (2D)",
    use_markers: bool = True,
):
    set_paper_style()
    N = feats.shape[0]
    perplexity = max(5, min(30, (N - 1) // 3))

    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=42,
        perplexity=perplexity,
        learning_rate="auto"
    )
    xy = tsne.fit_transform(feats)  # [N, 2]
    idxs = y_true.astype(int)

    plt.figure(figsize=(12, 10))
    palette = sns.color_palette(n_colors=len(class_names))
    markers = ['o', 's', '^', 'v', 'D', '*', 'X']

    for i, cname in enumerate(class_names):
        sel = (idxs == i)
        if use_markers:
            plt.scatter(
                xy[sel, 0], xy[sel, 1],
                marker=markers[i % len(markers)],
                color=palette[i],
                alpha=0.75, s=30, label=cname, edgecolors="none"
            )
        else:
            plt.scatter(
                xy[sel, 0], xy[sel, 1],
                color=palette[i], alpha=0.75, s=30, label=cname, edgecolors="none"
            )

    plt.legend(title="Classes", loc="best", fontsize=10, title_fontsize=11)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.minorticks_on()
    plt.grid(True, which="both", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=1000)
    print(f"Saved t-SNE -> {outpath}")
    plt.close()

# ---------------------------
# Training curves (accuracy & loss)
# ---------------------------
def _smooth(x: List[float], k: int = 0) -> np.ndarray:
    if k is None or k <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    k = int(k)
    if k > len(x): k = len(x)
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")

def plot_training_curves(
    train_acc: List[float],
    val_acc: List[float],
    train_loss: List[float],
    val_loss: List[float],
    outdir: str = ".",
    title_prefix: str = "Training and Validation",
    smooth_k: int = 0,
):
    set_paper_style()
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(10, 5))
    ta = _smooth(train_acc, smooth_k)
    va = _smooth(val_acc, smooth_k)
    plt.plot(ta, label="Training Accuracy", linewidth=2)
    plt.plot(va, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend(prop={"weight": "bold"})
    plt.minorticks_on(); plt.grid(True, which="both", linewidth=0.3)
    acc_path = os.path.join(outdir, "training_val_accuracy.png")
    plt.tight_layout(); plt.savefig(acc_path, dpi=400); plt.close()
    print(f"Saved training accuracy -> {acc_path}")

    # Loss
    plt.figure(figsize=(10, 5))
    tl = _smooth(train_loss, smooth_k)
    vl = _smooth(val_loss, smooth_k)
    plt.plot(tl, label="Training Loss", linewidth=2)
    plt.plot(vl, label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend(prop={"weight": "bold"})
    plt.minorticks_on(); plt.grid(True, which="both", linewidth=0.3)
    loss_path = os.path.join(outdir, "training_val_loss.png")
    plt.tight_layout(); plt.savefig(loss_path, dpi=400); plt.close()
    print(f"Saved training loss -> {loss_path}")
