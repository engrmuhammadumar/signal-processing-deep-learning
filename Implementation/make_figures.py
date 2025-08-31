# make_figures.py
# Build confusion matrix, ROC curves, t-SNE (from probs), and training curves (from history.json if provided).

import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from config import CLASSES

sns.set_context("talk")
sns.set_style("whitegrid")

# ---------- utils ----------
def ensure_outdir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def normalize_class_name(name: str): return name.strip().lower()

def build_probs_matrix(df: pd.DataFrame, classes):
    # Predicted class
    if "pred" in df.columns:
        y_pred = df["pred"].astype(str).tolist()
    elif "pred_class" in df.columns:
        y_pred = df["pred_class"].astype(str).tolist()
    else:
        raise ValueError("CSV must include 'pred' or 'pred_class'.")

    # Probability columns p_<ClassName>
    K = len(classes)
    norm_classes = [normalize_class_name(c) for c in classes]
    probs = np.zeros((len(df), K), dtype=np.float64)
    for col in df.columns:
        if not col.startswith("p_"): continue
        raw = col[2:]
        rn = normalize_class_name(raw)
        if rn in norm_classes:
            j = norm_classes.index(rn)
            probs[:, j] = df[col].astype(float).values
    # If missing probs, fallback to one-hot on pred
    if probs.sum() == 0:
        for i, pc in enumerate(y_pred):
            if pc in classes: probs[i, classes.index(pc)] = 1.0
    # Normalize rows
    row_sums = probs.sum(axis=1, keepdims=True); row_sums[row_sums==0] = 1.0
    probs = probs / row_sums
    return probs, y_pred

# ---------- plots ----------
def save_confusion_matrix(figpath, cm, labels):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=False,
                     annot_kws={"size": 20, "fontweight": "bold"})
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize=12, rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize=12)
    plt.tight_layout(); plt.savefig(figpath, dpi=800); plt.close()

def save_roc_curves(figpath, y_true_names, probs, labels):
    # map names -> indices
    name_to_idx = {c:i for i,c in enumerate(labels)}
    y_idx = np.array([name_to_idx[n] for n in y_true_names], dtype=int)
    y_bin = label_binarize(y_idx, classes=list(range(len(labels))))
    plt.figure(figsize=(10, 8))
    for i, c in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{c} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    plt.title('ROC (One-vs-Rest)', fontsize=18, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(figpath, dpi=800); plt.close()

def save_tsne(figpath, feats, y_true_names, labels):
    # feats = probabilities or embeddings [N, K or D]
    N = feats.shape[0]
    perplexity = max(5, min(30, (N - 1) // 3))
    tsne = TSNE(n_components=2, init="pca", random_state=42,
                perplexity=perplexity, learning_rate="auto")
    xy = tsne.fit_transform(feats)

    label_to_idx = {c: i for i, c in enumerate(labels)}
    idxs = np.array([label_to_idx[s] for s in y_true_names])

    plt.figure(figsize=(12, 10))
    markers = ['o', 's', '^', 'v', 'D', '*', 'X']
    palette = sns.color_palette(n_colors=len(labels))
    for i, c in enumerate(labels):
        sel = (idxs == i)
        plt.scatter(xy[sel, 0], xy[sel, 1], marker=markers[i % len(markers)],
                    color=palette[i], alpha=0.75, s=24, label=c, edgecolor='none')
    plt.legend(title="Classes", loc='best', fontsize=10, title_fontsize=11)
    plt.title('t-SNE of Ensemble Probabilities', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(figpath, dpi=800); plt.close()

def save_training_curves(history_json_path, outdir):
    """
    Expects a JSON like:
    {
      "cnn1": {"epoch":[...], "train_loss":[...], "val_loss":[...], "train_acc":[...], "val_acc":[...]},
      "cnn2": {...},
      "cnn3": {...}
    }
    """
    if not history_json_path or not Path(history_json_path).exists():
        print("No history.json provided; skipping training curves.")
        return
    hist = pd.read_json(history_json_path)

    # If stored as dict-of-dicts, normalize
    if isinstance(hist.iloc[0], (dict, list)):
        # read again in raw then restructure
        import json
        with open(history_json_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

    # Plot loss
    plt.figure(figsize=(10,7))
    for k in ["cnn1","cnn2","cnn3"]:
        if k in hist:
            h = hist[k]
            plt.plot(h["epoch"], h["train_loss"], linewidth=2, alpha=0.8, label=f"{k} train")
            plt.plot(h["epoch"], h["val_loss"],   linewidth=2, alpha=0.8, linestyle="--", label=f"{k} val")
    plt.xlabel("Epoch", fontsize=14, fontweight='bold'); plt.ylabel("Loss", fontsize=14, fontweight='bold')
    plt.title("Training/Validation Loss", fontsize=16, fontweight='bold'); plt.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "training_loss.png"), dpi=800); plt.close()

    # Plot accuracy
    plt.figure(figsize=(10,7))
    for k in ["cnn1","cnn2","cnn3"]:
        if k in hist:
            h = hist[k]
            plt.plot(h["epoch"], h["train_acc"], linewidth=2, alpha=0.8, label=f"{k} train")
            plt.plot(h["epoch"], h["val_acc"],   linewidth=2, alpha=0.8, linestyle="--", label=f"{k} val")
    plt.xlabel("Epoch", fontsize=14, fontweight='bold'); plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.title("Training/Validation Accuracy", fontsize=16, fontweight='bold'); plt.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "training_acc.png"), dpi=800); plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Make paper-style figures from predictions CSV (and optional history.json).")
    ap.add_argument("--csv", required=True, help="Path to predictions CSV (from export_predictions.py)")
    ap.add_argument("--outdir", default="figures", help="Directory to save figures/metrics")
    ap.add_argument("--history", default=None, help="Optional path to checkpoints/history.json for training curves")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    # true/pred/probs
    if "true" not in df.columns: raise ValueError("CSV must have a 'true' column.")
    y_true = df["true"].astype(str).tolist()
    probs, y_pred = build_probs_matrix(df, CLASSES)

    # metrics text
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)
    overall_acc = float(np.trace(cm)) / max(1, float(np.sum(cm)))
    with open(os.path.join(args.outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
        f.write("Classification Report (precision/recall/F1 per class):\n")
        f.write(report)

    # figures
    save_confusion_matrix(os.path.join(args.outdir, "confusion_matrix.png"), cm, CLASSES)
    save_roc_curves(os.path.join(args.outdir, "roc_curves.png"), y_true, probs, CLASSES)
    save_tsne(os.path.join(args.outdir, "tsne.png"), probs, y_true, CLASSES)
    save_training_curves(args.history, args.outdir)

    print("\n=== Summary ===")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print("Saved:", os.path.join(args.outdir, "metrics.txt"))
    print("Saved:", os.path.join(args.outdir, "confusion_matrix.png"))
    print("Saved:", os.path.join(args.outdir, "roc_curves.png"))
    print("Saved:", os.path.join(args.outdir, "tsne.png"))
    if args.history: print("Saved:", os.path.join(args.outdir, "training_loss.png"), "and training_acc.png")

if __name__ == "__main__":
    main()
