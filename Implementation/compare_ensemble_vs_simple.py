# compare_ensemble_vs_simple.py
# Compare your 3-CNN soft-voting ensemble vs a Simple CNN baseline on the exact same fixed split.
# Figures saved with UMAR's style: CM/TSNE no grid; ROC has a light grid. Labels shown as IF/MSH/MSS/N.

import os, json, argparse, math, random
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Global Plot Style (UMAR style)
# ---------------------------
def set_base_style():
    sns.set_style("white")
    sns.set_context("talk")
    plt.rcParams.update({
        "savefig.dpi": 1000,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.weight": "bold",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.grid": False
    })

# ---------------------------
# Models (match your training definitions exactly)
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
    # 3x3 kernels, 3 blocks
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
    # 5x5 kernels, 4 blocks
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
    # Depthwise 3x3 + pointwise 1x1 (names match your training: dw_bn, pw_bn)
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw    = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw    = nn.Conv2d(in_ch, out_ch, 1, bias=False)
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

# --- Simple CNN Baseline (compact, fair) ---
class SimpleCNN(nn.Module):
    # 3 x ConvBlock(3x3), fewer channels; GAP -> FC(4)
    def __init__(self, num_classes=4, in_ch=3, drop=0.2):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 32, k=3)
        self.b2 = ConvBlock(32, 64, k=3)
        self.b3 = ConvBlock(64, 128, k=3)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.head(self.b3(self.b2(self.b1(x))))

# ---------------------------
# Data loaders (use saved split)
# ---------------------------
def build_transforms(img_size=256, grayscale=False, weak_aug=True):
    base = []
    if grayscale:
        base.append(transforms.Grayscale(num_output_channels=1))
    train_tf = transforms.Compose(base + [
        transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(5 if weak_aug else 10),
        transforms.ColorJitter(0.05,0.05,0.05,0.02) if weak_aug else transforms.ColorJitter(0.15,0.15,0.15,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*(1 if grayscale else 3), [0.5]*(1 if grayscale else 3)),
    ])
    eval_tf = transforms.Compose(base + [
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*(1 if grayscale else 3), [0.5]*(1 if grayscale else 3)),
    ])
    return train_tf, eval_tf

def make_loaders_from_split(data_dir, split_path, img_size=256, batch_size=32, grayscale=False, num_workers=0):
    train_tf, eval_tf = build_transforms(img_size, grayscale=grayscale, weak_aug=True)
    base_train = datasets.ImageFolder(root=data_dir, transform=train_tf)
    base_eval  = datasets.ImageFolder(root=data_dir, transform=eval_tf)

    split = json.loads(Path(split_path).read_text())
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]

    ds_train = Subset(base_train, train_idx)
    ds_val   = Subset(base_eval,  val_idx)
    ds_test  = Subset(base_eval,  test_idx)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    idx_to_class = {v:k for k,v in base_eval.class_to_idx.items()}
    return train_loader, val_loader, test_loader, idx_to_class, base_eval, (train_idx,val_idx,test_idx)

# ---------------------------
# Train baseline
# ---------------------------
def class_weights_from_counts(base_eval, train_idx, num_classes):
    counts = {}
    for i in train_idx:
        _, y = base_eval[i]
        counts[y] = counts.get(y, 0) + 1
    total = sum(counts.get(i, 0) for i in range(num_classes))
    w = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        w.append(total / (num_classes * c))
    return torch.tensor(w, dtype=torch.float32), counts

def train_simple(model, train_loader, val_loader, epochs, lr, device, class_weights=None, label_smoothing=0.05):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,epochs))
    criterion = nn.CrossEntropyLoss(
        weight=(class_weights.to(device) if class_weights is not None else None),
        label_smoothing=label_smoothing
    )
    best_val = float('inf'); best_state=None; patience=10; no_imp=0
    for ep in range(1, epochs+1):
        model.train(); tr_loss=0; tr_cor=0; n=0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item()*x.size(0)
            tr_cor  += (logits.argmax(1)==y).sum().item()
            n += x.size(0)
        sched.step()
        tr_loss/=max(1,n); tr_acc = tr_cor/max(1,n)

        model.eval(); vl_loss=0; vl_cor=0; m=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                vl_loss+=loss.item()*x.size(0)
                vl_cor += (logits.argmax(1)==y).sum().item()
                m += x.size(0)
        vl_loss/=max(1,m); vl_acc = vl_cor/max(1,m)
        print(f"[SimpleCNN] Epoch {ep:03d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {vl_loss:.4f}/{vl_acc:.3f}")
        if vl_loss < best_val:
            best_val = vl_loss; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp += 1
            if no_imp>=patience: print("Early stopping."); break
    if best_state is not None: model.load_state_dict(best_state)
    return model

# ---------------------------
# Inference utilities
# ---------------------------
@torch.no_grad()
def predict_logits_probs(model, loader, device, tta=1):
    model.eval()
    all_logits=[]; all_probs=[]; all_y=[]
    for x,y in loader:
        x = x.to(device)
        if tta<=1:
            logits = model(x)
        else:
            acc = 0
            for i in range(tta):
                x_aug = x
                if i%2==1: x_aug = torch.flip(x_aug, dims=[3]) # H flip
                if i%4==2: x_aug = torch.flip(x_aug, dims=[2]) # V flip
                acc += model(x_aug)
            logits = acc/tta
        probs = F.softmax(logits, dim=1)
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_y.append(y.numpy())
    return np.vstack(all_logits), np.vstack(all_probs), np.concatenate(all_y)

def soft_vote(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probs_list, axis=0), axis=0)

# ---------------------------
# Plots (UMAR style)
# ---------------------------
def save_confusion_matrix(figpath, cm, labels):
    set_base_style()
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=False,
                     annot_kws={"size":20,"fontweight":"bold"})
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize=14, rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize=14)
    plt.tight_layout(); plt.savefig(figpath, dpi=1000); plt.close()
    print("Saved:", figpath)

def save_roc(figpath, y_true_idx, probs, labels):
    set_base_style()
    K = len(labels)
    y_bin = label_binarize(y_true_idx, classes=list(range(K)))
    plt.figure(figsize=(10,8))
    for i in range(K):
        fpr, tpr, _ = roc_curve(y_bin[:,i], probs[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(figpath, dpi=1000); plt.close()
    print("Saved:", figpath)

def save_tsne(figpath, feats, y_true_idx, labels, title):
    set_base_style()
    N = feats.shape[0]
    perplexity = max(5, min(30, (N - 1)//3))
    xy = TSNE(n_components=2, init="pca", random_state=42,
              perplexity=perplexity, learning_rate="auto").fit_transform(feats)
    markers = ['o','s','^','v']
    colors  = ['blue','red','green','purple']
    plt.figure(figsize=(12,10))
    for i, name in enumerate(labels):
        sel = (y_true_idx==i)
        plt.scatter(xy[sel,0], xy[sel,1],
                    marker=markers[i%len(markers)],
                    color=colors[i%len(colors)],
                    alpha=0.7, label=name, s=30)
    plt.legend(title="Classes", loc='upper right',
               prop={'weight':'bold','size':12}, title_fontsize=13)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold'); plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(figpath, dpi=1000); plt.close()
    print("Saved:", figpath)

# ---------------------------
# Label display (short names IF/MSH/MSS/N)
# ---------------------------
def short_labels_from_idx_map(idx_to_class: dict) -> List[str]:
    # Map full names to IF/MSH/MSS/N by simple heuristics
    order = [idx_to_class[i] for i in range(len(idx_to_class))]
    out=[]
    for name in order:
        nlow = name.lower()
        if "impeller" in nlow: out.append("IF")
        elif "hole" in nlow:   out.append("MSH")
        elif "scratch" in nlow:out.append("MSS")
        elif "normal" in nlow: out.append("N")
        else:
            out.append(name)  # fallback
    return out

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare Ensemble (soft vote) vs SimpleCNN baseline on fixed split.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoints", default="checkpoints", help="Folder with ensemble .pt and split_indices.json")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs_baseline", type=int, default=30)
    ap.add_argument("--lr_baseline", type=float, default=1e-3)
    ap.add_argument("--tta", type=int, default=8, help="TTA passes for evaluation (both models)")
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--outdir", default="compare_figures")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    set_base_style()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoints)
    split_path = ckpt_dir / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}. Train the ensemble first to save it.")

    device = device_auto()
    in_ch = 1 if args.grayscale else 3

    # Load data on saved split
    train_loader, val_loader, test_loader, idx_to_class, base_eval, (train_idx,val_idx,test_idx) = \
        make_loaders_from_split(args.data_dir, split_path, img_size=args.img_size,
                                batch_size=args.batch_size, grayscale=args.grayscale, num_workers=args.num_workers)
    class_names_short = short_labels_from_idx_map(idx_to_class)
    num_classes = len(idx_to_class)

    # ---------- Ensemble (load weights) ----------
    m1 = CNN1_Local(num_classes=num_classes, in_ch=in_ch)
    m2 = CNN2_Global(num_classes=num_classes, in_ch=in_ch)
    m3 = CNN3_Compact(num_classes=num_classes, in_ch=in_ch)

    def load_w(m, fname):
        p = ckpt_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        state = torch.load(p, map_location="cpu")
        m.load_state_dict(state)
        m.to(device).eval()

    load_w(m1, "cnn1_local.pt")
    load_w(m2, "cnn2_global.pt")
    load_w(m3, "cnn3_compact.pt")

    # Evaluate ensemble
    log1, prob1, y1 = predict_logits_probs(m1, test_loader, device, tta=args.tta)
    log2, prob2, y2 = predict_logits_probs(m2, test_loader, device, tta=args.tta)
    log3, prob3, y3 = predict_logits_probs(m3, test_loader, device, tta=args.tta)

    true_idx = y1  # all equal by construction
    probs_ens = soft_vote([prob1,prob2,prob3])
    logits_hybrid = np.concatenate([log1,log2,log3], axis=1)  # for t-SNE

    pred_ens = probs_ens.argmax(1)
    acc_ens  = accuracy_score(true_idx, pred_ens)

    cm_ens = confusion_matrix(true_idx, pred_ens, labels=list(range(num_classes)))
    report_ens = classification_report(true_idx, pred_ens, target_names=class_names_short, digits=4, zero_division=0)

    # Save ensemble artifacts
    save_confusion_matrix(os.path.join(args.outdir, "ensemble_confusion_matrix.png"), cm_ens, class_names_short)
    save_roc(os.path.join(args.outdir, "ensemble_ROC_curve.png"), true_idx, probs_ens, class_names_short)
    save_tsne(os.path.join(args.outdir, "ensemble_tsne.png"), logits_hybrid, true_idx, class_names_short,
              title="t-SNE of Ensemble Hybrid Logits (2D)")
    with open(os.path.join(args.outdir, "ensemble_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc_ens:.4f}\n\n")
        f.write(report_ens)

    # ---------- Simple CNN Baseline (train on same split) ----------
    simple = SimpleCNN(num_classes=num_classes, in_ch=in_ch)
    class_w, train_counts = class_weights_from_counts(base_eval, train_idx, num_classes)
    simple = train_simple(simple, train_loader, val_loader, epochs=args.epochs_baseline,
                          lr=args.lr_baseline, device=device, class_weights=class_w, label_smoothing=0.05)

    # Evaluate baseline (optionally with same TTA)
    log_b, prob_b, yb = predict_logits_probs(simple, test_loader, device, tta=args.tta)
    pred_b = prob_b.argmax(1)
    acc_b  = accuracy_score(yb, pred_b)

    cm_b = confusion_matrix(yb, pred_b, labels=list(range(num_classes)))
    report_b = classification_report(yb, pred_b, target_names=class_names_short, digits=4, zero_division=0)

    # Save baseline artifacts
    save_confusion_matrix(os.path.join(args.outdir, "simple_confusion_matrix.png"), cm_b, class_names_short)
    save_roc(os.path.join(args.outdir, "simple_ROC_curve.png"), yb, prob_b, class_names_short)
    save_tsne(os.path.join(args.outdir, "simple_tsne.png"), log_b, yb, class_names_short,
              title="t-SNE of SimpleCNN Logits (2D)")
    with open(os.path.join(args.outdir, "simple_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc_b:.4f}\n\n")
        f.write(report_b)

    # ---------- Compact comparison CSV ----------
    pd.DataFrame([
        {"model":"Ensemble(3-CNN+TTA+SoftVote)", "accuracy":acc_ens},
        {"model":"SimpleCNN baseline",            "accuracy":acc_b},
    ]).to_csv(os.path.join(args.outdir, "comparison_summary.csv"), index=False)

    print("\n=== Summary ===")
    print(f"Ensemble Acc: {acc_ens:.4f}")
    print(f"SimpleCNN Acc: {acc_b:.4f}")
    print("Saved figures & reports to:", args.outdir)

if __name__ == "__main__":
    main()
