# train_ensemble.py
# Author: UMAR+GPT | 3-CNN Ensemble with Soft/Meta Fusion, TTA, Mixup/CutMix, Label Smoothing
# Usage (Windows PowerShell example):
#   python train_ensemble.py --data_dir "E:\CP Dataset\WCA3" --epochs 40 --ensemble soft --img_size 256 --tta 8 --label_smoothing 0.05

import os, random, argparse, json, math, time
from pathlib import Path
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def class_counts_from_indices(dataset, indices):
    counts = {}
    for i in indices:
        _, y = dataset[i]
        counts[y] = counts.get(y, 0) + 1
    return counts

def compute_class_weights_from_counts(counts, num_classes):
    total = sum(counts.get(i, 0) for i in range(num_classes))
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        w = total / (num_classes * c)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

# ---------------------------
# Models
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, pool=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.pool(x)
        return x

class CNN1_Local(nn.Module):
    """3x3 kernels, 3 conv blocks"""
    def __init__(self, num_classes=4, in_ch=3, drop=0.2):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 32, k=3)
        self.b2 = ConvBlock(32, 64, k=3)
        self.b3 = ConvBlock(64, 128, k=3)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.head(self.b3(self.b2(self.b1(x))))

class CNN2_Global(nn.Module):
    """5x5 kernels, 4 conv blocks (slightly deeper)"""
    def __init__(self, num_classes=4, in_ch=3, drop=0.3):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 32, k=5)
        self.b2 = ConvBlock(32, 64, k=5)
        self.b3 = ConvBlock(64, 128, k=5)
        self.b4 = ConvBlock(128, 256, k=5)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.head(self.b4(self.b3(self.b2(self.b1(x)))))

class DWSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise 3x3 + pointwise 1x1"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

class CNN3_Compact(nn.Module):
    """Depthwise separable (MobileNet-style)"""
    def __init__(self, num_classes=4, in_ch=3, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            DWSeparableConv(32, 64, stride=1), nn.MaxPool2d(2),
            DWSeparableConv(64, 128, stride=1), nn.MaxPool2d(2),
            DWSeparableConv(128, 256, stride=1), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.head(self.blocks(self.stem(x)))

class MetaMLP(nn.Module):
    """Concatenate probs from 3 models (3*C) -> fused logits"""
    def __init__(self, num_models=3, num_classes=4, hidden=16, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

# ---------------------------
# Data / Augmentations
# ---------------------------
def build_transforms(img_size=224, grayscale=False, weak_aug=False):
    # Slightly stronger train augs; use --weak_aug for texture-like images where too much jitter hurts
    base = []
    if grayscale:
        base.append(transforms.Grayscale(num_output_channels=1))
    train_tf = transforms.Compose(base + [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(10 if not weak_aug else 5),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05) if not weak_aug else transforms.ColorJitter(0.05,0.05,0.05,0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*(1 if grayscale else 3), [0.5]*(1 if grayscale else 3)),
    ])
    test_tf = transforms.Compose(base + [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*(1 if grayscale else 3), [0.5]*(1 if grayscale else 3)),
    ])
    return train_tf, test_tf

def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    targets = [dataset[i][1] for i in range(len(dataset))]
    indices = np.arange(len(dataset))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, targets))
    y_temp = np.array(targets)[temp_idx]
    val_size = int(val_ratio * len(dataset))
    test_size = len(temp_idx) - val_size
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    val_rel, test_rel = next(sss2.split(temp_idx, y_temp))
    val_idx = temp_idx[val_rel]; test_idx = temp_idx[test_rel]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

def make_loaders(data_dir, img_size=224, batch_size=32, num_workers=0, seed=42, grayscale=False, weak_aug=False):
    train_tf, test_tf = build_transforms(img_size, grayscale=grayscale, weak_aug=weak_aug)
    # We instantiate two ImageFolders so Subset can carry different transforms
    base_eval = datasets.ImageFolder(root=data_dir, transform=test_tf)
    train_idx, val_idx, test_idx = stratified_split(base_eval, seed=seed)

    base_train = datasets.ImageFolder(root=data_dir, transform=train_tf)
    base_eval  = datasets.ImageFolder(root=data_dir, transform=test_tf)

    train_ds = Subset(base_train, train_idx)
    val_ds   = Subset(base_eval,  val_idx)
    test_ds  = Subset(base_eval,  test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    class_to_idx = base_eval.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    return train_loader, val_loader, test_loader, idx_to_class, (train_idx, val_idx, test_idx), base_eval

# ---------------------------
# Mixup / CutMix
# ---------------------------
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup_cutmix(x, y, num_classes, mixup_alpha=0.0, cutmix_alpha=0.0, device=None):
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return x, y, None  # no mixing
    lam = 1.0
    if cutmix_alpha > 0 and np.random.rand() < 0.5:
        # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x2 = x.flip(0)
        W = x.size(3); H = x.size(2)
        x1_, y1_, x2_, y2_ = rand_bbox(W, H, lam)
        x[:, :, y1_:y2_, x1_:x2_] = x2[:, :, y1_:y2_, x1_:x2_]
        lam = 1 - ((x2_-x1_) * (y2_-y1_) / (W * H))
        y_a, y_b = y, y.flip(0)
    else:
        # Mixup
        lam = np.random.beta(max(1e-8, mixup_alpha), max(1e-8, mixup_alpha))
        x2 = x.flip(0)
        x = lam * x + (1 - lam) * x2
        y_a, y_b = y, y.flip(0)
    # return mixed targets as tuple
    return x, (y_a, y_b, lam), "mixed"

def loss_with_mixing(criterion, logits, y_or_tuple):
    if isinstance(y_or_tuple, tuple):
        y_a, y_b, lam = y_or_tuple
        return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    return criterion(logits, y_or_tuple)

# ---------------------------
# Train / Eval
# ---------------------------
def train_one(model, train_loader, val_loader, epochs, lr, device, ckpt_path,
              class_weights=None, label_smoothing=0.0,
              mixup_alpha=0.0, cutmix_alpha=0.0):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None,
                                    label_smoothing=label_smoothing)
    best_val = float('inf'); best_state = None; patience=10; no_imp=0

    for ep in range(1, epochs+1):
        model.train()
        tr_loss=0; tr_correct=0; n=0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            x, y_mix, mixed_flag = apply_mixup_cutmix(
                x, y, num_classes=None, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, device=device
            )
            opt.zero_grad()
            logits = model(x)
            loss = loss_with_mixing(criterion, logits, y_mix if mixed_flag else y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
            # For accuracy during mixup/cutmix, use hard preds vs original y (approx)
            tr_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        sched.step()

        # Val
        model.eval()
        vl_loss=0; vl_correct=0; m=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                vl_loss += loss.item()*x.size(0)
                vl_correct += (logits.argmax(1)==y).sum().item()
                m += x.size(0)

        tr_loss/=max(1,n); tr_acc=tr_correct/max(1,n)
        vl_loss/=max(1,m); vl_acc=vl_correct/max(1,m)
        print(f"Epoch {ep:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {vl_loss:.4f} acc {vl_acc:.3f}")

        if vl_loss < best_val:
            best_val = vl_loss; best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict_probs(model, loader, device, tta=1, test_aug=None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs=[]; all_labels=[]
    for x,y in loader:
        x = x.to(device)
        if tta <= 1:
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        else:
            # simple TTA: flips + optional small rotations via test_aug
            probs_accum = 0
            for i in range(tta):
                x_aug = x
                if i % 2 == 1:
                    x_aug = torch.flip(x_aug, dims=[3])  # horizontal
                if i % 4 == 2:
                    x_aug = torch.flip(x_aug, dims=[2])  # vertical
                if test_aug is not None and i % 4 == 3:
                    x_aug = test_aug(x_aug)
                logits = model(x_aug)
                probs_accum += F.softmax(logits, dim=1)
            probs = (probs_accum / tta).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.vstack(all_probs), np.concatenate(all_labels)

def evaluate_from_probs(fused_probs: np.ndarray, labels: np.ndarray, idx_to_class: dict, title: str):
    preds = fused_probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    print(f"\n== {title} Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    print("\nClassification Report:")
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(classification_report(labels, preds, target_names=target_names, digits=4, zero_division=0))
    return acc

def soft_vote(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probs_list, axis=0), axis=0)

def train_meta_mlp(val_probs_list, val_labels, num_classes=4, epochs=100, lr=1e-3, hidden=16, device=None):
    X = np.concatenate(val_probs_list, axis=1)  # [N, 3*C]
    y = val_labels
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    model = MetaMLP(num_models=len(val_probs_list), num_classes=num_classes, hidden=hidden, drop=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_loss=float('inf'); best_state=None; patience=15; no_imp=0
    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        opt.step()
        with torch.no_grad():
            vl_loss = loss.item()
        if vl_loss < best_loss:
            best_loss=vl_loss; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            no_imp=0
        else:
            no_imp+=1
            if no_imp>=patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()  # IMPORTANT: ensure dropout is OFF in inference
    return model

# tiny rotation for TTA when tensor is already normalized (approx via grid_sample)
def small_rotate_tensor(x, degrees=5):
    # no-op if torch < 2 grid_sample alignment issues, but this is fine for small angles
    theta = torch.zeros(x.size(0), 2, 3, device=x.device, dtype=x.dtype)
    angle = (degrees * math.pi / 180.0) * (torch.rand(x.size(0), device=x.device) * 2 - 1)  # [-deg, +deg]
    cos = torch.cos(angle); sin = torch.sin(angle)
    theta[:,0,0] = cos; theta[:,0,1] = -sin
    theta[:,1,0] = sin; theta[:,1,1] =  cos
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Path containing 4 class folders")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ensemble", type=str, default="soft", choices=["soft","meta"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--grayscale", action="store_true", help="Use 1-channel pipeline")
    ap.add_argument("--weak_aug", action="store_true", help="Gentler augmentation")
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--mixup_alpha", type=float, default=0.0)
    ap.add_argument("--cutmix_alpha", type=float, default=0.0)
    ap.add_argument("--tta", type=int, default=1, help="TTA passes during eval (1 = off)")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = device_auto()
    print("Device:", dev)

    train_loader, val_loader, test_loader, idx_to_class, (train_idx,val_idx,test_idx), base_eval = make_loaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, grayscale=args.grayscale, weak_aug=args.weak_aug
    )
    num_classes = len(idx_to_class)
    in_ch = 1 if args.grayscale else 3

    # Class weights (from training indices)
    class_w = None
    if args.use_class_weights:
        counts = class_counts_from_indices(base_eval, train_idx)
        class_w = compute_class_weights_from_counts(counts, num_classes)
        print("Class counts (train):", counts, " | weights:", class_w.tolist())

    # Build models
    m1 = CNN1_Local(num_classes=num_classes, in_ch=in_ch)
    m2 = CNN2_Global(num_classes=num_classes, in_ch=in_ch)
    m3 = CNN3_Compact(num_classes=num_classes, in_ch=in_ch)
    print("Params (CNN1, CNN2, CNN3):", count_params(m1), count_params(m2), count_params(m3))

    os.makedirs("checkpoints", exist_ok=True)

    # Train each model
    m1 = train_one(m1, train_loader, val_loader, args.epochs, args.lr, dev, "checkpoints/cnn1_local.pt",
                   class_weights=class_w, label_smoothing=args.label_smoothing,
                   mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha)
    m2 = train_one(m2, train_loader, val_loader, args.epochs, args.lr, dev, "checkpoints/cnn2_global.pt",
                   class_weights=class_w, label_smoothing=args.label_smoothing,
                   mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha)
    m3 = train_one(m3, train_loader, val_loader, args.epochs, args.lr, dev, "checkpoints/cnn3_compact.pt",
                   class_weights=class_w, label_smoothing=args.label_smoothing,
                   mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha)

    # Collect probabilities (+TTA)
    test_aug = lambda t: small_rotate_tensor(t, degrees=5)
    v1, yv = predict_probs(m1.to(dev), val_loader, dev, tta=args.tta, test_aug=test_aug)
    v2, _  = predict_probs(m2.to(dev), val_loader, dev, tta=args.tta, test_aug=test_aug)
    v3, _  = predict_probs(m3.to(dev), val_loader, dev, tta=args.tta, test_aug=test_aug)

    t1, yt = predict_probs(m1.to(dev), test_loader, dev, tta=args.tta, test_aug=test_aug)
    t2, _  = predict_probs(m2.to(dev), test_loader, dev, tta=args.tta, test_aug=test_aug)
    t3, _  = predict_probs(m3.to(dev), test_loader, dev, tta=args.tta, test_aug=test_aug)

    # Per-model eval
    print("\n==== Individual Models ====")
    evaluate_from_probs(v1, yv, idx_to_class, "Validation CNN1")
    evaluate_from_probs(v2, yv, idx_to_class, "Validation CNN2")
    evaluate_from_probs(v3, yv, idx_to_class, "Validation CNN3")
    evaluate_from_probs(t1, yt, idx_to_class, "Test CNN1")
    evaluate_from_probs(t2, yt, idx_to_class, "Test CNN2")
    evaluate_from_probs(t3, yt, idx_to_class, "Test CNN3")

    # Ensembles
    print("\n==== Ensemble ====")
    if args.ensemble == "soft":
        val_fused = soft_vote([v1,v2,v3])
        test_fused = soft_vote([t1,t2,t3])

        evaluate_from_probs(val_fused, yv, idx_to_class, "Validation (Soft Voting)")
        evaluate_from_probs(test_fused, yt, idx_to_class, "Test (Soft Voting)")

    else:
        meta = train_meta_mlp([v1,v2,v3], yv, num_classes=num_classes, device=dev)
        # Validation (sanity)
        X_val = np.concatenate([v1,v2,v3], axis=1)
        with torch.no_grad():
            val_logits = meta(torch.tensor(X_val, dtype=torch.float32, device=dev)).cpu().numpy()
        val_probs  = F.softmax(torch.tensor(val_logits), dim=1).numpy()
        evaluate_from_probs(val_probs, yv, idx_to_class, "Validation (Meta-Classifier)")

        # Test
        X_test = np.concatenate([t1,t2,t3], axis=1)
        with torch.no_grad():
            test_logits = meta(torch.tensor(X_test, dtype=torch.float32, device=dev)).cpu().numpy()
        test_probs  = F.softmax(torch.tensor(test_logits), dim=1).numpy()
        evaluate_from_probs(test_probs, yt, idx_to_class, "Test (Meta-Classifier)")

    # Save label map + split for reproducibility
    with open("checkpoints/idx_to_class.json","w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)
    with open("checkpoints/split_indices.json","w") as f:
        json.dump({"train":train_idx, "val":val_idx, "test":test_idx}, f, indent=2)

if __name__ == "__main__":
    main()
