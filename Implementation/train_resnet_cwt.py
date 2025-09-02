# -*- coding: utf-8 -*-
r"""
CWT Image Classification — Efficient Transfer Learning (ResNet18/MobileNet/EfficientNet)
- Pretrained backbone (default: resnet18) + label smoothing
- Optional MixUp / CutMix for hard classes (e.g., MSH vs MSS)
- Cosine LR, AdamW, early stopping on val macro-F1
- Spectrogram-friendly augs (no rotations/flips)
- Test-time translation augmentation (TTA) for final score
- Saves: best checkpoint, metrics CSV, training curves, confusion matrix, predictions CSV

Usage (Windows CPU):
  python train_resnet_cwt.py --data_root "E:\CP Dataset\cwt" --pretrained --epochs 25 --img_size 224 --batch_size 32 --tta --save_dir ".\runs\exp1" --num_workers 0

CUDA (if available):
  python train_resnet_cwt.py --data_root "E:\CP Dataset\cwt" --pretrained --epochs 25 --img_size 224 --batch_size 64 --tta --amp --save_dir ".\runs\exp1"
"""
import argparse, time, random, math, contextlib, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class NullScaler:
    def scale(self, x): return x
    def step(self, o): o.step()
    def update(self): pass

def make_scaler(amp_enabled: bool):
    if not amp_enabled:
        return NullScaler()
    try:
        return torch.amp.GradScaler('cuda')
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(enabled=True)

@contextlib.contextmanager
def autocast_ctx(amp_enabled: bool):
    if not amp_enabled:
        yield; return
    try:
        with torch.amp.autocast('cuda'):
            yield
    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast
        with cuda_autocast():
            yield

class TimeMask(object):
    def __init__(self, max_width_ratio=0.05, p=0.20): self.max_width_ratio, self.p = max_width_ratio, p
    def __call__(self, img):
        if np.random.rand() > self.p or not torch.is_tensor(img): return img
        _, H, W = img.shape
        w = int(np.random.uniform(1, self.max_width_ratio) * W)
        s = np.random.randint(0, max(1, W - w))
        img[:, :, s:s+w] = 0.0
        return img

class FreqMask(object):
    def __init__(self, max_height_ratio=0.05, p=0.20): self.max_height_ratio, self.p = max_height_ratio, p
    def __call__(self, img):
        if np.random.rand() > self.p or not torch.is_tensor(img): return img
        _, H, W = img.shape
        h = int(np.random.uniform(1, self.max_height_ratio) * H)
        s = np.random.randint(0, max(1, H - h))
        img[:, s:s+h, :] = 0.0
        return img

class ImageFolderView(datasets.ImageFolder):
    def __init__(self, base: datasets.ImageFolder, transform=None):
        self.root = base.root
        self.loader = base.loader
        self.extensions = base.extensions
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx
        self.samples = base.samples
        self.targets = base.targets
        self.imgs = base.imgs
        self.transform = transform
        self.target_transform = None

def soft_cross_entropy(logits, targets, class_weights=None, label_smoothing=0.0):
    B, C = logits.shape
    log_probs = F.log_softmax(logits, dim=1)
    if targets.dtype == torch.long:
        with torch.no_grad():
            t = torch.zeros((B, C), device=logits.device, dtype=logits.dtype)
            t.scatter_(1, targets.view(-1,1), 1.0)
            if label_smoothing > 0:
                t = (1 - label_smoothing) * t + label_smoothing / C
    else:
        t = targets
        if label_smoothing > 0:
            t = (1 - label_smoothing) * t + label_smoothing / C
    if class_weights is not None:
        w = class_weights.view(1, -1)
        t = t * w
    loss = -(t * log_probs).sum(dim=1).mean()
    return loss

def mixup_data(x, y, alpha=0.0):
    if alpha <= 0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[index, :]
    return x_mix, (y, y[index]), lam

def rand_bbox(size, lam):
    B, C, H, W = size
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=0.0):
    if alpha <= 0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x_cut, (y, y[index]), lam_adj

def mix_criterion(logits, y_pair, lam, class_weights=None, label_smoothing=0.0):
    if not isinstance(y_pair, tuple):
        return soft_cross_entropy(logits, y_pair, class_weights, label_smoothing)
    y_a, y_b = y_pair
    loss_a = soft_cross_entropy(logits, y_a, class_weights, label_smoothing)
    loss_b = soft_cross_entropy(logits, y_b, class_weights, label_smoothing)
    return lam * loss_a + (1 - lam) * loss_b

def build_model(name, num_classes, pretrained=True):
    name = name.lower()
    if name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if name == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        return net
    if name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        return net
    raise ValueError(f"Unknown model: {name}")

def translate_batch(imgs, dx, dy):
    if dx == 0 and dy == 0: return imgs
    B,C,H,W = imgs.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(H, device=imgs.device),
                                    torch.arange(W, device=imgs.device), indexing='ij')
    x0 = torch.clamp(grid_y + dx, 0, W-1)
    y0 = torch.clamp(grid_x + dy, 0, H-1)
    return imgs[:, :, y0, x0]

def train_one_epoch(model, loader, device, optimizer, class_weights, amp=False,
                    mixup_alpha=0.0, cutmix_alpha=0.0, label_smoothing=0.05):
    model.train()
    scaler = make_scaler(amp)
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        use_mix = False; y_pair = None; lam = 1.0
        if (mixup_alpha > 0.0) or (cutmix_alpha > 0.0):
            if np.random.rand() < 0.5 and mixup_alpha > 0:
                x, y_pair, lam = mixup_data(x, y, mixup_alpha); use_mix = True
            elif cutmix_alpha > 0:
                x, y_pair, lam = cutmix_data(x, y, cutmix_alpha); use_mix = True

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(amp):
            logits = model(x)
            if use_mix:
                loss = mix_criterion(logits, y_pair, lam, class_weights, label_smoothing)
                y_ref = y if isinstance(y_pair, tuple) else y_pair
            else:
                loss = soft_cross_entropy(logits, y, class_weights, label_smoothing)
                y_ref = y
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()

        loss_sum += loss.item() * x.size(0)
        with torch.no_grad():
            preds = logits.argmax(1)
            correct += (preds == y_ref).sum().item()
            total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, class_weights, amp=False, label_smoothing=0.0):
    model.eval()
    loss_sum = correct = total = 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx(amp):
            logits = model(x)
            loss = soft_cross_entropy(logits, y, class_weights, label_smoothing)
        loss_sum += loss.item() * x.size(0); total += x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return loss_sum/total, correct/total, y_true, y_pred

@torch.no_grad()
def evaluate_tta(model, loader, device, amp=False, shifts=(0,8,-8)):
    model.eval()
    total = correct = 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        acc_prob = None; n = 0
        for dx in shifts:
            for dy in shifts:
                x_aug = translate_batch(x, dx, dy)
                with autocast_ctx(amp):
                    logits = model(x_aug)
                    p = F.softmax(logits, dim=1)
                acc_prob = p if acc_prob is None else (acc_prob + p); n += 1
        pred = (acc_prob / n).argmax(1)
        correct += (pred == y).sum().item(); total += x.size(0)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    return correct/total, np.concatenate(ys), np.concatenate(ps)

def plot_curves(history_df, out_png):
    fig = plt.figure(figsize=(8,5))
    ax1 = plt.gca()
    ax1.plot(history_df['epoch'], history_df['train_acc'], label='train_acc')
    ax1.plot(history_df['epoch'], history_df['val_acc'], label='val_acc')
    ax2 = ax1.twinx()
    ax2.plot(history_df['epoch'], history_df['val_macro_f1'], label='val_macro_f1', linestyle='--')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('accuracy'); ax2.set_ylabel('macro F1')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)

def save_confusion(y_true, y_pred, class_names, out_png):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix'); plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(range(len(class_names))); ax.set_xlabel('Pred')
    ax.set_yticks(tick_marks); ax.set_yticklabels(range(len(class_names))); ax.set_ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="./runs/exp1")
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18","mobilenet_v3_large","efficientnet_b0"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--mixup_alpha", type=float, default=0.0)
    ap.add_argument("--cutmix_alpha", type=float, default=0.0)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--amp", action="store_true", help="Enable CUDA AMP if cuda is available")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--freeze_epochs", type=int, default=2, help="Freeze backbone for first N epochs (warmup)")
    ap.add_argument("--tta", action="store_true", help="Run TTA after testing")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    amp_enabled = bool(args.amp and has_cuda)
    pin_mem = has_cuda
    print(f"Device: {device} | AMP: {amp_enabled}")

    train_tfms = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        TimeMask(0.05, 0.20),
        FreqMask(0.05, 0.20),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])
    test_tfms = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])

    base_ds = datasets.ImageFolder(args.data_root)
    targets = np.array(base_ds.targets); idxs = np.arange(len(base_ds))
    class_names = base_ds.classes; num_classes = len(class_names)
    with open(os.path.join(args.save_dir, "class_names.json"), "w", encoding="utf-8") as f:
        import json; json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"Classes ({num_classes}): {class_names}")

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_val_idx, test_idx = next(sss1.split(idxs, targets))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size/(1-args.test_size), random_state=args.seed)
    train_idx_rel, val_idx_rel = next(sss2.split(train_val_idx, targets[train_val_idx]))
    train_idx, val_idx = train_val_idx[train_idx_rel], train_val_idx[val_idx_rel]
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    np.savez(os.path.join(args.save_dir, "splits.npz"), train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    ds_train = ImageFolderView(base_ds, transform=train_tfms)
    ds_eval  = ImageFolderView(base_ds, transform=test_tfms)
    tr_sub   = Subset(ds_train, train_idx)
    va_sub   = Subset(ds_eval,   val_idx)
    te_sub   = Subset(ds_eval,   test_idx)

    train_targets = targets[train_idx]
    counts = np.bincount(train_targets, minlength=num_classes)
    cweights = torch.tensor((1.0/(counts+1e-6)) / (1.0/(counts+1e-6)).mean(), dtype=torch.float32, device=device)
    print("Class counts:", counts, "-> weights:", np.round((cweights.cpu().numpy()), 3))

    tr_loader = DataLoader(tr_sub, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=pin_mem)
    va_loader = DataLoader(va_sub, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=pin_mem)
    te_loader = DataLoader(te_sub, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=pin_mem)

    model = build_model(args.model, num_classes, pretrained=args.pretrained).to(device)
    print(f"Model: {args.model} | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # warmup freeze
    backbone = None
    if hasattr(model, 'fc'):
        backbone = [p for n,p in model.named_parameters() if not n.startswith('fc.')]
    elif hasattr(model, 'classifier'):
        last_name = list(model.classifier._modules.keys())[-1]
        backbone = [p for n,p in model.named_parameters() if not n.startswith(f'classifier.{last_name}')]
    if backbone:
        for p in backbone: p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def cosine_mult(epoch, total=args.epochs):
        return 0.5*(1+math.cos(math.pi*epoch/total))

    history = []
    best_f1, best_state, bad, patience = -1, None, 0, 7

    for epoch in range(1, args.epochs+1):
        if epoch == (args.freeze_epochs + 1) and backbone:
            for p in backbone: p.requires_grad = True

        for g in optimizer.param_groups:
            g['lr'] = max(args.lr * cosine_mult(epoch-1, args.epochs), 1e-6)

        tr_loss, tr_acc = train_one_epoch(
            model, tr_loader, device, optimizer, cweights,
            amp=bool(args.amp and torch.cuda.is_available()),
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
            label_smoothing=args.label_smoothing
        )
        va_loss, va_acc, yv, pv = evaluate(
            model, va_loader, device, cweights,
            amp=bool(args.amp and torch.cuda.is_available()),
            label_smoothing=args.label_smoothing
        )
        va_f1 = f1_score(yv, pv, average='macro')

        history.append({"epoch":epoch, "train_loss":tr_loss, "train_acc":tr_acc,
                        "val_loss":va_loss, "val_acc":va_acc, "val_macro_f1":va_f1})
        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, "metrics_log.csv"), index=False)

        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} macroF1={va_f1:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        if va_f1 > best_f1:
            best_f1, bad = va_f1, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(args.save_dir, "best.pth"))
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    te_loss, te_acc, yt, pt = evaluate(
        model, te_loader, device, cweights,
        amp=bool(args.amp and torch.cuda.is_available()),
        label_smoothing=args.label_smoothing
    )
    print(f"\nTEST: loss={te_loss:.4f} acc={te_acc:.4f}")
    print("\nClassification report:\n", classification_report(yt, pt, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(yt, pt))

    # save predictions and confusion matrix
    pd.DataFrame({"y_true": yt, "y_pred": pt}).to_csv(os.path.join(args.save_dir, "test_predictions.csv"), index=False)
    save_confusion(yt, pt, class_names, os.path.join(args.save_dir, "confusion_test.png"))

    # curves
    hist_df = pd.DataFrame(history)
    plot_curves(hist_df, os.path.join(args.save_dir, "training_curves.png"))

    # optional TTA
    if args.tta:
        tta_acc, y_tta, p_tta = evaluate_tta(model, te_loader, device, amp=bool(args.amp and torch.cuda.is_available()))
        print(f"\nTEST (TTA): acc={tta_acc:.4f}")
        print("Classification report (TTA):\n", classification_report(y_tta, p_tta, target_names=class_names, digits=4))
        print("Confusion matrix (TTA):\n", confusion_matrix(y_tta, p_tta))
        pd.DataFrame({"y_true": y_tta, "y_pred": p_tta}).to_csv(os.path.join(args.save_dir, "test_predictions_tta.csv"), index=False)
        save_confusion(y_tta, p_tta, class_names, os.path.join(args.save_dir, "confusion_test_tta.png"))

    with open(os.path.join(args.save_dir, "class_names.txt"), "w", encoding="utf-8") as f:
        for n in class_names: f.write(n+"\n")

    print(f"\nArtifacts saved in: {os.path.abspath(args.save_dir)}")

if __name__ == "__main__":
    main()