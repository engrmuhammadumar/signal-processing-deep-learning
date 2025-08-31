# export_predictions.py
# Exports test-set predictions (soft-voted ensemble) as a CSV with per-class probabilities.
# Requires checkpoints produced by train_ensemble.py (cnn1_local.pt, cnn2_global.pt, cnn3_compact.pt)
# and split_indices.json saved by that script.

import os, json, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import CLASSES

# ---------- models (same as training) ----------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, pool=True):
        super().__init__()
        if p is None: p = k//2
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
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(128, num_classes)
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
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.head(self.b4(self.b3(self.b2(self.b1(x)))))

class DWSeparableConv(nn.Module):
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
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(256, num_classes))
    def forward(self, x): return self.head(self.blocks(self.stem(x)))

# ---------- data ----------
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

@torch.no_grad()
def predict_probs(model, loader, device, tta=1):
    model.eval()
    all_probs=[]
    for x,_ in loader:
        x = x.to(device)
        if tta<=1:
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        else:
            acc = 0
            for i in range(tta):
                x_aug = x
                if i%2==1: x_aug = torch.flip(x_aug, dims=[3])
                if i%4==2: x_aug = torch.flip(x_aug, dims=[2])
                logits = model(x_aug)
                acc += F.softmax(logits, dim=1)
            probs = (acc/tta).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="WCA3 root with 4 class folders")
    ap.add_argument("--checkpoints", default="checkpoints", help="Folder with trained .pt and split_indices.json")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--out_csv", default="preds_test.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 1 if args.grayscale else 3

    # load split
    split_path = Path(args.checkpoints) / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Re-run training script so it saves the split.")
    split = json.loads(Path(split_path).read_text())
    test_idx = split["test"]

    # dataloader
    loader, base = make_eval_loader(args.data_dir, args.img_size, test_idx, grayscale=args.grayscale, batch_size=32)
    idx_to_class = {v:k for k,v in base.class_to_idx.items()}
    # paths + true labels
    all_paths = [base.samples[i][0] for i in test_idx]
    true_idx   = [base.samples[i][1] for i in test_idx]
    true_names = [idx_to_class[i] for i in true_idx]

    # models
    m1 = CNN1_Local(num_classes=len(CLASSES), in_ch=in_ch)
    m2 = CNN2_Global(num_classes=len(CLASSES), in_ch=in_ch)
    m3 = CNN3_Compact(num_classes=len(CLASSES), in_ch=in_ch)

    # load weights
    def load_w(m, name):
        p = Path(args.checkpoints) / name
        if not p.exists(): raise FileNotFoundError(f"Missing checkpoint: {p}")
        state = torch.load(p, map_location="cpu")
        m.load_state_dict(state)
        m.to(device)
        m.eval()

    load_w(m1, "cnn1_local.pt")
    load_w(m2, "cnn2_global.pt")
    load_w(m3, "cnn3_compact.pt")

    # predictions (probs)
    p1 = predict_probs(m1, loader, device, tta=args.tta)
    p2 = predict_probs(m2, loader, device, tta=args.tta)
    p3 = predict_probs(m3, loader, device, tta=args.tta)
    p_ens = (p1 + p2 + p3) / 3.0

    pred_idx  = p_ens.argmax(axis=1)
    pred_prob = p_ens.max(axis=1)
    pred_name = [CLASSES[i] for i in pred_idx]

    # build CSV
    rows = []
    for i in range(len(all_paths)):
        row = {
            "path": all_paths[i],
            "true": true_names[i],
            "pred": pred_name[i],
            "pred_prob": float(pred_prob[i]),
        }
        # per-class probabilities (ensemble)
        for j, cname in enumerate(CLASSES):
            row[f"p_{cname}"] = float(p_ens[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved predictions CSV -> {args.out_csv}")

if __name__ == "__main__":
    main()
