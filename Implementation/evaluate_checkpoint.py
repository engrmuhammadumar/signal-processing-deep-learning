# -*- coding: utf-8 -*-
r"""
Evaluate a saved checkpoint on the saved split (from training) OR on all data.
Useful for double-checking numbers and exporting predictions again.

Example:
  python evaluate_checkpoint.py --data_root "E:\CP Dataset\cwt" --ckpt ".\runs\exp1\best.pth" --split_npz ".\runs\exp1\splits.npz" --save_dir ".\runs\exp1"
"""
import argparse, os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

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

def build_model(name, num_classes):
    name = name.lower()
    if name == 'resnet18':
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if name == 'mobilenet_v3_large':
        net = models.mobilenet_v3_large(weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        return net
    if name == 'efficientnet_b0':
        net = models.efficientnet_b0(weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        return net
    raise ValueError("Unknown model")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split_npz", type=str, required=True, help="splits.npz from training")
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18","mobilenet_v3_large","efficientnet_b0"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.data_root, exist_ok=True)  # no-op if exists
    base = datasets.ImageFolder(args.data_root)
    test_tfms = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    ds = ImageFolderView(base, transform=test_tfms)
    classes = base.classes; K = len(classes)

    sp = np.load(args.split_npz)
    test_idx = sp["test_idx"]
    sub = Subset(ds, test_idx)

    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, K)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
            all_y.append(y.numpy()); all_p.append(pred)
    y = np.concatenate(all_y); p = np.concatenate(all_p)

    print("TEST metrics:")
    print(classification_report(y, p, target_names=classes, digits=4))
    print("Confusion:\n", confusion_matrix(y, p))

    pd.DataFrame({"y_true": y, "y_pred": p}).to_csv(os.path.join(os.path.dirname(args.ckpt), "test_predictions_reval.csv"), index=False)
    print("Saved predictions CSV next to checkpoint.")

if __name__ == "__main__":
    main()