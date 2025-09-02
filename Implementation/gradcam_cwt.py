# -*- coding: utf-8 -*-
r"""
Grad-CAM for CWT images (works with ResNet18/MobileNet/EfficientNet from train script)
Saves overlays for randomly sampled test images per class.

Example:
  python gradcam_cwt.py --data_root "E:\CP Dataset\cwt" --ckpt ".\runs\exp1\best.pth" --save_dir ".\runs\exp1\gradcam" --num_samples 3 --num_workers 0
"""
import argparse, os, random, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

def find_last_conv(module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._fwd_hook)
        self.h2 = target_layer.register_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, m, gin, gout):
        self.gradients = gout[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # [B,C]
        if class_idx is None:
            class_idx = logits.argmax(1)
        loss = logits.gather(1, class_idx.view(-1,1)).sum()
        loss.backward()
        # weights: GAP over gradients
        weights = self.gradients.mean(dim=(2,3), keepdim=True)  # [B,K,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        # normalize 0..1 per image
        B,_,H,W = cam.shape
        cams = []
        for i in range(B):
            m = cam[i,0]; m -= m.min(); m = m / (m.max() + 1e-6)
            cams.append(m.cpu().numpy())
        return logits.detach(), np.stack(cams, axis=0)

    def close(self):
        self.h1.remove(); self.h2.remove()

def overlay_heatmap(img_tensor, cam2d, out_path, alpha=0.4):
    # img_tensor: CHW normalized to [-1,1] (we used mean=0.5, std=0.5)
    img = img_tensor.clone()
    img = (img * 0.5 + 0.5).clamp(0,1)  # back to 0..1
    img = img.permute(1,2,0).cpu().numpy()  # HWC
    H, W, _ = img.shape
    cam = Image.fromarray((cam2d*255).astype(np.uint8)).resize((W,H), Image.BILINEAR)
    cam = np.array(cam)/255.0
    fig = plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pth from training")
    ap.add_argument("--save_dir", type=str, default="./gradcam_out")
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18","mobilenet_v3_large","efficientnet_b0"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_samples", type=int, default=3, help="Per class")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    tfm = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

    base = datasets.ImageFolder(args.data_root)
    classes = base.classes; K = len(classes)
    # create a small "test-like" sample set per class (uniform random)
    idxs = np.arange(len(base.targets))
    per_class = {i: np.where(np.array(base.targets)==i)[0] for i in range(K)}
    chosen = []
    for i in range(K):
        if len(per_class[i]) == 0: continue
        cidx = np.random.choice(per_class[i], size=min(args.num_samples, len(per_class[i])), replace=False)
        chosen.extend(cidx.tolist())
    ds = ImageFolderView(base, transform=tfm)
    subset = Subset(ds, chosen)
    loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=args.num_workers)

    # build model & load weights
    model = build_model(args.model, K)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # target layer: last conv
    last_conv = find_last_conv(model)
    cam = GradCAM(model, last_conv)

    # iterate and save overlays
    counter = 0
    for (x, y) in loader:
        x = x.to(device)
        logits, cams = cam(x)
        preds = logits.argmax(1).cpu().numpy()
        for i in range(x.size(0)):
            true_c = classes[y[i].item()]
            pred_c = classes[preds[i].item()]
            outp = os.path.join(args.save_dir, f"{counter:04d}_true-{true_c}_pred-{pred_c}.png")
            overlay_heatmap(x[i].cpu(), cams[i], outp)
            counter += 1
    cam.close()
    print(f"Saved {counter} Grad-CAM overlays to: {os.path.abspath(args.save_dir)}")

if __name__ == "__main__":
    main()