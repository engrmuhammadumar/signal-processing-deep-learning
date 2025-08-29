import os, re, random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

from cp_config import SEED

def set_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def canonical_label(folder_name: str) -> str:
    name = folder_name.lower()
    if "normal" in name: return "Normal"
    if "impeller" in name: return "ImpellerCrack"
    if "mechanical seal hole" in name or "seal hole" in name: return "SealHole"
    if "mechanical seal scratch" in name or "seal scratch" in name: return "SealScratch"
    return "Unknown"

def extract_pressure(folder_name: str) -> str:
    m = re.search(r"\(([^)]+)\)", folder_name)
    return m.group(1).replace(" ", "") if m else ""

def detrend_and_norm(x):
    x = x.astype(float)
    x = x - np.mean(x)
    x = detrend(x, type="linear")
    std = np.std(x) + 1e-8
    return x / std

def segment_indices(n_samples, fs, win_sec, overlap):
    win = int(win_sec * fs)
    if win <= 0 or n_samples < win: return [(0, n_samples)]
    hop = max(1, int(win * (1 - overlap)))
    return [(s, s+win) for s in range(0, n_samples - win + 1, hop)]

def plot_image_and_save(arr, out_png, title=None, extent=None):
    import matplotlib
    plt.figure(figsize=(6,4))
    if extent is not None:
        plt.imshow(arr, aspect="auto", origin="lower", extent=extent)
    else:
        plt.imshow(arr, aspect="auto", origin="lower")
    plt.axis("off")
    if title: plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()