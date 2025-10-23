# ae_vis.py
# Make cleaner 2D visualizations from existing features only (t-SNE + UMAP).
# Saves multiple variants (incl. supervised UMAP) to ./ae_outputs/.

import sys, subprocess, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- Ensure deps (won't touch your training libs) ---
def ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        __import__(import_name or pkg)

ensure("pandas"); ensure("numpy"); ensure("matplotlib","matplotlib")
ensure("seaborn"); ensure("scikit-learn","sklearn")
ensure("umap-learn","umap")
from umap import UMAP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE

# --- I/O ---
OUTDIR = Path("./ae_outputs"); OUTDIR.mkdir(exist_ok=True, parents=True)
XLSX = OUTDIR / "full_ae_features.xlsx"      # scaled features if created by ae.py
CSV  = OUTDIR / "features_final.csv"         # unscaled features if XLSX missing
RANDOM_STATE = 42

# --- Load features ---
if XLSX.exists():
    df = pd.read_excel(XLSX)
    print(f"[load] Using scaled features: {XLSX}")
    scaled = True
elif CSV.exists():
    df = pd.read_csv(CSV)
    print(f"[load] Using raw features: {CSV}")
    scaled = False
else:
    raise FileNotFoundError("No features found. Run ae.py first to create ae_outputs/full_ae_features.xlsx")

if "Label" not in df.columns:
    raise ValueError("Expected a 'Label' column in the features file.")

y_text = df["Label"].values
X = df.drop(columns=["Label"]).values

# Scale if needed (features_final.csv is unscaled)
if not scaled:
    print("[scale] Standardizing features for visualization…")
    X = StandardScaler().fit_transform(X)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_text)
class_names = list(le.classes_)

# --- nice plotting helper ---
def plot_2d(Z, y, class_names, title, save_path):
    markers = ['o','s','^','D','P','X','*','v']
    colors  = ['C0','C1','C2','C3','C4','C5','C6','C7']
    plt.figure(figsize=(8,6))
    for idx, name in enumerate(class_names):
        m = (y == idx)
        plt.scatter(Z[m,0], Z[m,1],
                    marker=markers[idx % len(markers)],
                    label=name, s=36, linewidths=0.5, alpha=0.9,
                    edgecolors='none', c=colors[idx % len(colors)])
    plt.xlabel(title.split()[0] + " 1", fontsize=14, fontweight="bold")
    plt.ylabel(title.split()[0] + " 2", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(title="Class", fontsize=10, title_fontsize=11, loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[save] {save_path}")

# --- t-SNE variants (try a couple that often separate better) ---
def make_tsne(**kw):
    # keep constructor version-flexible
    try:
        return TSNE(n_components=2, random_state=RANDOM_STATE, **kw)
    except TypeError:
        # older sklearn: drop unsupported args
        kw.pop("learning_rate", None)
        kw.pop("n_iter", None)
        kw.pop("init", None)
        return TSNE(n_components=2, random_state=RANDOM_STATE, **kw)

tsne_settings = [
    dict(perplexity=30, learning_rate=200, n_iter=1000, init="pca"),
    dict(perplexity=50, learning_rate=300, n_iter=1500, init="pca"),
]

for i, params in enumerate(tsne_settings, 1):
    tsne = make_tsne(**params)
    Z = tsne.fit_transform(X)
    tag = f"p{params.get('perplexity')}_lr{params.get('learning_rate', 'na')}"
    plot_2d(Z, y, class_names, f"t-SNE ({tag})", OUTDIR / f"tsne_{tag}.png")

# --- UMAP variants (unsupervised) ---
umap_settings = [
    dict(n_neighbors=15, min_dist=0.1, metric="euclidean"),
    dict(n_neighbors=30, min_dist=0.2, metric="euclidean"),
    dict(n_neighbors=50, min_dist=0.3, metric="euclidean"),
]
for params in umap_settings:
    umap = UMAP(n_components=2, random_state=RANDOM_STATE, **params)
    Z = umap.fit_transform(X)
    tag = f"nn{params['n_neighbors']}_md{params['min_dist']}".replace(".","p")
    plot_2d(Z, y, class_names, f"UMAP ({tag})", OUTDIR / f"umap_{tag}.png")

# --- Supervised UMAP (often the cleanest separation) ---
# Uses labels to inject class structure; choose a slightly larger n_neighbors to capture global clusters.
sup = UMAP(n_components=2, random_state=RANDOM_STATE,
           n_neighbors=30, min_dist=0.15, metric="euclidean", target_metric="categorical")
Z_sup = sup.fit_transform(X, y=y)
plot_2d(Z_sup, y, class_names, "UMAP (supervised, nn30_md0.15)", OUTDIR / "umap_supervised.png")

print("\nDone. Check the PNGs in ./ae_outputs/")
