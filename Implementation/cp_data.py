import os, scipy.io as sio, pandas as pd
from tqdm import tqdm
from cp_utils import canonical_label, ensure_dirs, extract_pressure
from cp_transforms import save_timefreq_images
from cp_config import BASE_DIR

def scan_mat_files(base_dir):
    recs = []
    for root, _, files in os.walk(base_dir):
        if "_paper_outputs" in root:  # skip outputs
            continue
        folder = os.path.basename(root)
        label = canonical_label(folder)
        if label == "Unknown": 
            continue
        for fn in files:
            if fn.lower().endswith(".mat"):
                recs.append({"mat_path": os.path.join(root, fn),
                             "label": label, "folder": folder})
    return recs

def build_image_dataset(out_dir):
    mats = scan_mat_files(BASE_DIR)
    if not mats:
        raise RuntimeError("No .mat files found. Check BASE_DIR in cp_config.py")
    print(f"Found {len(mats)} .mat files. Generating images...")

    img_root = os.path.join(out_dir, "images")
    ensure_dirs(img_root)

    rows = []
    for m in tqdm(mats):
        try:
            d = sio.loadmat(m["mat_path"])
            sig = d["signal"]; fs = float(d["fs"].squeeze())
            press = extract_pressure(m["folder"])
            stem = os.path.splitext(os.path.basename(m["mat_path"]))[0]
            rows.extend(save_timefreq_images(sig, fs, m["label"], press, stem, img_root, all_mats=mats))
        except Exception as e:
            print(f"[WARN] {m['mat_path']}: {e}")

    meta = pd.DataFrame(rows)
    meta_csv = os.path.join(out_dir, "metadata_images.csv")
    meta.to_csv(meta_csv, index=False)
    print(f"Saved: {meta_csv} (images: {len(meta)})")
    return meta