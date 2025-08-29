import os, pandas as pd, matplotlib.pyplot as plt
from cp_config import BASE_DIR, OUT_DIR_NAME, TRANSFORMS, ARCHS, SEQ_TRANSFORM, SEQ_T
from cp_utils import ensure_dirs, set_seeds
from cp_data import build_image_dataset
from cp_features import build_feature_table, train_classic_ml
from cp_dl import run_dl_experiments, run_cnn_lstm_experiment

def plot_overall_summary(out_dir, classic_df, dl_df):
    all_df = pd.concat([classic_df, dl_df], ignore_index=True)
    all_df.to_csv(os.path.join(out_dir, "overall_summary.csv"), index=False)

    plt.figure(figsize=(10,5))
    all_df.plot(kind="bar", x="method", y="accuracy", ax=plt.gca())
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Accuracy"); plt.title("Overall Accuracy Comparison")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "overall_accuracy_bar.png"), dpi=150); plt.close()

    dl_only = all_df[all_df["method"].str.startswith("DL_")]
    if not dl_only.empty:
        piv = dl_only.pivot(index="arch", columns="transform", values="accuracy")
        plt.figure(figsize=(7,5)); piv.plot(kind="bar", ax=plt.gca())
        plt.ylabel("Accuracy"); plt.title("DL Accuracy by Transform")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "dl_accuracy_by_transform.png"), dpi=150); plt.close()

def main():
    set_seeds()
    out_dir = os.path.join(BASE_DIR, OUT_DIR_NAME)
    ensure_dirs(out_dir)

    print("\n[1/4] Generate time–frequency images (STFT/CWT/WCA/EMD) ...")
    meta_images = build_image_dataset(out_dir)

    print("\n[2/4] Handcrafted features + Classic ML ...")
    feat_df = build_feature_table(out_dir)
    classic_summary = train_classic_ml(feat_df, out_dir)

    print("\n[3/4] Deep learning experiments ...")
    dl_summaries = []
    dl_out_root = os.path.join(out_dir, "dl"); ensure_dirs(dl_out_root)
    for tfm in TRANSFORMS:
        dl_summaries.append(run_dl_experiments(meta_images, transform=tfm, archs=ARCHS, out_root=dl_out_root))
    dl_summary = pd.concat([d for d in dl_summaries if d is not None], ignore_index=True) if dl_summaries else pd.DataFrame()

    print("\n[3b] CNN-LSTM (sequence model) ...")
    cnnlstm_summary = run_cnn_lstm_experiment(meta_images, transform_for_seq=SEQ_TRANSFORM,
                                              out_root=dl_out_root, T=SEQ_T)
    if not cnnlstm_summary.empty:
        dl_summary = pd.concat([dl_summary, cnnlstm_summary], ignore_index=True)

    print("\n[4/4] Plot overall comparisons ...")
    plot_overall_summary(out_dir, classic_summary, dl_summary)

    print(f"\nAll outputs saved under: {out_dir}")

if __name__ == "__main__":
    main()