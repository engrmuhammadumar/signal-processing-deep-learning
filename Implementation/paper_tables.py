# -*- coding: utf-8 -*-
r"""
Convert metrics_log.csv (from training) into a LaTeX table snippet for paper.
Example:
  python paper_tables.py --log_csv ".\runs\exp1\metrics_log.csv" --out_tex ".\runs\exp1\metrics_table.tex"
"""
import argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", type=str, required=True)
    ap.add_argument("--out_tex", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.log_csv)
    # show last-epoch and best macro-F1 row
    best = df.loc[df['val_macro_f1'].idxmax()]
    last = df.iloc[-1]

    tex = []
    tex.append(r"\begin{table}[h]")
    tex.append(r"\centering")
    tex.append(r"\caption{Validation performance during training (best macro-F1).}")
    tex.append(r"\begin{tabular}{c|ccc}")
    tex.append(r"\hline")
    tex.append(r"Epoch & Val Acc & Val Macro-F1 & Val Loss \\")
    tex.append(r"\hline")
    tex.append(f"{int(best['epoch'])} & {best['val_acc']:.3f} & {best['val_macro_f1']:.3f} & {best['val_loss']:.3f} \\\\")
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print("Wrote:", args.out_tex)

if __name__ == "__main__":
    main()