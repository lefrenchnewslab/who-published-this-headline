"""
Generate confusion matrix for the headline model (CamemBERT, temporal, entity-masked, seed 42).
Run from any machine with pyarrow + matplotlib installed (e.g. the Colab where phase1 ran).

Outputs:
  analysis/confusion_camembert_temporal_maskent.csv
  analysis/confusion_camembert_temporal_maskent.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent  # package root
PREDS_DIR = ROOT / "data" / "model_predictions"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(exist_ok=True)

# Use seed 42 (consistent with table3_per_outlet_temporal.csv)
pred_file = PREDS_DIR / "n10_camembert_pooled_ALL_headline_only_mask_entities_temporal_s42_lr1e_05.parquet"
df = pd.read_parquet(pred_file)

print(f"Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")
print(f"Overall accuracy: {df['correct'].mean():.4f}")

# Outlet order: sort by F1 descending (from table3) for readability
outlet_order = [
    'BFMTV', 'Ouest-France', 'TF1 INFO', 'Franceinfo',
    'Les Echos', 'Le Monde', 'Libération', 'Le Parisien', 'La Croix', 'Le Figaro'
]

# Build confusion matrix
cm = confusion_matrix(df['true'], df['pred'], labels=outlet_order)

# Save CSV
cm_df = pd.DataFrame(cm, index=outlet_order, columns=outlet_order)
csv_path = OUT_DIR / "confusion_camembert_temporal_maskent.csv"
cm_df.to_csv(csv_path)
print(f"Saved CSV: {csv_path}")

# Row-normalize for figure
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# Plot
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(len(outlet_order)))
ax.set_yticks(range(len(outlet_order)))
short = [o.replace("L'", "L'").replace('Ouest-France','O-F').replace('Le Parisien','Parisien')
           .replace('Franceinfo','Finfo').replace("L'Humanité","Huma").replace('Libération','Libé')
           .replace('Les Echos','Échos').replace('La Croix','Croix').replace('Le Figaro','Figaro')
           .replace('Le Monde','Monde').replace('TF1 INFO','TF1') for o in outlet_order]
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short, fontsize=9)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)

# Annotate cells
for i in range(len(outlet_order)):
    for j in range(len(outlet_order)):
        val = cm_norm[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.title('CamemBERT temporal entity-masked — row-normalized confusion matrix', fontsize=10)
plt.tight_layout()

png_path = OUT_DIR / "confusion_camembert_temporal_maskent.png"
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"Saved PNG: {png_path}")
plt.close()
print("Done.")
