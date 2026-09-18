"""
Generate all analysis outputs missing from analysis/ vs legacy_analysis/.
Run from the repository root on any machine with pyarrow + matplotlib + sklearn.

Outputs written to analysis/:
  - confusion_camembert_temporal_maskent.csv/png   (already done, will overwrite)
  - confusion_camembert_random_raw.csv/png
  - confusion_charsvm_temporal_maskent.csv/png
  - table4_features_char_svm.csv
  - table4_features_char_svm_mask_outlet.csv
  - table4_features_word_lr.csv
  - table4_features_word_lr_mask_outlet.csv
  - qualitative_examples.csv
  - audit_testB_templates.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent  # package root
PREDS = ROOT / "results/n10/runs/preds"
OUT   = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

OUTLET_ORDER = [
    'BFMTV', 'Ouest-France', "L'Humanité", 'TF1 INFO', 'Franceinfo',
    'Les Echos', 'Le Monde', 'Libération', 'Le Parisien', 'La Croix', 'Le Figaro'
]

SHORT = {
    'BFMTV':'BFMTV','Ouest-France':'O-F',"L'Humanité":'Huma','TF1 INFO':'TF1',
    'Franceinfo':'Finfo','Les Echos':'Échos','Le Monde':'Monde',
    'Libération':'Libé','Le Parisien':'Parisien','La Croix':'Croix','Le Figaro':'Figaro'
}

def plot_confusion(cm, labels, title, path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = [SHORT.get(l, l) for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(ticks, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(ticks, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('True', fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if v > 0.5 else 'black')
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PNG: {path}")

def save_confusion_csv(cm, labels, path):
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(path)
    print(f"  CSV: {path}")

# ── 1. CamemBERT temporal entity-masked (s42) ───────────────────────────────
print("1. CamemBERT temporal mask_entities s42")
df = pd.read_parquet(PREDS / "n10_camembert_pooled_ALL_headline_only_mask_entities_temporal_s42_lr1e_05.parquet")
print(f"   {len(df):,} rows, accuracy={df['correct'].mean():.4f}")
cm = confusion_matrix(df['true'], df['pred'], labels=OUTLET_ORDER)
save_confusion_csv(cm, OUTLET_ORDER, OUT / "confusion_camembert_temporal_maskent.csv")
plot_confusion(cm, OUTLET_ORDER, "CamemBERT temporal entity-masked (s42) — row-normalized",
               OUT / "confusion_camembert_temporal_maskent.png")

# ── 2. CamemBERT random raw (s42) ───────────────────────────────────────────
print("2. CamemBERT random raw s42")
df = pd.read_parquet(PREDS / "n10_camembert_pooled_ALL_headline_only_raw_random_s42_lr1e_05.parquet")
print(f"   {len(df):,} rows, accuracy={df['correct'].mean():.4f}")
cm = confusion_matrix(df['true'], df['pred'], labels=OUTLET_ORDER)
save_confusion_csv(cm, OUTLET_ORDER, OUT / "confusion_camembert_random_raw.csv")
plot_confusion(cm, OUTLET_ORDER, "CamemBERT random raw (s42) — row-normalized",
               OUT / "confusion_camembert_random_raw.png")

# ── 3. Char-SVM temporal entity-masked ──────────────────────────────────────
print("3. Char-SVM temporal mask_entities s42")
df = pd.read_parquet(PREDS / "n10_char_svm_pooled_ALL_headline_only_mask_entities_temporal_s42.parquet")
print(f"   {len(df):,} rows, accuracy={df['correct'].mean():.4f}")
cm = confusion_matrix(df['true'], df['pred'], labels=OUTLET_ORDER)
save_confusion_csv(cm, OUTLET_ORDER, OUT / "confusion_charsvm_temporal_maskent.csv")
plot_confusion(cm, OUTLET_ORDER, "Char-SVM temporal entity-masked — row-normalized",
               OUT / "confusion_charsvm_temporal_maskent.png")

# ── 4. Feature importance from feature_importance.parquet ───────────────────
print("4. Feature importance tables")
fi = pd.read_parquet(ROOT / "results/n10/feature_importance.parquet")
print(f"   Columns: {list(fi.columns)}")
print(f"   Experiments: {fi['experiment'].unique()}")

for exp_tag, pp_label in [
    ('char_svm_mask_outlet', 'mask_outlet'),
    ('char_svm_raw',         'raw'),
    ('word_lr_mask_outlet',  'mask_outlet'),
    ('word_lr_raw',          'raw'),
]:
    sub = fi[fi['experiment'].str.contains(exp_tag.split('_')[0]) &
             fi['experiment'].str.contains(pp_label if pp_label != 'raw' else 'raw')]
    if sub.empty:
        # fallback: match by model and preprocess columns if they exist
        model = 'char_svm' if 'char_svm' in exp_tag else 'word_lr'
        sub = fi[(fi['model'] == model) & (fi['preprocess'] == pp_label)]
    if sub.empty:
        print(f"   WARNING: no data for {exp_tag}")
        continue
    out = sub.groupby('outlet')['feature'].apply(lambda x: list(x[:15])).reset_index()
    out.columns = ['outlet', 'top_features']
    out['top_features'] = out['top_features'].apply(str)
    fname = f"table4_features_{exp_tag}.csv"
    out.to_csv(OUT / fname, index=False)
    print(f"   {OUT / fname}")

# ── 5. Qualitative examples (most confident correct + incorrect) ─────────────
print("5. Qualitative examples")
df = pd.read_parquet(PREDS / "n10_camembert_pooled_ALL_headline_only_mask_entities_temporal_s42_lr1e_05.parquet")
correct   = df[df['correct']].nlargest(20, 'confidence')[['true','pred','confidence']]
incorrect = df[~df['correct']].nlargest(20, 'confidence')[['true','pred','confidence']]
correct['kind']   = 'confident_correct'
incorrect['kind'] = 'confident_incorrect'
examples = pd.concat([correct, incorrect]).sort_values('confidence', ascending=False)
examples.to_csv(OUT / "qualitative_examples.csv", index=False)
print(f"   {OUT / 'qualitative_examples.csv'}")

# ── 6. Template leakage audit ────────────────────────────────────────────────
print("6. Template leakage audit")
df = pd.read_parquet(PREDS / "n10_camembert_pooled_ALL_headline_only_raw_temporal_s42_lr1e_05.parquet")
if 'text' in df.columns:
    # Extract first 8 words as "lead"
    df['lead8'] = df['text'].str.split().str[:8].str.join(' ').str.lower()
    rows = []
    for outlet, grp in df.groupby('true'):
        lead_counts = grp['lead8'].value_counts()
        top = lead_counts.index[0]
        share = lead_counts.iloc[0] / len(grp)
        rows.append({'outlet': outlet, 'top_lead8': top, 'share': round(share, 4)})
    audit = pd.DataFrame(rows).sort_values('share', ascending=False)
    audit.to_csv(OUT / "audit_testB_templates.csv", index=False)
    print(f"   {OUT / 'audit_testB_templates.csv'}")
    print(f"   Max template share: {audit['share'].max():.4f} ({audit.iloc[0]['outlet']})")
else:
    print("   WARNING: 'text' column not in pred parquet — skipping template audit")

print("\nDone. All outputs in", OUT)
