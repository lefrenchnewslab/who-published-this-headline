"""
Decomposition waterfall v2:
- CamemBERT stages (red): Raw, Outlet-masked, Entity-masked
- Visual gap + model-switch label
- Char-SVM stages (blue): Raw (0.496), Typo-norm, Surface-stripped, Surface+NER
- Correct 44%/14%/43% brackets over char-SVM stages only
- No cross-model delta annotation
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

CHANCE = 0.100

# Two groups; gap between them
# x positions: 0,1,2 for CamemBERT; 3.6,4.6,5.6,6.6 for char-SVM (0.6 gap)
cb_stages  = [("Raw",            0.615), ("Outlet\nmasked", 0.620), ("Entity\nmasked", 0.576)]
svm_stages = [("Char-SVM\nraw",  0.496), ("Typo\nnorm.",   0.427), ("Surface\nstripped", 0.324), ("Surface\n+NER", 0.270)]

cb_x  = np.array([0, 1, 2])
svm_x = np.array([3.5, 4.5, 5.5, 6.5])

CB_COL  = "#d7191c"
SVM_COL = "#2c7bb6"
W = 0.55

fig, ax = plt.subplots(figsize=(6.2, 3.2))

# Draw bars
for xi, (lbl, acc) in zip(cb_x, cb_stages):
    ax.bar(xi, acc, color=CB_COL, width=W, zorder=3, edgecolor='white', linewidth=0.4)
    ax.text(xi, acc + 0.008, f"{acc:.3f}", ha='center', va='bottom', fontsize=7.2, fontweight='bold')

for xi, (lbl, acc) in zip(svm_x, svm_stages):
    ax.bar(xi, acc, color=SVM_COL, width=W, zorder=3, edgecolor='white', linewidth=0.4)
    ax.text(xi, acc + 0.008, f"{acc:.3f}", ha='center', va='bottom', fontsize=7.2, fontweight='bold')

# Chance line
ax.axhline(CHANCE, color='black', linewidth=0.8, linestyle='--', zorder=2)
ax.text(6.8, CHANCE + 0.01, 'Chance', fontsize=6, color='#444')

# Small delta annotations within char-SVM group
svm_accs = [s[1] for s in svm_stages]
for i in range(1, len(svm_stages)):
    diff = svm_accs[i] - svm_accs[i-1]
    mid_x = (svm_x[i-1] + svm_x[i]) / 2
    mid_y = (svm_accs[i-1] + svm_accs[i]) / 2 + 0.01
    ax.text(mid_x, mid_y, f"{diff:+.3f}", fontsize=6, ha='center', color='#666', style='italic')

# Small delta within CamemBERT (entity masking only — outlet masking is essentially 0)
ax.text((cb_x[1]+cb_x[2])/2, (0.620+0.576)/2 + 0.01, "−0.044", fontsize=6, ha='center', color='#666', style='italic')

# ---- Decomposition brackets (char-SVM above-chance margin = 0.396) ----
MARGIN = svm_accs[0] - CHANCE  # 0.396

# 44% presentation: from SVM raw (0.496) → surface stripped (0.324)
# Shown as a brace spanning svm_x[0] to svm_x[2]
brace_y = -0.045
ax.annotate("", xy=(svm_x[0]-W/2+0.05, brace_y), xytext=(svm_x[2]+W/2-0.05, brace_y),
            xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle='<->', color='#555', lw=0.9))
ax.text((svm_x[0]+svm_x[2])/2, brace_y - 0.028, "44% surface/presentation",
        ha='center', fontsize=6.5, color='#555')

# 14% entities: surface stripped → surface+NER
ax.annotate("", xy=(svm_x[2]+W/2+0.05, brace_y), xytext=(svm_x[3]-W/2-0.05+0.55, brace_y),
            xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle='<->', color='#555', lw=0.9))
ax.text((svm_x[2]+svm_x[3])/2, brace_y - 0.028, "14%\nentities",
        ha='center', fontsize=6.5, color='#555')

# 43% lexical residual: bracket from 0.270 down to chance
res_x = svm_x[3] + 0.42
ax.annotate("", xy=(res_x, CHANCE + 0.005), xytext=(res_x, svm_accs[3] - 0.008),
            arrowprops=dict(arrowstyle='<->', color='#888', lw=0.9))
ax.text(res_x + 0.08, (CHANCE + svm_accs[3]) / 2, "43%\nlexical", fontsize=6, color='#888', va='center')

# Model-switch label between groups
ax.text(3.0, 0.55, "model\nswitch", fontsize=5.5, ha='center', color='#999', style='italic')
ax.axvline(3.0, color='#ccc', linewidth=0.8, linestyle=':', zorder=1)

# X-tick labels
all_x   = list(cb_x) + list(svm_x)
all_lbl = [s[0] for s in cb_stages] + [s[0] for s in svm_stages]
ax.set_xticks(all_x)
ax.set_xticklabels(all_lbl, fontsize=7.5)
ax.set_ylabel("Accuracy", fontsize=8)
ax.set_ylim(-0.10, 0.75)
ax.set_xlim(-0.55, 7.3)
ax.yaxis.set_tick_params(labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linewidth=0.4, color='#ddd', zorder=0)

patch_cb  = mpatches.Patch(color=CB_COL,  label='CamemBERT (temporal)')
patch_svm = mpatches.Patch(color=SVM_COL, label='Char-SVM (temporal)')
ax.legend(handles=[patch_cb, patch_svm], fontsize=6.5, loc='upper right',
          framealpha=0.85, edgecolor='#ccc')

plt.tight_layout(pad=0.5)
from pathlib import Path
_OUT_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
out = str(_OUT_DIR / "fig_decomp_waterfall.pdf")
plt.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
print("Saved waterfall v2")
