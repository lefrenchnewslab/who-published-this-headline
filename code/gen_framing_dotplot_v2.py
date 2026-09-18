import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

rows = [
    ("BFMTV — agent_subject",   +0.056, 0.018, "confirmatory"),
    ("TF1 INFO — passive",       +0.064, 0.024, "exploratory"),
    ("TF1 INFO — agent_subject", -0.074, 0.026, "exploratory"),
    ("BFMTV — verbless",         -0.074, 0.028, "exploratory"),
    ("Les Échos — passive",      -0.064, 0.027, "exploratory"),
    ("Les Échos — attribution",  -0.008, 0.021, "null"),
    ("BFMTV — attribution",      +0.020, 0.023, "null"),
]

labels = [r[0] for r in rows]
betas  = np.array([r[1] for r in rows])
ses    = np.array([r[2] for r in rows])
cats   = [r[3] for r in rows]

colors = {"confirmatory": "#d7191c", "exploratory": "#f4a623", "null": "#aaaaaa"}
dot_colors = [colors[c] for c in cats]

fig, ax = plt.subplots(figsize=(5.0, 2.8))
y = np.arange(len(rows))[::-1]

for yi, beta, se, col in zip(y, betas, ses, dot_colors):
    ci = 1.96 * se
    ax.plot([beta - ci, beta + ci], [yi, yi], color=col, linewidth=1.4, solid_capstyle='round')
    ax.scatter([beta], [yi], color=col, s=30, zorder=5)
    sign = '+' if beta > 0 else ''
    ha = 'left' if beta >= 0 else 'right'
    offset = 0.004 if beta >= 0 else -0.004
    ax.text(beta + offset, yi, f"{sign}{beta:.3f}",
            va='center', ha=ha, fontsize=6.5, color=col, fontweight='bold')

ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel(r"Within-event $\beta$ (deviation from event mean)", fontsize=7.5)
ax.set_xlim(-0.14, 0.14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linewidth=0.4, color='#ddd', zorder=0)

sep_y = (y[4] + y[5]) / 2
ax.axhline(sep_y, color='#ccc', linewidth=0.6, linestyle=':')
ax.text(0.135, sep_y + 0.18, 'pre-reg. n.s.', fontsize=5.5, color='#aaa', ha='right')

dot_c = mlines.Line2D([], [], color=colors["confirmatory"], marker='o', linestyle='None', markersize=5, label='Confirmatory (Bonferroni)')
dot_e = mlines.Line2D([], [], color=colors["exploratory"],  marker='o', linestyle='None', markersize=5, label='Exploratory (FDR)')
dot_n = mlines.Line2D([], [], color=colors["null"],         marker='o', linestyle='None', markersize=5, label='Pre-reg. n.s.')
ax.legend(handles=[dot_c, dot_e, dot_n], fontsize=6, loc='lower right', framealpha=0.9, edgecolor='#ccc')

plt.tight_layout(pad=0.4)
from pathlib import Path
_OUT_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
out = str(_OUT_DIR / "fig_framing_effects.pdf")
plt.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
print("Saved framing effects v2")
