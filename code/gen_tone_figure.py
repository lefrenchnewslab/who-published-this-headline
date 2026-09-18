"""
Regenerate fig_tier1_tone (Appendix D "Tone Analysis") from released tables — NO corpus, NO GPU.

Left panel : outlet-attribution accuracy by feature set (tone-only, surface-only, surface+tone),
             pooled and within each category. Reads data/derived_features/tone_accuracy.csv.
Right panel: per-outlet z-score profile across the seven tone measures.
             Reads data/derived_features/tone_zscores.csv.

Backs the paper's §5.3 / Appendix D numbers: tone-only 0.13 (1.3x chance), surface-only 0.27,
surface+tone lift +0.004 pooled / +0.010 within-category.

Run from anywhere:
    python3 code/gen_tone_figure.py            # -> paper/figures/fig_tier1_tone.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent  # package root
DF = ROOT / "data" / "derived_features"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CHANCE = 1 / 10
# Okabe-Ito colorblind-safe
C_TONE, C_SURF, C_COMB = "#D55E00", "#0072B2", "#009E73"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

SCOPES = ["pooled", "International", "Politique", "Économie"]
FSETS = [("surface_only", "surface", C_SURF),
         ("tone_only", "tone", C_TONE),
         ("surface+tone", "surface+tone", C_COMB)]
MEASURES = ["polarity_model", "intensity_model", "polarity_lex", "subjectivity",
            "hedging_rate", "attribution_rate", "sensational_rate"]
MEAS_LBL = ["polarity\n(model)", "intensity", "polarity\n(lex)", "subjectivity",
            "hedging", "attribution", "sensational"]


def main():
    acc = pd.read_csv(DF / "tone_accuracy.csv")
    z = pd.read_csv(DF / "tone_zscores.csv").set_index("outlet").reindex(
        columns=MEASURES)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                   gridspec_kw={"width_ratios": [1.05, 1.25]})

    # ---- Left: grouped accuracy bars ----
    x = np.arange(len(SCOPES))
    w = 0.26
    for k, (fs, lab, c) in enumerate(FSETS):
        ys = [float(acc[(acc.scope == s) & (acc.feature_set == fs)]["acc"].iloc[0])
              for s in SCOPES]
        axL.bar(x + (k - 1) * w, ys, width=w, color=c, label=lab,
                edgecolor="white", linewidth=0.4, zorder=3)
    axL.axhline(CHANCE, ls="--", color="gray", lw=1)
    axL.text(len(SCOPES) - 0.5, CHANCE + 0.008, "chance", color="gray", fontsize=6.5, ha="right")
    axL.set_xticks(x)
    axL.set_xticklabels(["pooled", "Inter.", "Polit.", "Écon."])
    axL.set_ylabel("Accuracy (temporal)")
    axL.set_ylim(0, 0.40)
    axL.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.3)
    # annotate the pooled tone-only value (the paper's headline 0.13 number)
    t_pool = float(acc[(acc.scope == "pooled") & (acc.feature_set == "tone_only")]["acc"].iloc[0])
    axL.annotate(f"tone-only {t_pool:.2f}", (0, t_pool), textcoords="offset points",
                 xytext=(2, 4), fontsize=6.2, color=C_TONE)
    axL.set_title("Accuracy by feature set", fontsize=8)

    # ---- Right: per-outlet z-score heatmap ----
    order = z.mean(axis=1).sort_values().index  # stable, legible ordering
    M = z.reindex(order).values
    vmax = np.nanmax(np.abs(M))
    im = axR.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axR.set_xticks(range(len(MEASURES)))
    axR.set_xticklabels(MEAS_LBL, rotation=45, ha="right", fontsize=6)
    axR.set_yticks(range(len(order)))
    axR.set_yticklabels(order, fontsize=6.5)
    axR.set_title("Per-outlet tone z-scores", fontsize=8)
    cb = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_tier1_tone.pdf", bbox_inches="tight")
    plt.close(fig)
    surf = float(acc[(acc.scope == "pooled") & (acc.feature_set == "surface_only")]["acc"].iloc[0])
    comb = float(acc[(acc.scope == "pooled") & (acc.feature_set == "surface+tone")]["acc"].iloc[0])
    print(f"wrote fig_tier1_tone.pdf  (tone-only={t_pool:.3f}, surface={surf:.3f}, "
          f"surface+tone={comb:.3f}, pooled lift={comb - surf:+.3f})")


if __name__ == "__main__":
    main()
