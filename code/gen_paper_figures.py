"""
Authoritative paper-figure generator (N=10 panel). Reproducible from committed CSVs.

Outputs vector PDFs into paper/figures/:
  fig_accuracy_ladder.pdf  - model ladder, temporal split, 3 text conditions, PLUS the
                             zero-shot LLM baseline shown as a separate left-hand cluster.
  fig_per_outlet_f1.pdf    - per-outlet F1 (CamemBERT, temporal, entity-masked), sorted.

Data sources (all committed, no GPU needed):
  data/derived_features/table2_main_results.csv        (the model ladder numbers)
  data/derived_features/table3_per_outlet_temporal.csv (per-outlet F1)
  llm/llm_baseline_aggregate.json                       (LLM zero-shot cluster; from gen_llm_baseline.py)

The LLM zero-shot points come from the unified harness (gen_llm_baseline.py); every
model with a 0-shot cell is plotted as one grey cluster (per-model detail lives in the
appendix table). A legacy llm_zeroshot_*.csv fallback is kept for older runs.

Run from the repository root :
  pip install pandas matplotlib
  python3 gen_paper_figures.py
"""
from pathlib import Path
import glob, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent  # package root
N10 = ROOT / "data" / "derived_features"
LLM_AGG = ROOT / "llm" / "llm_baseline_aggregate.json"
LLM_GLOB = str(ROOT / "llm" / "llm_zeroshot_*.csv")  # legacy fallback
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Okabe-Ito colorblind-safe
C_RAW, C_MOUT, C_MENT, C_LLM = "#0072B2", "#009E73", "#D55E00", "#444444"
CHANCE = 1 / 10

# trained ladder, low -> high capacity
TRAINED = [("word_lr", "Word-LR"), ("char_svm", "Char-SVM"),
           ("charcnn_word", "CharCNN"), ("camembert", "CamemBERT")]


def _acc(df, model, pp):
    """temporal accuracy for (model, preprocess), falling back to 'none'."""
    sub = df[(df.model == model) & (df.preprocess == pp) & (df.split == "temporal")]
    if sub.empty:
        sub = df[(df.model == model) & (df.preprocess == "none") & (df.split == "temporal")]
    return float(sub["acc_mean"].iloc[0])


def _llm_results():
    """[(label, acc), ...] zero-shot accuracy per model, ascending.

    Primary source: the unified harness aggregate (all 0-shot cells, mean over seeds).
    Fallback: legacy llm_zeroshot_*.csv, kept so older runs still plot.
    """
    if LLM_AGG.exists():
        import json
        data = json.loads(LLM_AGG.read_text())
        out = [(str(r.get("tag") or r.get("model") or "LLM"), float(r["mean_accuracy"]))
               for r in data if int(r["shots"]) == 0]
        if out:
            return sorted(out, key=lambda t: t[1])
    out = []
    for f in glob.glob(LLM_GLOB):
        m = re.search(r"llm_zeroshot_(.+)\.csv$", Path(f).name)
        name = m.group(1) if m else Path(f).stem
        if name in ("baseline",):
            continue
        d = pd.read_csv(f)
        out.append((f"LLM {name.replace('gpt-oss-', '').replace('-', ' ')}",
                    float((d["gold"] == d["pred"]).mean())))
    return sorted(out, key=lambda t: t[1])


def accuracy_ladder():
    df = pd.read_csv(N10 / "table2_main_results.csv")
    maj = _acc(df, "majority", "none")
    llm = _llm_results()

    # XLM-R: multilingual comparator, primary cell only (temporal, entity-masked)
    xlmr_ent = _acc(df, "xlmr", "mask_entities")

    # x layout: [Majority] [LLM cluster] | [trained ladder x4] [XLM-R]
    x_maj = 0
    x_llm = 1
    x_trained = list(range(2, 2 + len(TRAINED)))
    x_xlmr = x_trained[-1] + 1
    fig, ax = plt.subplots(figsize=(3.7, 2.5))

    # baselines (left of the separator), shown as standalone markers
    ax.scatter([x_maj], [maj], marker="x", color=C_LLM, s=22, zorder=3)
    # LLM zero-shot: plot every model as one grey cluster (all sit ~2x chance);
    # per-model numbers live in the appendix table, so annotate the cluster's range only.
    accs = [a for _, a in llm]
    ax.scatter([x_llm] * len(accs), accs, marker="o", facecolor="white",
               edgecolor=C_LLM, color=C_LLM, s=26, zorder=3,
               label=f"LLM 0-shot ({len(accs)} models)")
    if accs:
        lo, hi = min(accs), max(accs)
        rng = f"{lo:.2f}" if hi - lo < 0.005 else f"{lo:.2f}–{hi:.2f}"
        ax.annotate(rng, (x_llm, hi), textcoords="offset points",
                    xytext=(7, -1), fontsize=6.3, color=C_LLM)

    # separator between untrained baselines and the trained ladder
    ax.axvline(x_llm + 0.5, ls=":", color="0.6", lw=0.9)

    # trained ladder: 3 condition lines
    for pp, c, lab, mk in [("raw", C_RAW, "raw", "o"),
                           ("mask_outlet", C_MOUT, "mask-outlet", "s"),
                           ("mask_entities", C_MENT, "mask-entities", "^")]:
        ys = [_acc(df, m, pp) for m, _ in TRAINED]
        ax.plot(x_trained, ys, marker=mk, color=c, label=lab, linewidth=1.6, markersize=4.3)

    # XLM-R comparator: standalone star at its entity-masked accuracy, color-linked
    # to the mask-entities condition; a dotted tie shows it tracking CamemBERT's m-ent cell.
    cam_ent_x = x_trained[-1]
    ax.plot([cam_ent_x, x_xlmr], [_acc(df, "camembert", "mask_entities"), xlmr_ent],
            ls=":", color=C_MENT, lw=0.9, zorder=2)
    ax.scatter([x_xlmr], [xlmr_ent], marker="*", s=70, facecolor="white",
               edgecolor=C_MENT, linewidth=1.1, zorder=4, label="XLM-R (m-ent)")
    ax.annotate(f"{xlmr_ent:.2f}", (x_xlmr, xlmr_ent), textcoords="offset points",
                xytext=(6, -2), fontsize=6.3, color=C_MENT)

    ax.axhline(CHANCE, ls="--", color="gray", lw=1)
    ax.text(0.01, CHANCE + 0.012, "chance", color="gray", fontsize=6.5,
            transform=ax.get_yaxis_transform())

    labels = ["Majority", "LLM\n0-shot"] + [n for _, n in TRAINED] + ["XLM-R"]
    ax.set_xticks([x_maj, x_llm] + x_trained + [x_xlmr])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (temporal)")
    ax.set_ylim(0, 0.68)
    ax.set_xlim(-0.5, x_xlmr + 0.5)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_accuracy_ladder.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_accuracy_ladder.pdf  (majority={maj:.3f}, "
          f"LLM={[(l, round(a,3)) for l,a in llm]}, xlmr_ent={xlmr_ent:.3f})")


def per_outlet_f1():
    df = pd.read_csv(N10 / "table3_per_outlet_temporal.csv").sort_values("f1")
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    bars = ax.barh(df["outlet"], df["f1"], color="#4C72B0", height=0.7)
    ax.axvline(CHANCE, ls="--", color="gray", lw=1)
    ax.text(CHANCE + 0.005, 0.2, "chance", color="gray", fontsize=6.5, rotation=90, va="bottom")
    for b, v in zip(bars, df["f1"]):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set_xlabel("F1 (temporal, entity-masked)")
    ax.set_xlim(0, 0.92)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_per_outlet_f1.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_per_outlet_f1.pdf")


def beatstrip_curve():
    """Residual accuracy as outlet-distinctive content words are stripped
    (is the lexical residual just beat vocabulary?). Reads the ablation curve CSV."""
    f = ROOT / "data" / "derived_features" / "residual_beatstrip_curve.csv"
    if not f.exists():
        print("skip beatstrip_curve (run gen_residual_beatstrip.py first)")
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(d["n_words_stripped"], d["accuracy"], marker="o", color="#0072B2", lw=1.7, ms=4.2)
    ax.axhline(CHANCE, ls="--", color="gray", lw=1)
    ax.text(d["n_words_stripped"].max() * 0.55, CHANCE + 0.006, "chance", color="gray", fontsize=6.5)
    ax.annotate(f"residual {d['accuracy'].iloc[0]:.2f} ({d['x_chance'].iloc[0]:.1f}$\\times$)",
                (d["n_words_stripped"].iloc[0], d["accuracy"].iloc[0]),
                textcoords="offset points", xytext=(8, 3), fontsize=6.3)
    ax.annotate(f"floor {d['accuracy'].iloc[-1]:.2f} ({d['x_chance'].iloc[-1]:.1f}$\\times$)\n"
                f"after {int(d['n_words_stripped'].iloc[-1]):,} words",
                (d["n_words_stripped"].iloc[-1], d["accuracy"].iloc[-1]),
                textcoords="offset points", xytext=(-4, 12), fontsize=6.3, ha="right")
    ax.set_xlabel("outlet-distinctive content words stripped")
    ax.set_ylabel("residual accuracy (temporal)")
    ax.set_ylim(0.08, 0.35)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_beatstrip.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_beatstrip.pdf  (residual={d['accuracy'].iloc[0]:.3f} -> "
          f"floor={d['accuracy'].iloc[-1]:.3f}, {d['pct_residual_margin_closed'].iloc[-1]}% closed)")


if __name__ == "__main__":
    accuracy_ladder()
    per_outlet_f1()
    beatstrip_curve()
    print("done ->", OUT)
