"""
Topic-robustness evidence for "the signal is editorial style, not topic" (concern:
the within-category result uses an imperfect automated topic labeler, so residual topic could
leak). Two committed, reproducible checks from the persisted n10 runs -- no GPU, no retraining:

  (A) Category-only baseline. Because the corpus is balanced per outlet*category, the category
      label carries no marginal information about the outlet: a classifier given ONLY the
      category is at chance. We state this analytically and verify it from the sampled dataset
      (every category has equal outlet proportions). This rules out the COARSE topic confound by
      construction, independent of the topic labeler's accuracy.

  (B) Native within-category models. char-SVM and word-LR are trained AND tested inside a single
      category (not the pooled model sliced post-hoc). With topic held constant in *training*,
      the model cannot exploit cross-category topic->outlet associations; accuracy well above the
      within-category chance (1/10) is therefore stylistic, not topical. We read these directly
      from results/<panel>/runs/metrics (family == "within").

Usage:  python3 gen_topic_robustness.py n10
Writes analysis_<panel>/topic_robustness.csv and prints a summary.
"""
import sys, json, glob, warnings
from pathlib import Path
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent  # package root


def category_only_balance(panel):
    d = pd.read_parquet(ROOT / f"results/{panel}/sampled_dataset_{panel}.parquet")[["outlet", "category"]]
    # outlet share within each category; if balanced, max share ~ 1/n_outlets (= category-only chance)
    tab = (d.groupby("category", observed=True)["outlet"].value_counts(normalize=True)
           .rename("share").reset_index())
    worst = tab.groupby("category", observed=True)["share"].max()  # best a category-only guesser could do per cat
    n_out = d["outlet"].nunique()
    return n_out, worst


def within_category(panel):
    rows = []
    for f in glob.glob(str(ROOT / f"results/{panel}/runs/metrics/{panel}_*_within_*_headline_only_*.json")):
        m = json.load(open(f))
        if m.get("family") != "within":
            continue
        rows.append({k: m.get(k) for k in ("model", "category", "preprocess", "split", "accuracy", "macro_f1", "n")})
    return pd.DataFrame(rows)


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "n10"
    out = ROOT / f"analysis_{panel}"; out.mkdir(exist_ok=True)
    n_out, worst = category_only_balance(panel)
    chance = 1 / n_out
    print(f"== Topic robustness, panel {panel} (N={n_out}, within-category chance = {chance:.3f}) ==\n")
    print("(A) Category-only baseline (balanced design):")
    print(f"    best per-category outlet share = {worst.to_dict()}")
    print(f"    -> a category-only classifier is at {worst.mean():.3f} ~ chance {chance:.3f}: coarse topic carries no outlet signal.\n")

    wc = within_category(panel)
    if wc.empty:
        print("(B) No native within-category runs found.")
        return
    wc.to_csv(out / "topic_robustness.csv", index=False)
    print("(B) NATIVE within-category models (train+test inside one category; topic constant in training):")
    for split in ["temporal", "random"]:
        sub = wc[(wc.split == split)]
        if sub.empty:
            continue
        print(f"  -- {split} split --")
        piv = sub.pivot_table(index="model", columns=["preprocess"], values="accuracy")
        # order columns
        cols = [c for c in ["raw", "mask_outlet", "mask_entities"] if c in piv.columns]
        print(piv[cols].to_string(float_format=lambda x: f"{x:.3f}"))
        # ×chance for char_svm raw
        cs = sub[(sub.model == "char_svm") & (sub.preprocess == "raw")]
        if not cs.empty:
            print(f"     char-SVM raw: {cs.accuracy.mean():.3f} acc = {cs.accuracy.mean()/chance:.1f}x within-category chance")
    print(f"\nWrote {out}/topic_robustness.csv")


if __name__ == "__main__":
    main()
