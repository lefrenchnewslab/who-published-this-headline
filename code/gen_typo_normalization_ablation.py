"""
Typography-normalization ablation : does the editorial fingerprint survive
normalizing the surface typographic markers that dominate the char-SVM feature table?

We normalize ONLY the markers Table 5 highlights — guillemets/curly quotes -> straight,
curly apostrophes -> straight, ellipsis/dashes -> ASCII, and strip trailing terminal
punctuation — while leaving casing, word choice, and internal structure untouched. We then
re-fit the SAME char-SVM (TF-IDF char_wb 3-5, LinearSVC, balanced) on the temporal split and
compare to raw. If accuracy largely survives, the fingerprint is not reducible to a handful
of typographic tokens (deeper character-level style remains); if it collapses, the signal is
mostly a formatting mandate.

Committed, reproducible. Run from the repository root :
    python3 gen_typo_normalization_ablation.py n10
Writes analysis_<panel>/typo_normalization_ablation.csv and prints the delta.
"""

import sys, re, argparse
from pathlib import Path
from split_utils import temporal_split  # canonical paper split (carves 10% late-2024 val)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent  # package root
TRAIN_MAX_YEAR, TEST_YEAR = 2024, 2025  # temporal split: train <=2024, test 2025

# Typographic normalization (markers named in Table 5; casing/words left intact).
_QUOTES = {"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‟": '"', "‹": '"', "›": '"'}
_APOS = {"’": "'", "‘": "'", "‛": "'", "ʼ": "'", "´": "'", "`": "'"}
_DASH = {"–": "-", "—": "-", "―": "-"}
_TRANS = {**_QUOTES, **_APOS, **_DASH, "…": "..."}
_TRAIL = re.compile(r"[\s\.\!\?:;,\|…-]+$")

def normalize_typo(t: str) -> str:
    for k, v in _TRANS.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t).strip()
    return _TRAIL.sub("", t)


def fit_eval(train_txt, y_train, test_txt, y_test, labels, max_features=200000, ngram_max=5):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, ngram_max),
                          max_features=max_features, sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(train_txt)
    clf = LinearSVC(class_weight="balanced")
    clf.fit(Xtr, y_train)
    pred = clf.predict(vec.transform(test_txt))
    return accuracy_score(y_test, pred), f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--ngram-max", type=int, default=5)
    a = ap.parse_args()
    samp = ROOT / f"results/{a.panel}/sampled_dataset_{a.panel}.parquet"
    if not samp.exists():
        sys.exit(f"ERROR: {samp} not found.")
    df = pd.read_parquet(samp)[["title", "outlet", "year", "dt"]].dropna(subset=["title"])
    tr, te = temporal_split(df)  # canonical paper split (carves 10% late-2024 val out of train)
    labels = sorted(df["outlet"].unique())
    print(f"panel {a.panel}: train={len(tr):,} (<= {TRAIN_MAX_YEAR})  test={len(te):,} ({TEST_YEAR})  outlets={len(labels)}")

    rows = []
    for cond, fn in [("raw", lambda s: s), ("typo_normalized", normalize_typo)]:
        acc, f1 = fit_eval(tr["title"].map(fn), tr["outlet"], te["title"].map(fn), te["outlet"], labels,
                           max_features=a.max_features, ngram_max=a.ngram_max)
        rows.append({"condition": cond, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        print(f"  char-SVM {cond:16s}: acc={acc:.4f}  macroF1={f1:.4f}")

    d_acc = rows[0]["accuracy"] - rows[1]["accuracy"]
    d_f1 = rows[0]["macro_f1"] - rows[1]["macro_f1"]
    out = ROOT / f"analysis_{a.panel}"; out.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.loc[len(res)] = {"condition": "delta(raw-norm)", "accuracy": round(d_acc, 4), "macro_f1": round(d_f1, 4)}
    res.to_csv(out / "typo_normalization_ablation.csv", index=False)
    pct = 100 * d_acc / rows[0]["accuracy"]
    print(f"  -> normalization costs {d_acc:.4f} acc ({pct:.1f}% relative); {1-d_acc/rows[0]['accuracy']:.1%} of char-SVM accuracy survives.")
    print(f"Wrote {out}/typo_normalization_ablation.csv")


if __name__ == "__main__":
    main()
