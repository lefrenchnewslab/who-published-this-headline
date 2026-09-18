"""
Bootstrap CIs for the 44/14/43 editorial-fingerprint decomposition.
the three-way partition of the
char-SVM above-chance margin into presentation / named-entities / lexical-residual
is reported as a single point estimate; this script attaches 95% CIs.

WHAT IT DOES
------------
Reuses the EXACT char-SVM stripping pipeline from gen_residual_decomposition.py
(same TF-IDF char_wb 3-5, LinearSVC balanced, same canonical temporal split), fits
the three decomposition rungs ONCE on train, predicts on the 2025 test year, and
caches a per-headline correctness vector for each rung:

  raw                        accuracy a_raw   (~0.4965)
  surface_stripped           accuracy a_surf  (~0.3240)  [lowercase, drop punct/digits/ws]
  surface_stripped_nermask   accuracy a_ent   (~0.2703)  [+ spaCy PER/LOC/ORG/MISC removed]

Decomposition of the above-chance margin (chance c = 1/n_outlets = 0.10), per the paper:

  presentation = (a_raw  - a_surf) / (a_raw - c)     ~44%
  entities     = (a_surf - a_ent ) / (a_raw - c)     ~14%
  residual     = (a_ent  - c     ) / (a_raw - c)     ~43%

Then it draws B bootstrap resamples OF THE TEST SET (
"bootstrap ... over 1,000 resamples of the test set"), recomputing the three
accuracies and the three proportions on each resample, and reports percentile 95% CIs.
Models are fit once and held fixed; only the test set is resampled. This isolates
sampling uncertainty of the test estimate, which is what the headline percentages need.

  --refit-each   (optional, slow) also refit every model on a bootstrap of the
                 *training* set per replicate, to additionally fold in estimator
                 variance. Off by default (B refits of a 200k-feature SVM is hours).

USAGE
-----
  python3 gen_decomposition_bootstrap.py n10 --spacy fr_core_news_lg
  python3 gen_decomposition_bootstrap.py n10 --spacy fr_core_news_lg --bootstrap 1000 --seed 0

  # quick smoke test without spaCy (uses the capitalized-token proxy for the entity rung;
  # gives a slightly different point estimate than the paper's NER rung, for plumbing checks only):
  python3 gen_decomposition_bootstrap.py n10 --entity-proxy

OUTPUT
------
  analysis_<panel>/decomposition_bootstrap.csv   one row per component:
        component, point_pct, ci_lo_pct, ci_hi_pct, plus the raw accuracies + CIs
  Prints a paste-ready line, e.g.:
        presentation 44.0% [41.2, 46.7] | entities 13.5% [10.9, 16.1] | residual 42.9% [40.0, 45.8]

Requires: the same env that runs gen_residual_decomposition.py (sklearn, pandas,
pyarrow, and — for the real entity rung — spaCy + fr_core_news_lg). No GPU needed.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the paper's canonical split and the EXACT stripping functions, so every
# point estimate here is identical to gen_residual_decomposition.py by construction.
from split_utils import temporal_split
from gen_residual_decomposition import (
    surface_strip,
    ner_mask_then_strip,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent  # package root


def fit_predict_correct(train_txt, y_train, test_txt, y_test, max_features, ngram_max):
    """Fit the paper's char-SVM once; return a boolean per-test-headline correctness vector."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, ngram_max),
                          max_features=max_features, sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(train_txt)
    clf = LinearSVC(class_weight="balanced")
    clf.fit(Xtr, y_train)
    pred = clf.predict(vec.transform(test_txt))
    return (np.asarray(pred) == np.asarray(y_test)).astype(np.float64)


def proportions(a_raw, a_surf, a_ent, chance):
    """The three-way partition of the above-chance margin. Returns (pres, ent, res) as fractions."""
    margin = a_raw - chance
    pres = (a_raw - a_surf) / margin
    ent = (a_surf - a_ent) / margin
    res = (a_ent - chance) / margin
    return pres, ent, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spacy", default=None,
                    help="spaCy model for the real NER entity rung (e.g. fr_core_news_lg). "
                         "This reproduces the paper's surface_stripped_nermask = 0.2703 rung.")
    ap.add_argument("--entity-proxy", action="store_true",
                    help="Plumbing-only: use the capitalized-token proxy instead of spaCy for the "
                         "entity rung. Faster but NOT the paper's NER rung; for smoke tests.")
    ap.add_argument("--refit-each", action="store_true",
                    help="Also refit models on a train bootstrap per replicate (slow; folds in "
                         "estimator variance on top of test-sampling variance).")
    a = ap.parse_args()

    if not a.spacy and not a.entity_proxy:
        sys.exit("ERROR: pass --spacy fr_core_news_lg (paper-faithful entity rung) "
                 "or --entity-proxy (smoke test only).")

    df = pd.read_parquet(ROOT / f"results/{a.panel}/sampled_dataset_{a.panel}.parquet")[
        ["title", "outlet", "year", "dt"]].dropna(subset=["title"])
    tr, te = temporal_split(df)
    labels = sorted(df["outlet"].unique())
    chance = 1.0 / len(labels)
    y_test = te["outlet"].to_numpy()
    n = len(te)
    out = ROOT / f"analysis_{a.panel}"
    out.mkdir(exist_ok=True)
    print(f"panel {a.panel}: train={len(tr):,} test={n:,} outlets={len(labels)} "
          f"chance={chance:.3f} B={a.bootstrap}", flush=True)

    # --- prepare the three text variants for both train and test ---
    def variants(series):
        raw = series.astype(str)
        surf = series.map(lambda s: surface_strip(s, drop_proper=False))
        if a.spacy:
            ent = ner_mask_then_strip(series, a.spacy)        # spaCy NER mask -> surface strip
        else:
            ent = series.map(lambda s: surface_strip(s, drop_proper=True))  # capitalized-token proxy
        return raw, surf, list(ent)

    print("  preparing text variants (raw / surface_stripped / entity-stripped) ...", flush=True)
    tr_raw, tr_surf, tr_ent = variants(tr["title"])
    te_raw, te_surf, te_ent = variants(te["title"])

    # --- fit each rung ONCE, cache per-headline correctness on the test set ---
    print("  fitting char-SVM rungs (raw, surface_stripped, entity-stripped) ...", flush=True)
    c_raw = fit_predict_correct(tr_raw, tr["outlet"], te_raw, y_test, a.max_features, a.ngram_max)
    c_surf = fit_predict_correct(tr_surf, tr["outlet"], te_surf, y_test, a.max_features, a.ngram_max)
    c_ent = fit_predict_correct(tr_ent, tr["outlet"], te_ent, y_test, a.max_features, a.ngram_max)

    a_raw, a_surf, a_ent = c_raw.mean(), c_surf.mean(), c_ent.mean()
    p_pres, p_ent, p_res = proportions(a_raw, a_surf, a_ent, chance)
    print(f"  point: a_raw={a_raw:.4f} a_surf={a_surf:.4f} a_ent={a_ent:.4f}  "
          f"margin={a_raw - chance:.4f}", flush=True)
    print(f"  point: presentation {100*p_pres:.1f}% | entities {100*p_ent:.1f}% | "
          f"residual {100*p_res:.1f}%", flush=True)

    # --- bootstrap the TEST SET ---
    rng = np.random.default_rng(a.seed)
    boot = {"a_raw": [], "a_surf": [], "a_ent": [], "pres": [], "ent": [], "res": []}
    for b in range(a.bootstrap):
        idx = rng.integers(0, n, n)  # resample test headlines with replacement
        br, bs, be = c_raw[idx].mean(), c_surf[idx].mean(), c_ent[idx].mean()
        pr, en, re_ = proportions(br, bs, be, chance)
        boot["a_raw"].append(br); boot["a_surf"].append(bs); boot["a_ent"].append(be)
        boot["pres"].append(pr); boot["ent"].append(en); boot["res"].append(re_)
        if (b + 1) % 200 == 0:
            print(f"    bootstrap {b + 1}/{a.bootstrap}", flush=True)

    def ci(key):
        arr = np.array(boot[key])
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    rows = []
    for comp, point, key in [("presentation", p_pres, "pres"),
                             ("entities", p_ent, "ent"),
                             ("lexical_residual", p_res, "res")]:
        lo, hi = ci(key)
        rows.append({"component": comp,
                     "point_pct": round(100 * point, 1),
                     "ci_lo_pct": round(100 * lo, 1),
                     "ci_hi_pct": round(100 * hi, 1)})
    # also persist the underlying accuracy CIs for the appendix / transparency
    acc_rows = []
    for nm, pt, key in [("acc_raw", a_raw, "a_raw"), ("acc_surface_stripped", a_surf, "a_surf"),
                        ("acc_entity_stripped", a_ent, "a_ent")]:
        lo, hi = ci(key)
        acc_rows.append({"component": nm, "point_pct": round(pt, 4),
                         "ci_lo_pct": round(lo, 4), "ci_hi_pct": round(hi, 4)})

    res_df = pd.DataFrame(rows + acc_rows)
    res_df.to_csv(out / "decomposition_bootstrap.csv", index=False)

    pr, en, re_ = rows
    print("\n=== PASTE-READY (95% CI, percentile bootstrap, B={}) ===".format(a.bootstrap))
    print(f"presentation {pr['point_pct']}% [{pr['ci_lo_pct']}, {pr['ci_hi_pct']}] | "
          f"entities {en['point_pct']}% [{en['ci_lo_pct']}, {en['ci_hi_pct']}] | "
          f"residual {re_['point_pct']}% [{re_['ci_lo_pct']}, {re_['ci_hi_pct']}]")
    print(f"\nWrote {out}/decomposition_bootstrap.csv")
    print("DONE")


if __name__ == "__main__":
    main()
