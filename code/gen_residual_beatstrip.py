#!/usr/bin/env python3
"""
Residual-error analysis: how much of the lexical residual is outlet BEAT vocabulary?

Motivation: the 43% "lexical residual" (word choice surviving surface-stripping
and NER masking) may still be topic/beat signal, not style -- its distinctive words
include beat nouns (evêques -> La Croix, cryptos -> Les Echos) that spaCy NER never
removes because they are common nouns, not entities.

This runs the requested test. Starting from the surface-stripped residual text (the
~2.7x-chance word-level residual), we rank outlet-associated CONTENT words by weighted
log-odds with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008,
"Fightin' Words"), computed on TRAIN only, then additionally strip the top-k such words
per outlet and re-fit the paper's word-LR. Accuracy is reported as a function of k, so
we can see how much of the residual's above-chance margin this tested content-word
blocklist closes and whether a lexical residual remains after those words are removed.

Interpretation:
  - if accuracy collapses toward chance as k grows -> most of the residual is beat;
    the paper's style claim rests on the matched-event analysis (Sec 5.4), not this.
  - if a floor well above chance persists after stripping hundreds of ranked words per
    outlet -> lexical signal survives this intervention, without proving it is non-topical.

Reuses the paper's exact split, stripping, and word-LR (gen_residual_decomposition.py).
No new data; needs only sampled_dataset_<panel>.parquet.

Run:    python3 gen_residual_beatstrip.py n10
Writes: analysis_<panel>/residual_beatstrip_curve.csv
        analysis_<panel>/residual_beatstrip_blocklist.csv   (the stripped beat words)
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from split_utils import temporal_split
from gen_residual_decomposition import surface_strip  # identical stripping pipeline

ROOT = Path(__file__).resolve().parent.parent  # package root
K_GRID = [0, 25, 50, 100, 200, 400]   # top-k beat words PER OUTLET
TOKEN = re.compile(r"[a-zà-ÿ]{2,}")   # residual text is already lowercased/de-punctuated

# Minimal French stopword set: we strip outlet-associated *content* words (the
# ask), so function words are excluded from the blocklist and remain in the floor.
STOP = set("""
le la les un une des du de d au aux a à et ou mais donc or ni car que qui quoi dont où
ce cet cette ces son sa ses leur leurs mon ma mes ton ta tes notre nos votre vos
je tu il elle on nous vous ils elles se sa lui y en
ne pas plus moins tres si non oui pour par sur sous dans entre vers avec sans chez
est sont etait etaient sera seront ete etre avoir a ont avait avaient
comme quand alors aussi apres avant depuis pendant contre selon
son ses leur au aux ce il elle qui que
""".split())


def weighted_logodds(train_txt: pd.Series, labels: pd.Series, outlets: list[str],
                     alpha: float = 0.01) -> dict[str, list[str]]:
    """Top outlet-distinctive CONTENT words per outlet via Monroe et al. (2008)
    weighted log-odds with an informative Dirichlet prior. Train only."""
    # per-outlet content-word counts
    counts: dict[str, Counter] = {o: Counter() for o in outlets}
    for txt, o in zip(train_txt, labels):
        for w in TOKEN.findall(txt):
            if w not in STOP:
                counts[o][w] += 1
    vocab = sorted(set().union(*[set(c) for c in counts.values()]))
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    mat = np.zeros((len(outlets), V))
    for r, o in enumerate(outlets):
        for w, n in counts[o].items():
            mat[r, idx[w]] = n
    total = mat.sum(0)                     # background counts
    a0 = alpha * V
    ranked: dict[str, list[str]] = {}
    for r, o in enumerate(outlets):
        yi = mat[r]; ni = yi.sum()
        yrest = total - yi; nrest = yrest.sum()
        # log-odds-ratio with informative Dirichlet prior (prior = corpus frequency)
        num = (yi + alpha) / (ni + a0 - (yi + alpha))
        den = (yrest + alpha) / (nrest + a0 - (yrest + alpha))
        delta = np.log(num) - np.log(den)
        var = 1.0 / (yi + alpha) + 1.0 / (yrest + alpha)
        z = delta / np.sqrt(var)           # z-scored: distinctive AND reliable
        order = np.argsort(z)[::-1]
        ranked[o] = [vocab[i] for i in order if z[i] > 0][:max(K_GRID)]
    return ranked


def strip_words(txt: str, block: set[str]) -> str:
    return " ".join(w for w in TOKEN.findall(txt) if w not in block)


def word_lr(train_txt, y_train, test_txt, y_test, labels):
    """The paper's residual word-LR (TF-IDF word 1-2gram, multinomial LR, C=4)."""
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=60000,
                          sublinear_tf=True, min_df=3)
    Xtr = vec.fit_transform(train_txt)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(Xtr, y_train)
    pred = clf.predict(vec.transform(test_txt))
    return (accuracy_score(y_test, pred),
            f1_score(y_test, pred, labels=labels, average="macro", zero_division=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--alpha", type=float, default=0.01, help="Dirichlet prior for log-odds")
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / f"results/{args.panel}/sampled_dataset_{args.panel}.parquet")[
        ["title", "outlet", "year", "dt"]].dropna(subset=["title"])
    tr, te = temporal_split(df)
    labels = sorted(df["outlet"].unique()); chance = 1.0 / len(labels)
    out = ROOT / f"analysis_{args.panel}"; out.mkdir(exist_ok=True)

    # residual text = surface-stripped (lowercase, de-punctuated, de-digited)
    tr_res = tr["title"].map(lambda s: surface_strip(s, drop_proper=False))
    te_res = te["title"].map(lambda s: surface_strip(s, drop_proper=False))

    ranked = weighted_logodds(tr_res, tr["outlet"].to_numpy(), labels, alpha=args.alpha)
    # save the full top-k blocklist per outlet for transparency
    pd.DataFrame([{"outlet": o, "rank": i + 1, "word": w}
                  for o in labels for i, w in enumerate(ranked[o])]
                 ).to_csv(out / "residual_beatstrip_blocklist.csv", index=False)

    print(f"panel {args.panel}: train={len(tr):,} test={len(te):,} "
          f"outlets={len(labels)} chance={chance:.3f}", flush=True)
    print("sample beat words (top-8/outlet):", flush=True)
    for o in labels:
        print(f"  {o:14s} {', '.join(ranked[o][:8])}", flush=True)

    rows = []
    base_margin = None
    for k in K_GRID:
        block = set() if k == 0 else set().union(*[set(ranked[o][:k]) for o in labels])
        trx = tr_res.map(lambda s: strip_words(s, block)) if block else tr_res
        tex = te_res.map(lambda s: strip_words(s, block)) if block else te_res
        acc, f1 = word_lr(trx, tr["outlet"], tex, te["outlet"], labels)
        margin = acc - chance
        if k == 0:
            base_margin = margin
        closed = 100 * (1 - margin / base_margin) if base_margin else 0.0
        rows.append({"k_per_outlet": k, "n_words_stripped": len(block),
                     "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                     "x_chance": round(acc / chance, 2),
                     "pct_residual_margin_closed": round(closed, 1)})
        pd.DataFrame(rows).to_csv(out / "residual_beatstrip_curve.csv", index=False)
        print(f"  k={k:<4} strip={len(block):<5} acc={acc:.4f} {acc/chance:.2f}x "
              f"margin_closed={closed:.1f}%", flush=True)

    floor = rows[-1]
    print("\n=== READ ===", flush=True)
    print(f"residual (k=0): {rows[0]['accuracy']:.3f} ({rows[0]['x_chance']}x chance)", flush=True)
    print(f"after stripping top-{K_GRID[-1]} beat words/outlet "
          f"({floor['n_words_stripped']} words): {floor['accuracy']:.3f} "
          f"({floor['x_chance']}x); {floor['pct_residual_margin_closed']}% of the residual "
          f"margin is closed by the tested blocklist; residual topic/beat signal may remain.", flush=True)
    print(f"\nwrote {out/'residual_beatstrip_curve.csv'}", flush=True)
    print(f"wrote {out/'residual_beatstrip_blocklist.csv'}", flush=True)


if __name__ == "__main__":
    main()
