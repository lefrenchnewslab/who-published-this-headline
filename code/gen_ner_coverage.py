"""
NER coverage audit for the editorial-fingerprinting corpus.

Quantifies "how much do the entity-masked numbers depend on NER quality?"
by measuring how many headlines / tokens the entity masker actually touches. It reuses the
EXACT masking logic from the Phase-1 training notebook (mask_outlet patterns from
OUTLET_DOMAINS, then spaCy fr_core_news_lg PER/LOC/ORG/MISC span replacement), so the numbers
characterize the masking the models actually saw — not a re-implementation.

Why this is a standalone script: the training pipeline applies mask_entities inline while
building text variants and never persists the masked text or any coverage statistic, so the
audit cannot be recomputed from the saved artifacts (which hold raw titles + predictions only).
This script regenerates it deterministically from the sampled dataset.

Usage (from the repository root ):
    python3 gen_ner_coverage.py n10                 # full corpus
    python3 gen_ner_coverage.py n15 --sample 20000  # faster estimate on a random sample
    python3 gen_ner_coverage.py n10 --model fr_core_news_sm   # override model (default: lg, matches pipeline)

Outputs (to analysis_<panel>/):
    ner_coverage.csv            overall + per-category coverage
    ner_coverage_by_outlet.csv  per-outlet coverage
and prints a paragraph-ready summary.

Requires: spacy + the chosen model. The pipeline uses fr_core_news_lg; install with
    python3 -m spacy download fr_core_news_lg
"""

import sys, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # package root

# Canonical outlet-mention dictionary — identical to the Phase-1 notebook config cell.
OUTLET_DOMAINS = {
    "Le Monde": ["lemonde", "le monde"], "Le Figaro": ["lefigaro", "le figaro"],
    "Les Echos": ["lesechos", "les echos", "les Échos"], "La Croix": ["la-croix", "la croix"],
    "Ouest-France": ["ouest-france", "ouest france"], "Le Parisien": ["leparisien", "le parisien"],
    "TF1 INFO": ["tf1info", "tf1 info", "tf1", "lci"], "Libération": ["liberation", "libération", "libe", "libé"],
    "Franceinfo": ["franceinfo", "france info", "francetvinfo"], "BFMTV": ["bfmtv", "bfm tv", "bfm"],
    "L'Humanité": ["l'humanité", "l humanite", "humanite", "huma"],
    "20 Minutes": ["20 minutes", "20minutes", "20 min"], "Le Point": ["le point", "lepoint"],
    "Le Nouvel Obs": ["nouvel obs", "nouvelobs", "l'obs", "le nouvel observateur"],
    "L'Express": ["l'express", "lexpress", "express"], "Le HuffPost": ["huffpost", "huffington", "le huffpost"],
    "RFI": ["rfi", "radio france internationale"], "CNEWS": ["cnews", "c news", "cnews.fr"],
}
_pat = [re.escape(v) for variants in OUTLET_DOMAINS.values() for v in variants]
OUTLET_RE = re.compile(r"(?i)\b(" + "|".join(sorted(_pat, key=len, reverse=True)) + r")\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
ENT_LABELS = {"PER", "LOC", "ORG", "MISC"}


def mask_outlet(text):
    text = URL_RE.sub("[URL]", text)
    return OUTLET_RE.sub("[OUTLET]", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--model", default="fr_core_news_lg", help="spaCy model (pipeline uses fr_core_news_lg)")
    ap.add_argument("--sample", type=int, default=0, help="random-sample N headlines (0 = full corpus)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    samp = ROOT / f"results/{a.panel}/sampled_dataset_{a.panel}.parquet"
    if not samp.exists():
        sys.exit(f"ERROR: {samp} not found.")
    df = pd.read_parquet(samp)[["title", "category", "outlet"]].dropna(subset=["title"])
    if a.sample and a.sample < len(df):
        df = df.sample(a.sample, random_state=a.seed).reset_index(drop=True)

    try:
        import spacy
    except ImportError:
        sys.exit("ERROR: spaCy not installed. pip install spacy")
    try:
        nlp = spacy.load(a.model, disable=["morphologizer", "parser", "attribute_ruler", "lemmatizer"])
    except OSError:
        sys.exit(f"ERROR: spaCy model '{a.model}' not installed. "
                 f"Run: python3 -m spacy download {a.model}")

    masked_outlet = df["title"].map(mask_outlet)
    n_ent = np.zeros(len(df), dtype=int)
    masked_chars = np.zeros(len(df), dtype=int)
    title_chars = masked_outlet.str.len().to_numpy()
    for i, doc in enumerate(nlp.pipe(masked_outlet.tolist(), batch_size=256)):
        ents = [e for e in doc.ents if e.label_ in ENT_LABELS]
        n_ent[i] = len(ents)
        masked_chars[i] = sum(e.end_char - e.start_char for e in ents)

    df = df.assign(n_ent=n_ent, has_ent=(n_ent > 0),
                   frac_chars_masked=np.where(title_chars > 0, masked_chars / title_chars, 0.0))

    def summarize(g):
        return pd.Series({
            "n_headlines": len(g),
            "pct_with_entity": round(100 * g["has_ent"].mean(), 1),
            "mean_entities_per_headline": round(g["n_ent"].mean(), 3),
            "pct_chars_masked": round(100 * g["frac_chars_masked"].mean(), 1),
        })

    out = ROOT / f"analysis_{a.panel}"; out.mkdir(exist_ok=True)
    overall = summarize(df); overall.name = "ALL"
    by_cat = df.groupby("category", observed=True).apply(summarize)
    pd.concat([overall.to_frame().T, by_cat]).to_csv(out / "ner_coverage.csv")
    df.groupby("outlet", observed=True).apply(summarize).sort_values(
        "pct_with_entity", ascending=False).to_csv(out / "ner_coverage_by_outlet.csv")

    print(f"== NER coverage, panel {a.panel}, model {a.model} "
          f"({'sample '+str(a.sample) if a.sample else 'full corpus'}, n={len(df):,}) ==")
    print(f"  headlines with >=1 masked entity : {overall['pct_with_entity']}%")
    print(f"  mean entities masked per headline: {overall['mean_entities_per_headline']}")
    print(f"  mean % of characters masked      : {overall['pct_chars_masked']}%")
    print("  by category:")
    print(by_cat.to_string())
    print(f"\nWrote {out}/ner_coverage.csv and ner_coverage_by_outlet.csv")


if __name__ == "__main__":
    main()
