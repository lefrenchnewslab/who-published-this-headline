"""
NER recall hand-check packet .

Coverage (gen_ner_coverage.py) tells us how often the masker fires; it does NOT tell us
RECALL against gold entities. This script builds a small, stratified annotation packet so a
human can mark the true entities and we can report spaCy fr_core_news_lg recall on headlines.

Two modes:
  build  : sample N headlines (stratified by outlet), run spaCy lg, write a packet CSV with
           the model's detected spans and an empty `gold_entities` column for a human to fill
           (list the PER/LOC/ORG/MISC surface strings actually present, ';'-separated).
  score  : after the packet is annotated, compute span-level recall = (gold spans the model
           caught) / (all gold spans), plus headline-level recall.

Usage (from the repository root ):
  python3 gen_ner_recall_packet.py build n10 --n 120
  # ... a human fills the gold_entities column in analysis_n10/ner_recall_packet.csv ...
  python3 gen_ner_recall_packet.py score n10
"""
import sys, re, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # package root
OUTLET_DOMAINS_PAT = None  # built lazily from gen_ner_coverage if needed

def _mask_outlet_factory():
    from gen_ner_coverage import mask_outlet  # reuse identical masking
    return mask_outlet


def build(panel, n, seed):
    # Sampling only -- no spaCy needed here, so the packet can be produced anywhere.
    # spaCy fr_core_news_lg is run at SCORE time (where the model is installed).
    samp = ROOT / f"results/{panel}/sampled_dataset_{panel}.parquet"
    df = pd.read_parquet(samp)[["title", "outlet", "category"]].dropna(subset=["title"])
    per = max(1, n // df["outlet"].nunique())  # stratified by outlet for register coverage
    sample = (df.groupby("outlet", group_keys=False)
                .apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
                .sample(frac=1, random_state=seed).reset_index(drop=True))  # shuffle
    rows = [{"idx": i, "outlet": r["outlet"], "category": r["category"], "headline": r["title"],
             "gold_entities": "", "notes": ""} for i, (_, r) in enumerate(sample.iterrows())]
    out = ROOT / f"analysis_{panel}"; out.mkdir(exist_ok=True)
    p = out / "ner_recall_packet.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"Wrote {p} with {len(rows)} headlines.\n"
          f"ANNOTATE: in `gold_entities`, list the PER/LOC/ORG/MISC surface strings actually present in the\n"
          f"headline, ';'-separated (leave blank if none). Then, where fr_core_news_lg is installed, run:\n"
          f"  python3 gen_ner_recall_packet.py score {panel}")


def _norm(s):
    return {x.strip().lower() for x in str(s).split(";") if x.strip()}


def score(panel, model="fr_core_news_lg"):
    p = ROOT / f"analysis_{panel}" / "ner_recall_packet.csv"
    df = pd.read_csv(p).fillna("")
    ann = df[df["gold_entities"].astype(str).str.strip() != ""]
    if ann.empty:
        sys.exit("No annotated rows (gold_entities all empty). Fill the packet first.")
    import spacy
    mask_outlet = _mask_outlet_factory()
    nlp = spacy.load(model, disable=["morphologizer", "parser", "attribute_ruler", "lemmatizer"])
    print(f"[scoring with spaCy model: {model}]")
    ENT = {"PER", "LOC", "ORG", "MISC"}

    def predict(text):
        doc = nlp(mask_outlet(text))
        return {e.text.strip().lower() for e in doc.ents if e.label_ in ENT}

    tp = fn = 0
    hl_hit = hl_tot = 0
    for _, r in ann.iterrows():
        gold = _norm(r["gold_entities"]); pred = predict(r["headline"])
        caught = sum(1 for g in gold if any(g in pr or pr in g for pr in pred))
        tp += caught; fn += len(gold) - caught
        if gold:
            hl_tot += 1
            hl_hit += 1 if caught == len(gold) else 0
    span_recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"Annotated headlines: {len(ann)}")
    print(f"Span-level NER recall ({model}): {span_recall:.3f}  ({tp}/{tp+fn} gold spans caught)")
    print(f"Headlines where ALL gold entities were caught: {hl_hit}/{hl_tot} ({100*hl_hit/max(hl_tot,1):.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score"])
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="fr_core_news_lg", help="spaCy model for scoring (pipeline used lg)")
    a = ap.parse_args()
    (build(a.panel, a.n, a.seed) if a.mode == "build" else score(a.panel, a.model))


if __name__ == "__main__":
    main()
