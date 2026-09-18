#!/usr/bin/env python3
"""
gen_withincat_seed_stability.py

Quantify cross-seed stability of the within-category accuracy reported in Table 3.
Table 3 shows the seed-42 number only; this recomputes per-category accuracy for ALL
three temporal seeds (42, 13, 1234) of the headline model (CamemBERT, pooled, temporal,
entity-masked) and reports the across-seed mean / min / max / range.

Reuses the already-saved per-seed prediction parquets (no GPU, no retrain). Each parquet
has one row per 2025 test headline with columns: article_id, category, year, true, pred,
correct. Within-category accuracy = mean(correct) restricted to each category, exactly
matching the paper's "pooled model evaluated on each category's test headlines".

Run from anywhere :
    python3 code/gen_withincat_seed_stability.py n10
Writes: analysis_n10/withincat_seed_stability.csv (under the package root)
"""
import sys, glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # package root
panel = sys.argv[1] if len(sys.argv) > 1 else "n10"
seeds = [42, 13, 1234]
tmpl = str(ROOT / "data" / "model_predictions" /
           f"{panel}_camembert_pooled_ALL_headline_only_mask_entities_temporal_s{{s}}_lr1e_05.parquet")

per_seed = {}
overall = {}
for s in seeds:
    path = tmpl.format(s=s)
    df = pd.read_parquet(path, columns=["category", "correct"])
    per_seed[s] = df.groupby("category", observed=True)["correct"].mean()
    overall[s] = df["correct"].mean()

acc = pd.DataFrame(per_seed)  # rows=category, cols=seed
acc["mean"] = acc.mean(axis=1)
acc["min"] = acc[seeds].min(axis=1)
acc["max"] = acc[seeds].max(axis=1)
acc["range"] = acc["max"] - acc["min"]

out_dir = ROOT / f"analysis_{panel}"
out_dir.mkdir(exist_ok=True)
out = str(out_dir / "withincat_seed_stability.csv")
acc.round(4).to_csv(out)

pd.set_option("display.width", 120)
print("Within-category accuracy by seed (CamemBERT, temporal, entity-masked):\n")
print(acc.round(4).to_string())
print("\nOverall pooled accuracy by seed:", {s: round(v, 4) for s, v in overall.items()})
print(f"Max across-seed range over categories: {acc['range'].max():.4f}")
print(f"\nWrote {out}")
