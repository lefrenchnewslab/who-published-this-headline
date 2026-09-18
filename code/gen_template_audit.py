"""
Template leakage audit — generates audit_testB_templates.csv (Level 3; needs the rebuilt corpus).

Reads the rebuilt corpus at results/n10/sampled_dataset_n10.parquet (produced by the §A
corpus-rebuild step in the README). Run from anywhere with the project venv.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # package root
OUT  = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

SAMPLED = ROOT / "results/n10/sampled_dataset_n10.parquet"
if not SAMPLED.exists():
    raise FileNotFoundError(
        f"Cannot find {SAMPLED}. Rebuild the corpus first (README §A) — this is a Level-3 "
        "script that computes on headline text; its output audit_testB_templates.csv is "
        "already shipped in data/derived_features/ for verification without rebuilding."
    )

df = pd.read_parquet(SAMPLED)
print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

# Use temporal TEST set only (year 2025) — same as the headline model evaluation
test = df[df["year"] == 2025].copy()
print(f"Temporal test (2025): {len(test):,} rows")

# Extract first 8 words of headline as the "lead"
text_col = "title" if "title" in test.columns else "headline"
test["lead8"] = test[text_col].str.lower().str.split().str[:8].str.join(" ")

rows = []
for outlet, grp in test.groupby("outlet"):
    counts = grp["lead8"].value_counts()
    top_lead = counts.index[0]
    share = counts.iloc[0] / len(grp)
    rows.append({"outlet": outlet, "top_lead8": top_lead, "share": round(share, 4), "n": len(grp)})

audit = pd.DataFrame(rows).sort_values("share", ascending=False)
print("\nTemplate leakage audit (temporal test, 2025):")
print(audit.to_string(index=False))
print(f"\nMax share: {audit['share'].max():.4f} — paper claims < 0.004 (0.4%)")

audit.to_csv(OUT / "audit_testB_templates.csv", index=False)
print(f"\nSaved: {OUT / 'audit_testB_templates.csv'}")
