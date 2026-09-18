"""
Systematic extraction/template-leakage audit across ALL outlets .

For the headline model's predictions, flag each test headline for common web-extraction
artifacts, then per outlet compare the model's recall on artifact-bearing vs clean headlines.
A large gap = the model is exploiting a template/CMS artifact rather than editorial style.

Run: python3 gen_leakage_audit.py n10
Writes analysis_<panel>/leakage_audit.csv
"""
import sys, re, glob
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent  # package root

OUTLET_NAMES = ["le monde","le figaro","les echos","les échos","libération","liberation","la croix",
                "l'humanité","humanité","humanite","huma","le parisien","franceinfo","france info",
                "bfmtv","bfm","tf1","ouest-france","ouest france","20 minutes","le point","nouvel obs",
                "l'express","huffpost"]

def detect(t):
    s = str(t); low = s.lower()
    return {
        # trailing " - X" / " | X" / " — X" site/section suffix
        "trail_suffix": bool(re.search(r"\s[-|–—]\s+[A-ZÀ-Ÿ0-9][^-|–—]{1,25}$", s)),
        # the title literally ends in an outlet name (post any masking this is the structural tell)
        "ends_outlet": any(low.rstrip().endswith(n) for n in OUTLET_NAMES),
        # leading "Section. " or "Section : " label
        "lead_label": bool(re.search(r"^[A-ZÀ-Ÿ][\w'’ .-]{1,22}\s*[.:]\s", s)),
        # live/video/replay/"en direct"/"suivez" broadcast or liveblog markers
        "live_video": bool(re.search(r"(?i)\b(en direct|le direct|direct|vid[eé]o|live|replay|suivez)\b", s)),
        # index/landing-page boilerplate
        "index_page": bool(re.search(r"(?i)(toute l.actualit|actualit[eé], infos|donn[eé]es pays|infos? et actualit)", s)),
        # a URL / domain fragment leaked into the title
        "url_frag": bool(re.search(r"(?i)(https?://|www\.|\.fr\b|\.com\b)", s)),
    }

def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "n10"
    rid = glob.glob(f"results/{panel}/runs/preds/{panel}_camembert_pooled_ALL_headline_only_mask_entities_temporal_s42_*")[0]
    pr = pd.read_parquet(rid)
    d = pd.read_parquet(f"results/{panel}/sampled_dataset_{panel}.parquet")[["article_id","title"]]
    m = pr.merge(d, on="article_id", how="left")
    flags = m["title"].apply(detect).apply(pd.Series)
    m = pd.concat([m, flags], axis=1)
    m["any_artifact"] = flags.any(axis=1)

    cols = list(flags.columns)
    rows = []
    for o, g in m.groupby("true"):
        rec_art = g[g.any_artifact]["correct"].mean() if g.any_artifact.any() else np.nan
        rec_clean = g[~g.any_artifact]["correct"].mean() if (~g.any_artifact).any() else np.nan
        row = {"outlet": o, "n": len(g), "recall": round(g["correct"].mean(), 3),
               "pct_any_artifact": round(100*g.any_artifact.mean(), 1),
               "recall_artifact": round(rec_art, 3) if rec_art==rec_art else None,
               "recall_clean": round(rec_clean, 3) if rec_clean==rec_clean else None,
               "gap": round(rec_art-rec_clean, 3) if (rec_art==rec_art and rec_clean==rec_clean) else None}
        for c in cols:
            row[f"pct_{c}"] = round(100*g[c].mean(), 1)
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("gap", ascending=False, na_position="last")
    out = ROOT / f"analysis_{panel}"; out.mkdir(exist_ok=True)
    res.to_csv(out / "leakage_audit.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(res[["outlet","n","recall","pct_any_artifact","recall_artifact","recall_clean","gap"]].to_string(index=False))
    print("\nartifact prevalence by type (% of each outlet's headlines):")
    print(res[["outlet"]+[f"pct_{c}" for c in cols]].to_string(index=False))

if __name__ == "__main__":
    main()
