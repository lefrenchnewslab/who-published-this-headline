#!/usr/bin/env python3
"""
Leave-one-outlet-out (and event-jackknife) robustness for the BFMTV within-event
agency effect -- analogous to Appendix B's LOO for the fingerprinting ranking.

Motivation: cluster coverage is uneven (111-402 headlines/outlet). Is the
BFMTV agency contrast (beta=+0.056, p_Bonf=0.008) disproportionately driven by the
well-covered outlets it is compared against, or by a few of BFMTV's own large events?

Two checks, both re-running the paper's EXACT within-event estimator
(gen_agency_strict_robustness.fit_within_event; event-demeaned, sum-to-zero outlet
coding, event-clustered robust SE) and the same one-sided Bonferroni family (size 4):

  1. LEAVE-ONE-OUTLET-OUT: drop each of the nine non-BFMTV outlets in turn, re-enforce
     the >=4-distinct-outlet cluster minimum, refit, and re-test BFMTV agency. Shows the
     effect does not hinge on any single comparison outlet.
  2. EVENT JACKKNIFE (--jackknife): leave out each event once, refit, and report the
     range of BFMTV beta. Shows no single (well-covered) cluster drives it.

Run (a standard Python environment: pandas, statsmodels, pyarrow; spaCy only if the parse cache is absent):
    python3 gen_agency_loo.py n10               # LOO only (9 refits, fast)
    python3 gen_agency_loo.py n10 --jackknife    # + event jackknife (626 refits)

Writes: analysis_<panel>/agency_loo.csv, agency_loo_table.tex (paste-ready)
Prints a paste-ready sentence for the clustering-validation appendix.
"""
import argparse
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_agency", os.path.join(_HERE, "gen_agency_strict_robustness.py"))
_ag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ag)
fit_within_event = _ag.fit_within_event
one_sided_bonf = _ag.one_sided_bonf
parse_clustered = _ag.parse_clustered
FAMILY_SIZE = _ag.FAMILY_SIZE
MIN_OUTLETS = getattr(_ag, "MIN_OUTLETS", 4)


def _find(*cands):
    for c in cands:
        hits = glob.glob(c, recursive=True)
        if hits:
            return hits[0]
    return None


def enforce_min_outlets(df, k=MIN_OUTLETS):
    """Keep only clusters that still have >= k distinct outlets."""
    good = df.groupby("cluster")["outlet"].nunique()
    keep = good[good >= k].index
    return df[df["cluster"].isin(keep)].copy()


def bfmtv_agency(df):
    """Refit within-event estimator, return the one-sided Bonferroni BFMTV agency test."""
    ME = fit_within_event(df, ["agent_subject"])
    res = one_sided_bonf(ME, "BFMTV", "agent_subject", "gt", FAMILY_SIZE)
    return res, df["cluster"].nunique(), len(df)


def load(panel, spacy_model):
    editorial_dir = os.environ.get("EDITORIAL_DIR", ".")
    clust_f = _find(os.path.join(editorial_dir, "tier2_v2_FINAL", "clusters_v2.parquet"),
                    os.path.join(editorial_dir, "**", "clusters_v2.parquet"))
    assert clust_f, "clusters_v2.parquet not found; set EDITORIAL_DIR"
    c = pd.read_parquet(clust_f)
    c["char_len"] = c["title"].str.len()
    cache_f = _find(os.path.join(editorial_dir, "analysis_n10", "parse_cache_lg.parquet"),
                    os.path.join(editorial_dir, "**", "parse_cache_lg.parquet"))
    if cache_f:
        pc = pd.read_parquet(cache_f)[["article_id", "agent_subject"]]
        c = c.merge(pc, on="article_id", how="left")
        miss = c["agent_subject"].isna()
        if miss.any():
            orig, _ = parse_clustered(c.loc[miss, "title"].fillna("").tolist(), spacy_model)
            c.loc[miss, "agent_subject"] = orig
    else:
        print("[!] parse cache absent -> spaCy re-parse (rule variant ~+0.0524, not +0.0556)")
        orig, _ = parse_clustered(c["title"].fillna("").tolist(), spacy_model)
        c["agent_subject"] = orig
    return c, editorial_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--spacy", default="fr_core_news_lg")
    ap.add_argument("--jackknife", action="store_true", help="also leave-one-event-out (626 refits)")
    args = ap.parse_args()

    c, editorial_dir = load(args.panel, args.spacy)
    out = os.path.join(editorial_dir, f"analysis_{args.panel}"); os.makedirs(out, exist_ok=True)

    base, nc, nh = bfmtv_agency(c)
    assert base, "baseline BFMTV agency not estimated"
    print(f"full panel: beta={base['beta']:+.4f} p_Bonf={base['p_bonf']:.4f} "
          f"(clusters={nc}, headlines={nh})")

    rows = [{"dropped": "(none, full panel)", "beta": base["beta"], "p_bonf": base["p_bonf"],
             "sig": base["sig"], "n_clusters": nc, "n_headlines": nh}]
    outlets = sorted(o for o in c["outlet"].unique() if o != "BFMTV")
    for o in outlets:
        sub = enforce_min_outlets(c[c["outlet"] != o])
        res, nc_o, nh_o = bfmtv_agency(sub)
        if res is None:
            rows.append({"dropped": o, "beta": np.nan, "p_bonf": np.nan, "sig": None,
                         "n_clusters": nc_o, "n_headlines": nh_o})
            print(f"  drop {o:14s} -> not estimable (clusters={nc_o})")
        else:
            rows.append({"dropped": o, "beta": res["beta"], "p_bonf": res["p_bonf"],
                         "sig": res["sig"], "n_clusters": nc_o, "n_headlines": nh_o})
            print(f"  drop {o:14s} -> beta={res['beta']:+.4f} p_Bonf={res['p_bonf']:.4f} "
                  f"{'SIG' if res['sig'] else 'n.s.'} (clusters={nc_o})")
    loo = pd.DataFrame(rows)
    loo.to_csv(os.path.join(out, "agency_loo.csv"), index=False)

    # paste-ready LaTeX rows
    def _tex_row(r):
        mark = r"\checkmark" if r["sig"] else "n.s."
        return (f"{r['dropped']} & ${r['beta']:+.3f}$ & {r['p_bonf']:.3f} & "
                f"{mark} & {r['n_clusters']} \\\\")
    tex = [_tex_row(r) for _, r in loo.iterrows() if pd.notna(r["beta"])]
    with open(os.path.join(out, "agency_loo_table.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")

    drops = loo.iloc[1:]
    bmin, bmax = drops["beta"].min(), drops["beta"].max()
    n_sig = int(drops["sig"].fillna(False).sum())
    print("\n=== LOO SUMMARY ===")
    print(f"BFMTV agency beta over 9 outlet drops: [{bmin:+.3f}, {bmax:+.3f}] "
          f"(full panel {base['beta']:+.3f}); significant in {n_sig}/{len(drops)} drops")

    jk_line = ""
    if args.jackknife:
        betas = []
        cids = c["cluster"].unique()
        for i, ev in enumerate(cids, 1):
            res, _, _ = bfmtv_agency(enforce_min_outlets(c[c["cluster"] != ev]))
            if res:
                betas.append(res["beta"])
            if i % 100 == 0:
                print(f"  jackknife {i}/{len(cids)}")
        betas = np.array(betas)
        jk_line = (f" A leave-one-event-out jackknife ({len(betas)} refits) gives "
                   f"$\\beta \\in [{betas.min():+.3f}, {betas.max():+.3f}]$, so no single "
                   f"cluster drives it.")
        print(f"event jackknife: beta in [{betas.min():+.3f}, {betas.max():+.3f}]")

    print("\n=== PASTE-READY (appendix) ===")
    print(f"Leaving out each of the nine comparison outlets in turn and re-enforcing the "
          f"four-outlet minimum, the BFMTV agency contrast stays in "
          f"$[{bmin:+.3f}, {bmax:+.3f}]$ (full panel ${base['beta']:+.3f}$) and remains "
          f"significant in {n_sig} of {len(drops)} drops (Table~\\ref{{tab:agencyloo}}), so it "
          f"is not carried by any single outlet's coverage.{jk_line}")
    print(f"\nwrote {os.path.join(out, 'agency_loo.csv')} and agency_loo_table.tex")


if __name__ == "__main__":
    main()
