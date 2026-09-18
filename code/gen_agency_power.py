#!/usr/bin/env python3
"""
Power for the BINARY agency estimator (matched-event confirmatory test).

Motivation: the paper's ~0.12 minimum detectable effect (MDE) is the general
figure for the CONTINUOUS within-event measures (valence, attribution). The one
contrast that survives is the BINARY agent_subject measure, estimated over all
~3,006 clustered headlines, which is more precise. A power calculation specific to
that binary estimator (not the general 0.12) so the reported power is specific to the binary estimator rather than the general MDE.

This recomputes the design-based MDE from the SAME within-event estimator used for
the confirmatory test (gen_agency_strict_robustness.fit_within_event): it fits the
event-demeaned, sum-to-zero, event-clustered-robust OLS for agent_subject, reads the
BFMTV coefficient's cluster-robust standard error, and derives the MDE at 80/90%
power under the pre-specified one-sided Bonferroni threshold (family size 4). The SE
is a property of the design and residual variance, not of the observed effect size,
so the MDE is design-based rather than post-hoc/observed power.

Run (a standard Python environment: pandas, statsmodels, pyarrow; spaCy only if the parse cache is absent):
    python3 gen_agency_power.py n10

Writes: analysis_<panel>/agency_power_binary.csv  (SE, MDE@80, MDE@90, power@observed)
        appends an agent_subject row to tier2_v2_FINAL/tier2_power_analysis.csv
Prints a paste-ready sentence matching Appendix (clustering validation).
"""
import argparse
import glob
import importlib.util
import os
from statistics import NormalDist

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_agency", os.path.join(_HERE, "gen_agency_strict_robustness.py"))
_ag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ag)
fit_within_event = _ag.fit_within_event
parse_clustered = _ag.parse_clustered
FAMILY_SIZE = _ag.FAMILY_SIZE            # 4 -- same Bonferroni family as the confirmatory test

N = NormalDist()
TARGET = ("BFMTV", "agent_subject", "gt")   # the surviving pre-specified contrast


def _find(*cands):
    for c in cands:
        hits = glob.glob(c, recursive=True)
        if hits:
            return hits[0]
    return None


def mde(se: float, power: float, family_size: int) -> float:
    """Minimum detectable effect at `power`, one-sided Bonferroni alpha = 0.05/family."""
    z_alpha = N.inv_cdf(1 - 0.05 / family_size)
    z_power = N.inv_cdf(power)
    return (z_alpha + z_power) * se


def attained_power(beta: float, se: float, alpha_onesided: float) -> float:
    return N.cdf(beta / se - N.inv_cdf(1 - alpha_onesided))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--spacy", default="fr_core_news_lg")
    args = ap.parse_args()

    editorial_dir = os.environ.get("EDITORIAL_DIR", ".")
    clust_f = _find(os.path.join(editorial_dir, "tier2_v2_FINAL", "clusters_v2.parquet"),
                    os.path.join(editorial_dir, "**", "clusters_v2.parquet"))
    assert clust_f, "clusters_v2.parquet not found; set EDITORIAL_DIR"
    clustered = pd.read_parquet(clust_f)
    clustered["char_len"] = clustered["title"].str.len()

    # canonical cached agent_subject (the confirmatory number beta=+0.0556); fall back to reparse
    cache_f = _find(os.path.join(editorial_dir, "analysis_n10", "parse_cache_lg.parquet"),
                    os.path.join(editorial_dir, "**", "parse_cache_lg.parquet"))
    if cache_f:
        pc = pd.read_parquet(cache_f)[["article_id", "agent_subject"]]
        clustered = clustered.merge(pc, on="article_id", how="left")
        miss = clustered["agent_subject"].isna()
        if miss.any():
            orig, _ = parse_clustered(clustered.loc[miss, "title"].fillna("").tolist(), args.spacy)
            clustered.loc[miss, "agent_subject"] = orig
    else:
        print("[!] parse cache absent -> spaCy re-parse (rule variant gives ~+0.0524, not +0.0556)")
        orig, _ = parse_clustered(clustered["title"].fillna("").tolist(), args.spacy)
        clustered["agent_subject"] = orig

    n_events = clustered["cluster"].nunique()
    n_head = len(clustered)

    ME = fit_within_event(clustered, ["agent_subject"])
    row = ME[(ME.outlet == TARGET[0]) & (ME.measure == TARGET[1])]
    assert len(row), "BFMTV agent_subject not estimated"
    beta, se = float(row.iloc[0]["beta"]), float(row.iloc[0]["se"])

    mde80, mde90 = mde(se, 0.80, FAMILY_SIZE), mde(se, 0.90, FAMILY_SIZE)
    pw_bonf = attained_power(beta, se, 0.05 / FAMILY_SIZE)
    pw_uncorr = attained_power(beta, se, 0.05)

    res = {"measure": "agent_subject", "n_events": n_events, "n_headlines": n_head,
           "cluster_robust_se": round(se, 4), "observed_beta": round(beta, 4),
           "mde_80": round(mde80, 4), "mde_90": round(mde90, 4),
           "power_at_observed_bonf": round(pw_bonf, 2),
           "power_at_observed_uncorr": round(pw_uncorr, 2),
           "continuous_mde_ref": 0.123}
    out = os.path.join(editorial_dir, f"analysis_{args.panel}")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([res]).to_csv(os.path.join(out, "agency_power_binary.csv"), index=False)

    # append a binary row to the existing continuous-measure power table (for completeness)
    pa_f = _find(os.path.join(editorial_dir, "tier2_v2_FINAL", "tier2_power_analysis.csv"))
    if pa_f:
        pa = pd.read_csv(pa_f)
        pa = pa[~((pa.get("outlet") == "BFMTV") & (pa.get("measure") == "agent_subject"))]
        pa = pd.concat([pa, pd.DataFrame([{
            "outlet": "BFMTV", "measure": "agent_subject", "n_events": n_events,
            "n_needed": "", "powered": bool(mde80 <= 0.123),  # at least as well-powered as continuous
            "min_detectable_eff": round(mde80, 3)}])],
            ignore_index=True)
        pa.to_csv(pa_f, index=False)

    print(f"events={n_events} headlines={n_head}")
    print(f"BFMTV agent_subject: beta={beta:.4f}  cluster-robust SE={se:.4f}")
    print(f"MDE @80% power (Bonf 1-sided) = {mde80:.4f}   @90% = {mde90:.4f}   "
          f"(continuous-measure MDE = 0.123, so binary is {0.123/mde80:.1f}x tighter)")
    print(f"attained power at observed {beta:.3f}: {pw_bonf:.2f} (Bonf), {pw_uncorr:.2f} (uncorrected)")
    print("\n=== PASTE-READY (appendix) ===")
    print(f"Its within-event, event-clustered robust standard error is {se:.3f}; under the "
          f"pre-specified one-sided Bonferroni threshold this gives a minimum detectable effect of "
          f"{mde80:.3f} at 80\\% power and {mde90:.3f} at 90\\% power, roughly half the "
          f"continuous-measure 0.12. The observed beta={beta:.3f} sits at the estimator's own "
          f"80\\%-power threshold (attained power ~{pw_bonf:.2f} Bonferroni, ~{pw_uncorr:.2f} uncorrected).")


if __name__ == "__main__":
    main()
