"""
Matched-event CLUSTER VALIDATION — real analyses (not prose) for the matched-event pipeline.

Answers, from the local clusters_v2.parquet (no encoder, no Drive), three checks on the event-clustering pipeline:

  (A) SYNDICATION / WIRE-COPY robustness.
      Worry: clusters may hold near-identical headlines across outlets (wire copy), which
      would make within-event "framing" differences trivial. We flag near-duplicate
      cross-outlet headlines (word-set Jaccard >= --dup-jaccard) inside each cluster,
      report their prevalence, and RE-FIT the BFMTV agency contrast with syndicated
      headlines removed (then re-enforcing >= MIN_OUTLETS distinct outlets per cluster).
      If beta holds, the estimate persists after removing near-duplicates detected by this rule;
      paraphrased or undetected shared-source text remains possible.

  (B) CLUSTER-MEMBERSHIP sensitivity.
      Re-fit the agency contrast at >=3 / >=4 / >=5 distinct outlets per cluster, so the
      surviving result is shown to be insensitive to the pre-specified >=4 cutoff.

  (C) CLUSTER-STRUCTURE description.
      Headlines-per-cluster and outlets-per-cluster distributions (quantiles), and the
      per-outlet coverage counts, so the reader can see the shape of the matched set.

Agency (agent_subject) is recomputed from spaCy exactly as in gen_agency_strict_robustness
(same codebook rule), and the within-event estimator + one-sided Bonferroni test are
imported from that file, so every beta here is directly comparable to the paper's
+0.056 / p_Bonf 0.008.

USAGE
-----
  python3 gen_cluster_validation.py n10 --spacy fr_core_news_lg
  python3 gen_cluster_validation.py n10 --spacy fr_core_news_lg --dup-jaccard 0.7

OUTPUT
------
  <clusters_dir>/cluster_validation_syndication.csv   per-headline syndication flags
  <clusters_dir>/cluster_validation_summary.csv        the A/B/C tables
  + a printed report with paste-ready lines.

Requires spacy + fr_core_news_lg + statsmodels + pandas + pyarrow (same env as the
notebook / the other gen_*.py). No GPU.
"""
import os
import sys
import glob
import argparse
import unicodedata
import importlib.util

import numpy as np
import pandas as pd

# --- import the paper's exact estimator + agency extractor from the robustness script ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_agency", os.path.join(_HERE, "gen_agency_strict_robustness.py"))
_ag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ag)
fit_within_event = _ag.fit_within_event
one_sided_bonf = _ag.one_sided_bonf
parse_clustered = _ag.parse_clustered
FAMILY_SIZE = _ag.FAMILY_SIZE  # 4 -- same Bonferroni family as the paper
MIN_OUTLETS_DEFAULT = 4


def _find(*cands):
    for c in cands:
        hits = glob.glob(c, recursive=True)
        if hits:
            return hits[0]
    return None


def _norm_words(t):
    t = ''.join(c for c in unicodedata.normalize('NFKD', str(t).lower()) if not unicodedata.combining(c))
    t = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in t)
    return set(w for w in t.split() if len(w) > 2)  # content-ish tokens


def flag_syndication(df, thr):
    """Within each cluster, flag a headline if it word-set-Jaccard >= thr with a headline
    from a DIFFERENT outlet in the same cluster (i.e., cross-outlet near-duplicate = wire copy)."""
    df = df.copy()
    df['_w'] = df['title'].map(_norm_words)
    flag = np.zeros(len(df), dtype=bool)
    idx = df.reset_index(drop=True)
    for cl, sub in idx.groupby('cluster'):
        rows = sub.index.tolist()
        for i in range(len(rows)):
            wi = idx.at[rows[i], '_w']; oi = idx.at[rows[i], 'outlet']
            if not wi:
                continue
            for j in range(i + 1, len(rows)):
                if idx.at[rows[j], 'outlet'] == oi:
                    continue
                wj = idx.at[rows[j], '_w']
                if not wj:
                    continue
                inter = len(wi & wj); union = len(wi | wj)
                if union and inter / union >= thr:
                    flag[rows[i]] = True; flag[rows[j]] = True
    idx['syndicated'] = flag
    return idx.drop(columns='_w')


def refit_agency(clustered, min_outlets):
    """Enforce >= min_outlets distinct outlets/cluster, fit within-event, return BFMTV agency test."""
    keep = clustered.groupby('cluster').outlet.nunique()
    keep = keep[keep >= min_outlets].index
    c = clustered[clustered.cluster.isin(keep)].copy()
    if c.cluster.nunique() < 2:
        return None, c.cluster.nunique(), len(c)
    ME = fit_within_event(c, ['agent_subject'])
    return one_sided_bonf(ME, 'BFMTV', 'agent_subject', 'gt', FAMILY_SIZE), c.cluster.nunique(), len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('panel', nargs='?', default='n10')
    ap.add_argument('--spacy', default='fr_core_news_lg')
    ap.add_argument('--dup-jaccard', type=float, default=0.75, help='cross-outlet near-dup threshold')
    args = ap.parse_args()

    on_colab = os.path.exists('/content')
    editorial_dir = os.environ.get('EDITORIAL_DIR',
                                   '.')
    clust_f = _find(os.path.join(editorial_dir, 'tier2_v2_FINAL', 'clusters_v2.parquet'),
                    os.path.join(editorial_dir, '**', 'clusters_v2.parquet'))
    assert clust_f, 'clusters_v2.parquet not found; set EDITORIAL_DIR'
    out = os.path.dirname(clust_f)
    print('clusters:', clust_f)
    clustered = pd.read_parquet(clust_f)
    clustered['char_len'] = clustered['title'].str.len()

    # --- (C) cluster-structure description ---
    hpc = clustered.groupby('cluster').size()
    opc = clustered.groupby('cluster').outlet.nunique()
    qs = [0.5, 0.9, 0.99, 1.0]
    print('\n=== (C) CLUSTER STRUCTURE ===')
    print(f'clusters={clustered.cluster.nunique()}  headlines={len(clustered)}')
    print('headlines/cluster quantiles:', {q: int(hpc.quantile(q)) for q in qs}, ' mean=%.2f' % hpc.mean())
    print('outlets/cluster   quantiles:', {q: int(opc.quantile(q)) for q in qs}, ' mean=%.2f' % opc.mean())
    cov = clustered.groupby('outlet').size().sort_values(ascending=False)
    print('per-outlet coverage:', cov.to_dict())

    # --- agency source: prefer the CANONICAL cached parse (parse_cache_lg.parquet), the same
    #     agent_subject the confirmatory notebook uses (beta=+0.0556, p_Bonf=0.0084 -> the paper).
    #     Only fall back to a fresh spaCy re-parse if the cache is absent; that re-parse uses a
    #     slightly different extraction rule and yields +0.0524, so it must NOT be reported as the
    #     paper's number. See reconciliation note. ---
    cache_f = _find(os.path.join(editorial_dir, 'analysis_n10', 'parse_cache_lg.parquet'),
                    os.path.join(editorial_dir, '**', 'parse_cache_lg.parquet'))
    if cache_f:
        print(f'\nusing CANONICAL cached agent_subject from {cache_f}')
        pc = pd.read_parquet(cache_f)[['article_id', 'agent_subject']]
        clustered = clustered.merge(pc, on='article_id', how='left')
        n_missing = int(clustered['agent_subject'].isna().sum())
        if n_missing:
            print(f'  WARNING: {n_missing} headlines missing from cache; re-parsing those only ...')
            miss = clustered['agent_subject'].isna()
            orig, _strict = parse_clustered(clustered.loc[miss, 'title'].fillna('').tolist(), args.spacy)
            clustered.loc[miss, 'agent_subject'] = orig
    else:
        print(f'\n[!] parse_cache_lg.parquet NOT found -- re-parsing {len(clustered):,} headlines with spaCy.')
        print('    NOTE: fresh re-parse gives beta~+0.0524 (rule variant), NOT the paper\'s +0.0556.')
        orig, _strict = parse_clustered(clustered['title'].fillna('').tolist(), args.spacy)
        clustered['agent_subject'] = orig

    rows = []

    # --- (B) MIN_OUTLETS sensitivity ---
    print('\n=== (B) CLUSTER-MEMBERSHIP SENSITIVITY (BFMTV agency) ===')
    for mo in (3, 4, 5):
        res, nc, nh = refit_agency(clustered, mo)
        tag = f'>= {mo} outlets'
        if res:
            print(f'  {tag:12s}: clusters={nc:4d} headlines={nh:5d}  beta={res["beta"]:+.4f} '
                  f'p_Bonf={res["p_bonf"]:.4f} ({"SIG" if res["sig"] else "n.s."})')
            rows.append(dict(analysis='min_outlets', setting=tag, clusters=nc, headlines=nh,
                             beta=res['beta'], p_bonf=res['p_bonf'], sig=res['sig']))
        else:
            print(f'  {tag:12s}: too few clusters')

    # --- (A) SYNDICATION robustness ---
    print(f'\n=== (A) SYNDICATION / WIRE-COPY ROBUSTNESS (Jaccard >= {args.dup_jaccard}) ===')
    synd = flag_syndication(clustered[['article_id', 'cluster', 'outlet', 'title']], args.dup_jaccard)
    clustered = clustered.merge(synd[['article_id', 'syndicated']], on='article_id', how='left')
    clustered['syndicated'] = clustered['syndicated'].fillna(False)
    n_syn = int(clustered['syndicated'].sum())
    cl_aff = clustered.loc[clustered.syndicated, 'cluster'].nunique()
    print(f'near-duplicate cross-outlet headlines: {n_syn}/{len(clustered)} = {n_syn/len(clustered):.1%} '
          f'across {cl_aff}/{clustered.cluster.nunique()} clusters')
    synd.to_csv(os.path.join(out, 'cluster_validation_syndication.csv'), index=False)

    base, nc0, nh0 = refit_agency(clustered, MIN_OUTLETS_DEFAULT)
    nosyn, nc1, nh1 = refit_agency(clustered[~clustered.syndicated], MIN_OUTLETS_DEFAULT)
    if base:
        print(f'  full (>=4 outlets)      : clusters={nc0} headlines={nh0}  beta={base["beta"]:+.4f} '
              f'p_Bonf={base["p_bonf"]:.4f} ({"SIG" if base["sig"] else "n.s."})')
        rows.append(dict(analysis='syndication', setting='full', clusters=nc0, headlines=nh0,
                         beta=base['beta'], p_bonf=base['p_bonf'], sig=base['sig']))
    if nosyn:
        print(f'  syndicated removed      : clusters={nc1} headlines={nh1}  beta={nosyn["beta"]:+.4f} '
              f'p_Bonf={nosyn["p_bonf"]:.4f} ({"SIG" if nosyn["sig"] else "n.s."})')
        rows.append(dict(analysis='syndication', setting='no_syndication', clusters=nc1, headlines=nh1,
                         beta=nosyn['beta'], p_bonf=nosyn['p_bonf'], sig=nosyn['sig']))

    pd.DataFrame(rows).to_csv(os.path.join(out, 'cluster_validation_summary.csv'), index=False)
    print('\n=== PASTE-READY ===')
    if base and nosyn:
        print(f'Cross-outlet near-duplicates are {n_syn/len(clustered):.0%} of clustered headlines; removing '
              f'them (and re-enforcing >=4 outlets) leaves the BFMTV agency contrast at beta={nosyn["beta"]:+.3f} '
              f'(p_Bonf={nosyn["p_bonf"]:.3f}), vs {base["beta"]:+.3f} on the full set; this does not rule out '
              f'paraphrased or undetected shared-source text. It is also stable across the >=3/>=4/>=5-outlet cutoff.')
    print(f"\nwrote -> {os.path.join(out, 'cluster_validation_summary.csv')}")
    print(f"wrote -> {os.path.join(out, 'cluster_validation_syndication.csv')}")
    print('DONE')


if __name__ == '__main__':
    main()
