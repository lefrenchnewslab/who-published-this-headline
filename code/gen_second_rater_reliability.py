"""
Second-rater reliability for the Tier-2 framing instruments.

Motivation: the paper's gold validation currently rests on a SINGLE expert annotator,
so it establishes instrument-vs-human construct validity but NOT human-human reliability.
Reporting a human-human kappa (multi-rater) study converts a disclosed limitation
into a reported number, and tells us whether agent_subject's kappa=0.63 is an INSTRUMENT
ceiling or a TASK ceiling (the human-human kappa is that ceiling).

This is a two-step tool. No GPU, no models -- just pandas + scikit-learn.

STEP 1 -- make a blinded packet for a second rater:
    python3 gen_second_rater_reliability.py packet --n 80 --seed 7
  Reads tier2_v2_FINAL/tier2_gold_packet.csv (the first rater's source rows) and writes
  tier2_v2_FINAL/second_rater_packet.csv with ONLY: article_id, category, title, and five
  EMPTY columns r2_valence, r2_attribution, r2_passive, r2_agent_subject, r2_hedging.
  It deliberately DROPS outlet, the first rater's gold_* labels, and the rationale, so the
  second rater codes blind, from tier2_gold_codebook.md alone. --n stratifies by category
  (omit --n to export all rows). Hand this file + the codebook to a second annotator.

STEP 2 -- score human-human agreement once the packet is filled in:
    python3 gen_second_rater_reliability.py score \
        --rater2 tier2_v2_FINAL/second_rater_annotated.csv
  Aligns the second rater's r2_* columns with the first rater's gold_* labels (from
  tier2_gold_annotated.csv if present, else the gold_* columns of tier2_gold_packet.csv),
  on the overlapping article_ids, and reports:
    * Cohen's kappa per binary measure (attribution, passive, agent_subject, hedging),
    * Spearman rho for valence (ordinal),
    * raw percent agreement and n,
  writing tier2_v2_FINAL/second_rater_reliability.csv and printing a paste-ready line.

Read the printout against the instrument-vs-gold table (tier2_measure_validation.csv):
if human-human kappa ~ instrument-vs-gold kappa for a measure, the instrument is at the
human ceiling; if human-human is much higher, the instrument has recoverable headroom.
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

BINARY = ['attribution', 'passive', 'agent_subject', 'hedging']
GOLD_COLS = ['gold_valence', 'gold_attribution', 'gold_passive', 'gold_agent_subject', 'gold_hedging']
R2_COLS = ['r2_valence', 'r2_attribution', 'r2_passive', 'r2_agent_subject', 'r2_hedging']


def _find(*cands):
    for c in cands:
        hits = glob.glob(c, recursive=True)
        if hits:
            return hits[0]
    return None


def default_dir():
    on_colab = os.path.exists('/content')
    base = os.environ.get('EDITORIAL_DIR',
                          '.')
    d = _find(os.path.join(base, 'tier2_v2_FINAL'), os.path.join(base, '**', 'tier2_v2_FINAL'))
    return d or os.path.join(base, 'tier2_v2_FINAL')


def make_packet(args):
    d = default_dir()
    src = args.source or _find(os.path.join(d, 'tier2_gold_packet.csv'),
                               os.path.join(d, 'tier2_gold_annotated.csv'))
    assert src and os.path.exists(src), f'gold packet not found under {d}'
    g = pd.read_csv(src)
    keep = g[['article_id', 'category', 'title']].drop_duplicates('article_id').reset_index(drop=True)
    if args.n and args.n < len(keep):
        # stratify by category, proportional, deterministic
        rng = np.random.RandomState(args.seed)
        parts = []
        for cat, sub in keep.groupby('category'):
            k = max(1, round(args.n * len(sub) / len(keep)))
            parts.append(sub.sample(n=min(k, len(sub)), random_state=rng))
        keep = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True).head(args.n)
    for c in R2_COLS:
        keep[c] = ''
    out = os.path.join(d, 'second_rater_packet.csv')
    keep.to_csv(out, index=False)
    print(f'wrote blinded second-rater packet: {out}')
    print(f'  rows: {len(keep)}  | columns: {list(keep.columns)}')
    print('  hand this + tier2_gold_codebook.md to a second annotator; they fill r2_* (0/1; valence -1/0/+1).')


def _cohen_kappa(y1, y2, weights=None):
    """Cohen's kappa (numpy-only). weights=None (nominal) or 'linear' (ordinal)."""
    y1 = np.asarray(y1); y2 = np.asarray(y2)
    cats = np.unique(np.concatenate([y1, y2]))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    O = np.zeros((k, k))
    for a, b in zip(y1, y2):
        O[idx[a], idx[b]] += 1
    n = O.sum()
    if n == 0:
        return float('nan')
    O /= n
    r = O.sum(axis=1); c = O.sum(axis=0)
    E = np.outer(r, c)
    if weights == 'linear':
        W = np.abs(np.subtract.outer(np.arange(k), np.arange(k))).astype(float)
        Wm = W.max() if W.max() > 0 else 1.0
        W = W / Wm
        po = 1 - (W * O).sum()
        pe = 1 - (W * E).sum()
    else:
        po = np.trace(O)
        pe = (r * c).sum()
    return float((po - pe) / (1 - pe)) if (1 - pe) > 1e-12 else float('nan')


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    def rank(x):
        order = x.argsort()
        ranks = np.empty(len(x), float)
        ranks[order] = np.arange(len(x))
        # average ties
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        avg = {}
        for u in np.unique(x):
            m = x == u
            avg[u] = ranks[m].mean()
        return np.array([avg[v] for v in x])
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def score(args):
    cohen_kappa_score = _cohen_kappa
    def spearmanr(a, b):
        class _R:  # mimic scipy's .correlation attribute
            pass
        r = _R(); r.correlation = _spearman(a, b); return r
    d = default_dir()
    r1p = args.rater1 or _find(os.path.join(d, 'tier2_gold_annotated.csv'),
                               os.path.join(d, 'tier2_gold_packet.csv'))
    assert r1p and os.path.exists(r1p), f'first-rater gold not found under {d}'
    assert args.rater2 and os.path.exists(args.rater2), f'--rater2 file not found: {args.rater2}'
    r1 = pd.read_csv(r1p)
    r2 = pd.read_csv(args.rater2)
    have_gold = [c for c in GOLD_COLS if c in r1.columns]
    assert have_gold, f'{r1p} has no gold_* columns'
    m = r1[['article_id'] + have_gold].merge(r2[['article_id'] + [c for c in R2_COLS if c in r2.columns]],
                                             on='article_id', how='inner')
    print(f'first rater : {r1p}\nsecond rater: {args.rater2}\noverlap article_ids scored: {len(m)}\n')

    rows = []
    # valence (ordinal) -> Spearman rho + weighted kappa
    if 'gold_valence' in m.columns and 'r2_valence' in m.columns:
        sv = m.dropna(subset=['gold_valence', 'r2_valence'])
        sv = sv[sv['r2_valence'].astype(str).str.strip() != '']
        if len(sv) > 2:
            a = sv['gold_valence'].astype(float)
            b = sv['r2_valence'].astype(float)
            rho = spearmanr(a, b).correlation
            wk = cohen_kappa_score(a.round().astype(int), b.round().astype(int), weights='linear')
            rows.append(('valence', 'spearman_rho', round(float(rho), 3), len(sv)))
            rows.append(('valence', 'weighted_kappa', round(float(wk), 3), len(sv)))
    # binary measures -> Cohen's kappa + percent agreement
    for meas in BINARY:
        gc, rc = 'gold_' + meas, 'r2_' + meas
        if gc not in m.columns or rc not in m.columns:
            continue
        sub = m.copy()
        sub = sub[sub[rc].astype(str).str.strip() != ''].dropna(subset=[gc, rc])
        if len(sub) < 3:
            continue
        y1 = sub[gc].astype(int)
        y2 = sub[rc].astype(int)
        k = cohen_kappa_score(y1, y2)
        agr = float((y1.values == y2.values).mean())
        rows.append((meas, 'cohen_kappa', round(float(k), 3), len(sub)))
        rows.append((meas, 'pct_agreement', round(agr, 3), len(sub)))

    R = pd.DataFrame(rows, columns=['measure', 'metric', 'value', 'n'])
    out = os.path.join(d, 'second_rater_reliability.csv')
    R.to_csv(out, index=False)
    print('=' * 60)
    print(R.to_string(index=False))
    print('=' * 60)
    kap = R[R.metric == 'cohen_kappa']
    if len(kap):
        paste = '; '.join(f"{r.measure} \\kappa={r.value:.2f}" for _, r in kap.iterrows())
        print('\n=== PASTE-READY (human-human) ===')
        print(f'A second annotator independently coded n={R.n.max()} headlines from the codebook; '
              f'inter-annotator agreement was {paste}.')
    print(f'\nwrote -> {out}')


def main():
    ap = argparse.ArgumentParser(description='second-rater reliability for Tier-2 framing instruments')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('packet', help='make a blinded packet for a second rater')
    p.add_argument('--n', type=int, default=None, help='subsample size (stratified by category); omit=all')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--source', default=None, help='override path to the first-rater gold packet')
    p.set_defaults(func=make_packet)
    s = sub.add_parser('score', help='compute human-human agreement from a filled packet')
    s.add_argument('--rater2', required=True, help='path to second_rater_annotated.csv (r2_* filled)')
    s.add_argument('--rater1', default=None, help='override first-rater gold (default: tier2_gold_annotated.csv)')
    s.set_defaults(func=score)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
