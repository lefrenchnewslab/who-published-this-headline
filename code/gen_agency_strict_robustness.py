"""
Stricter-definition robustness check for the BFMTV within-event AGENCY finding.
the one pre-specified contrast
that survives Bonferroni -- BFMTV assigns agentive subjects more often on the same
event (beta = +0.056, p_Bonf = 0.008) -- rests on the lowest-agreement instrument
(agent_subject, kappa = 0.63). We test whether the effect survives a
STRICTER definition of agentive subject: "requiring both a subject NP and an
ANIMATE head noun."

This script re-derives agency under an animacy restriction and re-runs the paper's
EXACT within-event estimator + confirmatory family, so the strict beta is directly
comparable to the published 0.056 / p_Bonf 0.008.

WHAT "STRICT" CHANGES
---------------------
Codebook v2 (tier2_gold_codebook.md, section 4) admits as an agent, in active voice:
  (a) any PROPN,                      <- includes ORG proper nouns (Renault, BFMTV, l'Elysee)
  (b) a named PER/ORG entity token,
  (c) a common-noun/institutional actor from the ACTOR_NOUNS list
      (gouvernement, etat, parquet, tribunal, entreprise, banque, cathedrale, ...),
  (d) a capitalized plural human group (demonym: les Palestiniens, les Francais).

The STRICT variant keeps only ANIMATE / HUMAN subjects:
  * a token inside a PER entity span (named person),                     -> kept
  * a HUMAN role/agent common noun (ministre, maire, depute, manifestant,
    salarie, juge, ... -- the human subset of ACTOR_NOUNS),              -> kept
  * a capitalized plural human group (demonym),                          -> kept
  * everything institutional/inanimate (bare ORG proper nouns, gouvernement,
    parquet, entreprise, cathedrale, l'inflation-style abstractions)     -> DROPPED.
Passive / unaccusative / patient subjects remain 0, exactly as in the codebook.

This is an "animate head noun" restriction, implemented as a superset of
the codebook's own passive/unaccusative exclusions.

WHAT IT DOES (no GPU, no re-embed, no full-corpus re-parse)
-----------------------------------------------------------
Mirrors cell 46 of tier2_matched_event_framing_v2.ipynb:
  1. loads clusters_v2.parquet (the 626-event / ~3,006-headline matched set),
     parse_cache_lg.parquet, valence_cache.parquet from analysis_<panel>/;
  2. re-parses ONLY the clustered headlines with spaCy fr_core_news_lg
     (~3k titles, CPU, seconds) to recover subject tokens + entity types;
  3. computes agent_subject_repro (the ORIGINAL rule) and agent_subject_strict;
  4. SANITY-CHECKS agent_subject_repro against the cached agent_subject column
     (agreement should be ~>=0.98 -- confirms the harness reproduces the instrument);
  5. refits the paper's within-event FE estimator (sum-to-zero outlet coding,
     event-demeaned, char_len covariate, event-clustered robust SE) for the
     original and strict measures;
  6. re-runs the one-sided Bonferroni confirmatory test (family size 4, same as the
     paper) for BFMTV agency under BOTH definitions and prints them side by side.

USAGE
-----
  python3 gen_agency_strict_robustness.py n10 --spacy fr_core_news_lg
  python3 gen_agency_strict_robustness.py n10 --spacy fr_core_news_lg --seed 0

OUTPUT
------
  analysis_<panel>/tier2_v2/agency_strict_robustness.csv    (both definitions, BFMTV row highlighted)
  analysis_<panel>/tier2_v2/tier2_mixedeffects_strict.csv   (full within-event offsets, strict measure)
  Prints a paste-ready verdict line, e.g.:
    BFMTV agency  original: beta=+0.0556 p=0.0021 p_Bonf=0.0084 (SIG)
                  strict  : beta=+0.0xx  p=0.0xxx p_Bonf=0.0xxx (SIG/n.s.)
    strict base rate 0.xxx vs original 0.xxx  (retention 0.xx)

Requires the same env as the other gen_*.py / the notebook: spacy + fr_core_news_lg,
statsmodels, pandas, pyarrow, scikit-learn. No GPU.
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. paths + config -- SAME convention as cell 4 / cell 46 of the notebook
# ----------------------------------------------------------------------------
FRAMING = ['valence', 'attribution', 'agent_subject', 'passive', 'verbless', 'hedging']
REF_OUTLET = 'Le Monde'
MIN_OUTLETS = 4
# The paper's pre-specified confirmatory family (size 4 -> Bonferroni multiplier 4).
CONFIRMATORY = [
    ('Libération', 'valence', 'lt'),
    ('Les Echos', 'attribution', 'lt'),
    ('BFMTV', 'attribution', 'gt'),
    ('BFMTV', 'agent_subject', 'gt'),
]
FAMILY_SIZE = len(CONFIRMATORY)  # 4 -- strict beta is judged at the SAME correction the paper uses


def _find(*cands):
    for c in cands:
        hits = glob.glob(c, recursive=True)
        if hits:
            return hits[0]
    return None


def locate_dirs(panel):
    on_colab = os.path.exists('/content')
    editorial_dir = os.environ.get(
        'EDITORIAL_DIR',
        '.')
    adir = os.path.join(editorial_dir, f'analysis_{panel}')
    out = os.path.join(adir, 'tier2_v2')
    os.makedirs(out, exist_ok=True)
    return editorial_dir, adir, out


# ----------------------------------------------------------------------------
# 1. strict + reproduction agency extractors
#    (subset of feats_from_doc, cell 10 -- rules kept identical except animacy)
# ----------------------------------------------------------------------------
import unicodedata


def _norm(t):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(t).lower())
                   if not unicodedata.combining(c))


def _l(tok):
    return _norm(tok.lemma_)


def _has(tok, attr, val):
    try:
        return val in tok.morph.get(attr)
    except Exception:
        return False


# unaccusative / happening verbs -- subject is an undergoer, never an agent (codebook 4, verbatim)
UNACCUSATIVE = {'etre', 'deceder', 'mourir', 'exister', 'rester', 'devenir', 'arriver', 'tomber',
                'grimper', 'chuter', 'baisser', 'augmenter', 'monter', 'survenir', 'apparaitre',
                'disparaitre', 'naitre', 'exploser', 'seffondrer', 'effondrer', 'progresser',
                'reculer', 'stagner', 'culminer', 'flamber', 'plonger', 'derailler', 'seproduire'}

# ORIGINAL actor nouns (codebook 4 ACTOR_NOUNS, verbatim) -- used only to REPRODUCE the published measure
ACTOR_NOUNS = {'gouvernement', 'etat', 'ministere', 'ministre', 'president', 'presidente', 'maire',
               'prefet', 'parquet', 'justice', 'tribunal', 'police', 'gendarmerie', 'armee', 'syndicat',
               'patronat', 'parti', 'opposition', 'majorite', 'entreprise', 'groupe', 'societe', 'banque',
               'direction', 'conseil', 'commission', 'assemblee', 'senat', 'parlement', 'union', 'club',
               'equipe', 'federation', 'association', 'ong', 'eglise', 'cathedrale', 'manifestant',
               'salarie', 'greviste', 'electeur', 'habitant', 'depute', 'senateur', 'elu', 'dirigeant',
               'autorite', 'autorites', 'etats-unis', 'etat-unis'}

# STRICT: the ANIMATE / HUMAN subset of ACTOR_NOUNS (human role/agent common nouns). Institutions and
# abstractions (gouvernement, etat, parquet, tribunal, entreprise, banque, cathedrale, union, ...) are
# deliberately EXCLUDED -- they are agentive but not animate, which is exactly the relevant edge case.
HUMAN_ACTOR_NOUNS = {'ministre', 'president', 'presidente', 'maire', 'prefet', 'depute', 'senateur',
                     'elu', 'dirigeant', 'manifestant', 'salarie', 'greviste', 'electeur', 'habitant',
                     'candidat', 'candidate', 'juge', 'avocat', 'magistrat', 'policier', 'gendarme',
                     'patron', 'ouvrier', 'enseignant', 'medecin', 'militant', 'syndicaliste',
                     'chercheur', 'porte-parole', 'secretaire', 'chef'}


def _subject_tokens(doc):
    """Active-voice subjects whose head verb is not passive/unaccusative (shared gate, codebook 4)."""
    # passive gate: replicate cell-10's passive detection at the doc level (any passive marker present)
    passive = any(t.dep_ in ('aux:pass', 'nsubj:pass') or _has(t, 'Voice', 'Pass') for t in doc)
    subs = []
    if passive:
        return subs, True
    for t in doc:
        if t.dep_ != 'nsubj':
            continue
        h = t.head
        if _has(h, 'Voice', 'Pass') or _l(h) in UNACCUSATIVE:
            continue
        subs.append(t)
    return subs, False


def agent_original(doc, ent_acts):
    """Reproduce codebook-v2 agent_subject EXACTLY (for the sanity check against the cache)."""
    subs, passive = _subject_tokens(doc)
    if passive:
        return 0
    for t in subs:
        tl = t.text.lower()
        is_actor = (t.pos_ == 'PROPN'
                    or tl in ent_acts
                    or _l(t) in ACTOR_NOUNS
                    or (t.pos_ in ('NOUN', 'PROPN') and _has(t, 'Number', 'Plur') and t.text[:1].isupper()))
        if is_actor:
            return 1
    return 0


def agent_strict(doc, per_token_idx):
    """Animacy-restricted agent_subject: keep only human/personal active subjects."""
    subs, passive = _subject_tokens(doc)
    if passive:
        return 0
    for t in subs:
        in_per = t.i in per_token_idx                                   # named person
        human_noun = _l(t) in HUMAN_ACTOR_NOUNS                         # human role noun
        human_group = (t.pos_ in ('NOUN', 'PROPN') and _has(t, 'Number', 'Plur')
                       and t.text[:1].isupper() and t.i not in per_token_idx
                       and t.pos_ != 'PROPN')                           # plural human demonym (les Palestiniens)
        # a plural-capitalized common NOUN group (not a single ORG proper noun) counts as an animate group
        if in_per or human_noun or human_group:
            return 1
    return 0


def parse_clustered(titles, spacy_model):
    import spacy
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        from spacy.cli import download
        download(spacy_model)
        nlp = spacy.load(spacy_model)
    print('spaCy', spacy.__version__, '| model', spacy_model, '| pipes:', nlp.pipe_names)

    orig, strict = [], []
    for doc in nlp.pipe(titles, batch_size=256):
        ent_acts = {e.text.lower() for e in doc.ents if e.label_ in ('PER', 'ORG')}      # original rule input
        per_idx = {i for e in doc.ents if e.label_ == 'PER' for i in range(e.start, e.end)}  # strict input
        orig.append(agent_original(doc, ent_acts))
        strict.append(agent_strict(doc, per_idx))
    return np.array(orig, dtype=int), np.array(strict, dtype=int)


# ----------------------------------------------------------------------------
# 2. within-event estimator + confirmatory test -- COPIED from cell 46 (unchanged)
# ----------------------------------------------------------------------------
def fit_within_event(clust, measures):
    import statsmodels.api as sm
    a = clust.groupby(['cluster', 'outlet'])[measures + ['char_len']].mean().reset_index()
    recs = []
    for m in measures:
        d = a[['cluster', 'outlet', m, 'char_len']].dropna().copy().rename(columns={m: 'y'})
        if d.y.nunique() < 2 or d.outlet.nunique() < 3:
            continue
        outs = sorted(d.outlet.unique())
        ref = REF_OUTLET if REF_OUTLET in outs else outs[-1]
        coded = [o for o in outs if o != ref]
        d['y_dev'] = d.groupby('cluster')['y'].transform(lambda x: x - x.mean())
        d['clen_dev'] = d.groupby('cluster')['char_len'].transform(lambda x: x - x.mean())
        if d['y_dev'].abs().sum() == 0:
            continue
        ov = d.outlet.values
        Xc = {'outlet_' + o: np.where(ov == o, 1.0, np.where(ov == ref, -1.0, 0.0)) for o in coded}
        X = pd.DataFrame(Xc, index=d.index)
        X['clen_dev'] = d['clen_dev'].values
        X = sm.add_constant(X)
        try:
            mod = sm.OLS(d['y_dev'].values, X).fit(cov_type='cluster',
                                                   cov_kwds={'groups': d['cluster'].values})
            got = [dict(measure=m, outlet=o, beta=mod.params['outlet_' + o], se=mod.bse['outlet_' + o],
                        z=mod.tvalues['outlet_' + o], p=mod.pvalues['outlet_' + o]) for o in coded]
            recs += got
            recs.append(dict(measure=m, outlet=ref, beta=-sum(g['beta'] for g in got),
                             se=np.nan, z=np.nan, p=np.nan))
        except Exception as e:
            print('within-event OLS failed for', m, '->', repr(e))
    return pd.DataFrame(recs)


def one_sided_bonf(ME, outlet, measure, direction, family_size):
    row = ME[(ME.outlet == outlet) & (ME.measure == measure)]
    if len(row) == 0 or pd.isna(row.iloc[0].p):
        return None
    r = row.iloc[0]
    p1 = r.p / 2 if ((r.beta < 0) == (direction == 'lt')) else 1 - r.p / 2
    p_bonf = min(1.0, p1 * family_size)
    return dict(outlet=outlet, measure=measure, direction=direction, beta=round(float(r.beta), 4),
                p_onesided=round(float(p1), 4), p_bonf=round(float(p_bonf), 4), sig=bool(p_bonf < 0.05))


# ----------------------------------------------------------------------------
# 3. main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('panel', nargs='?', default='n10')
    ap.add_argument('--spacy', default='fr_core_news_lg')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    editorial_dir, adir, out = locate_dirs(args.panel)
    # clusters_v2.parquet is the ONLY hard requirement (article_id,title,outlet,category,dt,cluster).
    # It carries the matched-event structure; agency is re-derived here from spaCy, so the parse/valence
    # caches (which lived on Colab/Drive) are OPTIONAL -- used only to also reproduce the other 3
    # confirmatory family rows when present.
    clust_f = _find(os.path.join(editorial_dir, 'tier2_v2_FINAL', 'clusters_v2.parquet'),  # canonical first
                    os.path.join(out, 'clusters_v2.parquet'),
                    os.path.join(adir, '**', 'clusters_v2.parquet'),
                    os.path.join(editorial_dir, '**', 'clusters_v2.parquet'))
    assert clust_f, ('clusters_v2.parquet not found. Set EDITORIAL_DIR to the folder containing it, '
                     'e.g.  EDITORIAL_DIR=. python3 gen_agency_strict_robustness.py n10')
    # write outputs beside the canonical clusters file (the Colab analysis_<panel>/tier2_v2 dir is not local)
    out = os.path.dirname(clust_f)
    parse_f = _find(os.path.join(adir, 'parse_cache_lg.parquet'),
                    os.path.join(editorial_dir, '**', 'parse_cache_lg.parquet'))
    val_f = _find(os.path.join(adir, 'valence_cache.parquet'),
                  os.path.join(editorial_dir, '**', 'valence_cache.parquet'))
    print('clusters:', clust_f)
    print('parse   :', parse_f or 'absent (optional; other-family rows will be skipped)')
    print('valence :', val_f or 'absent (optional)')

    clustered = pd.read_parquet(clust_f)                       # article_id,title,outlet,category,dt,cluster
    clustered['char_len'] = clustered['title'].str.len()
    have_caches = bool(parse_f and val_f)
    other_measures = []
    if have_caches:
        need = ['attribution', 'passive', 'agent_subject', 'verbless', 'hedging']
        parse = pd.read_parquet(parse_f)
        cached_col = 'agent_subject' in parse.columns
        clustered = clustered.merge(parse[['article_id'] + [c for c in need if c in parse.columns]],
                                    on='article_id', how='left')
        val = pd.read_parquet(val_f)[['article_id', 'v_model']]
        clustered = clustered.merge(val, on='article_id', how='left')
        v = clustered['v_model'].fillna(0).values
        sd = v.std()
        clustered['valence'] = (v - v.mean()) / sd if sd > 0 else v * 0
        # the other three confirmatory rows (valence, attribution) need these cached measures
        other_measures = [c for c in ['valence', 'attribution', 'passive', 'verbless', 'hedging']
                          if c in clustered.columns]
    print(f'clustered set: {len(clustered):,} headlines | {clustered.cluster.nunique():,} clusters | '
          f'caches {"present" if have_caches else "absent -> agency-only run"}')

    # --- re-parse clustered titles for the strict + reproduction agency measures (self-contained) ---
    print(f'\nre-parsing {len(clustered):,} clustered headlines with spaCy (CPU) ...')
    orig, strict = parse_clustered(clustered['title'].fillna('').tolist(), args.spacy)
    # 'agent_subject' used in the fit is the FAITHFUL recomputation of the codebook rule. On the same
    # spaCy model this is per-headline identical to the published cached measure, so if the 'original'
    # row below reproduces the paper's +0.056 / p_Bonf 0.008, the harness is validated end-to-end.
    clustered['agent_subject'] = orig
    clustered['agent_subject_strict'] = strict

    # --- optional sanity: recomputed vs cached instrument, only if the parse cache is present ---
    if have_caches and 'agent_subject' in (locals().get('parse', pd.DataFrame())).columns:
        # re-merge cached column under a distinct name to compare
        cachedvals = pd.read_parquet(parse_f)[['article_id', 'agent_subject']].rename(
            columns={'agent_subject': 'agent_subject_cached'})
        cmp = clustered.merge(cachedvals, on='article_id', how='left')
        mask = cmp['agent_subject_cached'].notna()
        if mask.any():
            agree = float((cmp.loc[mask, 'agent_subject'].astype(int)
                           == cmp.loc[mask, 'agent_subject_cached'].astype(int)).mean())
            print(f'SANITY  recomputed vs cached agent_subject: agreement={agree:.4f} '
                  f'(n={int(mask.sum())})')
            if agree < 0.97:
                print('  WARNING: agreement < 0.97 -- spaCy version/model drift; the ORIGINAL-row '
                      'reproduction below is the ground-truth fidelity check regardless.')

    base_o = float(clustered['agent_subject'].mean())
    base_s = float(clustered['agent_subject_strict'].mean())
    print(f'base rate  original={base_o:.3f}  strict={base_s:.3f}  retention={base_s / base_o:.2f}'
          if base_o else 'base rate original=0')

    # --- refit within-event FE for original + strict (+ other family measures if caches present) ---
    ME = fit_within_event(clustered, ['agent_subject', 'agent_subject_strict'] + other_measures)
    ME.to_csv(os.path.join(out, 'tier2_mixedeffects_strict.csv'), index=False)

    res_orig = one_sided_bonf(ME, 'BFMTV', 'agent_subject', 'gt', FAMILY_SIZE)
    res_strict = one_sided_bonf(ME, 'BFMTV', 'agent_subject_strict', 'gt', FAMILY_SIZE)

    rows = []
    for tag, r in [('original', res_orig), ('strict_animate', res_strict)]:
        if r is None:
            print(f'  {tag}: BFMTV absent from fit (skipped)')
            continue
        r = dict(definition=tag, **r)
        rows.append(r)
    RES = pd.DataFrame(rows)
    RES.to_csv(os.path.join(out, 'agency_strict_robustness.csv'), index=False)

    print('\n' + '=' * 74)
    print('BFMTV within-event AGENCY -- original vs stricter (animate-subject) instrument')
    print('  (one-sided, Bonferroni x%d, same family as the paper)' % FAMILY_SIZE)
    print('=' * 74)
    for r in rows:
        flag = 'SIG' if r['sig'] else 'n.s.'
        print(f"  {r['definition']:15s}: beta={r['beta']:+.4f}  p={r['p_onesided']:.4f}  "
              f"p_Bonf={r['p_bonf']:.4f}  ({flag})")
    print('\n=== PASTE-READY ===')
    if res_strict is not None and res_orig is not None:
        print(f"Under a stricter agentive-subject definition requiring an animate (human) head noun "
              f"(base rate {base_s:.3f} vs {base_o:.3f}), the BFMTV within-event agency effect is "
              f"beta={res_strict['beta']:+.3f} (p_Bonf={res_strict['p_bonf']:.3f}), "
              f"{'still surviving' if res_strict['sig'] else 'no longer surviving'} Bonferroni "
              f"correction (published animacy-inclusive estimate: beta=+0.056, p_Bonf=0.008).")
    print(f"\nwrote -> {os.path.join(out, 'agency_strict_robustness.csv')}")
    print(f"wrote -> {os.path.join(out, 'tier2_mixedeffects_strict.csv')}")
    print('DONE')


if __name__ == '__main__':
    main()
