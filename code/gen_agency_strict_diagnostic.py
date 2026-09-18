"""
DIAGNOSTIC for gen_agency_strict_robustness.py -- audits whether the "strict / animate"
agency flag is trustworthy or subpar before interpreting its null result.

The strict flag demoted BFMTV's agency effect from significant to null. That is only
meaningful if the subjects it drops (original=1 -> strict=0) are genuinely institutions
/ non-animate. If instead it is dropping real PEOPLE that spaCy's NER simply failed to
tag as PER (surname-only subjects like "Mbappe marque", "Le Pen denonce"), then the null
is an artifact of the instrument's recall, not a fact about BFMTV.

This script does NOT re-fit anything. It re-parses the 3,006 matched-event headlines and,
for every headline the ORIGINAL rule scores as agentive, records the SUBJECT token and
classifies it BY SPACY'S OWN LABELS (PER/ORG/LOC/MISC/none + POS), so the audit does not
depend on the hand-built human-noun list. It then reports, overall and per outlet (BFMTV
highlighted):

  * how many agentive subjects the strict flag demotes,
  * the demoted subjects broken down by spaCy entity label + POS,
  * a person-vs-institution estimate of the demoted set:
        likely_person      = PER entity, OR a bare PROPN with NO entity / MISC label
                             (i.e. a proper name NER probably missed as a person),
        likely_institution = ORG entity, OR a common-noun institutional actor
                             (gouvernement/parquet/entreprise/... = ACTOR_NOUNS minus humans),
  * the top demoted subject lemmas, and a dump of BFMTV's demoted headlines to eyeball.

It also reports two ALTERNATIVE strict definitions' base rates so we can see how sensitive
the null is to the animacy operationalization:
  strict_A (current)  : PER span OR human-noun-list OR capitalized-plural demonym
  strict_B (person-proxy, list-free) : PER span OR (PROPN not tagged ORG/LOC) OR
                                       human-noun-list OR capitalized-plural demonym
strict_B is the key robustness check on threat #1: it counts NER-missed surnames as
animate (any proper noun that spaCy did NOT tag as an organization/place), so if the null
holds under strict_A but dissolves under strict_B, my original strict flag was dropping
people, not institutions.

USAGE
-----
  python3 gen_agency_strict_diagnostic.py n10 --spacy fr_core_news_lg
  python3 gen_agency_strict_diagnostic.py n10 --spacy fr_core_news_lg --dump 60

OUTPUT
------
  <clusters_dir>/agency_strict_diagnostic.csv   one row per original-agentive headline:
     outlet, title, subj_text, subj_pos, subj_ent, orig_reason, strict_A, strict_B,
     demoted_A, demoted_B, person_or_institution
  + a printed report.

Requires spacy + fr_core_news_lg + pandas + pyarrow. No GPU, no statsmodels.
"""
import os
import sys
import glob
import argparse
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

# ---- shared lexicons (identical to gen_agency_strict_robustness.py) ----
UNACCUSATIVE = {'etre', 'deceder', 'mourir', 'exister', 'rester', 'devenir', 'arriver', 'tomber',
                'grimper', 'chuter', 'baisser', 'augmenter', 'monter', 'survenir', 'apparaitre',
                'disparaitre', 'naitre', 'exploser', 'seffondrer', 'effondrer', 'progresser',
                'reculer', 'stagner', 'culminer', 'flamber', 'plonger', 'derailler', 'seproduire'}
ACTOR_NOUNS = {'gouvernement', 'etat', 'ministere', 'ministre', 'president', 'presidente', 'maire',
               'prefet', 'parquet', 'justice', 'tribunal', 'police', 'gendarmerie', 'armee', 'syndicat',
               'patronat', 'parti', 'opposition', 'majorite', 'entreprise', 'groupe', 'societe', 'banque',
               'direction', 'conseil', 'commission', 'assemblee', 'senat', 'parlement', 'union', 'club',
               'equipe', 'federation', 'association', 'ong', 'eglise', 'cathedrale', 'manifestant',
               'salarie', 'greviste', 'electeur', 'habitant', 'depute', 'senateur', 'elu', 'dirigeant',
               'autorite', 'autorites', 'etats-unis', 'etat-unis'}
HUMAN_ACTOR_NOUNS = {'ministre', 'president', 'presidente', 'maire', 'prefet', 'depute', 'senateur',
                     'elu', 'dirigeant', 'manifestant', 'salarie', 'greviste', 'electeur', 'habitant',
                     'candidat', 'candidate', 'juge', 'avocat', 'magistrat', 'policier', 'gendarme',
                     'patron', 'ouvrier', 'enseignant', 'medecin', 'militant', 'syndicaliste',
                     'chercheur', 'porte-parole', 'secretaire', 'chef'}
INSTITUTION_NOUNS = ACTOR_NOUNS - HUMAN_ACTOR_NOUNS   # for the person-vs-institution estimate


def _norm(t):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(t).lower()) if not unicodedata.combining(c))


def _l(tok):
    return _norm(tok.lemma_)


def _has(tok, attr, val):
    try:
        return val in tok.morph.get(attr)
    except Exception:
        return False


def _find(*cands):
    for c in cands:
        hits = glob.glob(c, recursive=True)
        if hits:
            return hits[0]
    return None


def active_subjects(doc):
    """codebook-4 subject gate: active-voice nsubj whose head is not passive/unaccusative."""
    if any(t.dep_ in ('aux:pass', 'nsubj:pass') or _has(t, 'Voice', 'Pass') for t in doc):
        return []
    subs = []
    for t in doc:
        if t.dep_ != 'nsubj':
            continue
        h = t.head
        if _has(h, 'Voice', 'Pass') or _l(h) in UNACCUSATIVE:
            continue
        subs.append(t)
    return subs


def analyze_doc(doc, outlet, title):
    ent_label = {}
    for e in doc.ents:
        for i in range(e.start, e.end):
            ent_label[i] = e.label_
    ent_acts = {e.text.lower() for e in doc.ents if e.label_ in ('PER', 'ORG')}
    per_idx = {i for e in doc.ents if e.label_ == 'PER' for i in range(e.start, e.end)}

    subs = active_subjects(doc)

    def orig_reason(t):
        tl = t.text.lower()
        if t.pos_ == 'PROPN':
            return 'PROPN'
        if tl in ent_acts:
            return 'ent_PER/ORG'
        if _l(t) in ACTOR_NOUNS:
            return 'actor_noun'
        if t.pos_ in ('NOUN', 'PROPN') and _has(t, 'Number', 'Plur') and t.text[:1].isupper():
            return 'plural_cap'
        return None

    def is_strict_A(t):
        in_per = t.i in per_idx
        human_noun = _l(t) in HUMAN_ACTOR_NOUNS
        human_group = (t.pos_ == 'NOUN' and _has(t, 'Number', 'Plur') and t.text[:1].isupper()
                       and t.i not in per_idx)
        return in_per or human_noun or human_group

    def is_strict_B(t):
        # list-free person proxy: any proper noun spaCy did NOT tag as ORG/LOC counts as a person
        lab = ent_label.get(t.i)
        propn_person = (t.pos_ == 'PROPN' and lab not in ('ORG', 'LOC'))
        return is_strict_A(t) or propn_person

    # pick the first subject that fires the ORIGINAL rule as the "trigger" for reporting
    trig = None
    for t in subs:
        if orig_reason(t) is not None:
            trig = t
            break
    original = int(trig is not None)
    strict_A = int(any(is_strict_A(t) for t in subs)) if original else 0
    strict_B = int(any(is_strict_B(t) for t in subs)) if original else 0

    if not original:
        return None

    lab = ent_label.get(trig.i, 'none')
    # person-vs-institution estimate for the trigger subject (spaCy-label driven, list only as backstop)
    if lab == 'PER' or (trig.pos_ == 'PROPN' and lab in ('none', 'MISC')):
        p_or_i = 'likely_person'
    elif lab == 'ORG' or _l(trig) in INSTITUTION_NOUNS:
        p_or_i = 'likely_institution'
    elif _l(trig) in HUMAN_ACTOR_NOUNS:
        p_or_i = 'likely_person'
    else:
        p_or_i = 'ambiguous'

    return dict(outlet=outlet, title=title, subj_text=trig.text, subj_pos=trig.pos_,
                subj_ent=lab, orig_reason=orig_reason(trig),
                strict_A=strict_A, strict_B=strict_B,
                demoted_A=int(original and not strict_A), demoted_B=int(original and not strict_B),
                person_or_institution=p_or_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('panel', nargs='?', default='n10')
    ap.add_argument('--spacy', default='fr_core_news_lg')
    ap.add_argument('--dump', type=int, default=40, help='how many BFMTV demoted headlines to print')
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
    titles = clustered['title'].fillna('').tolist()
    outlets = clustered['outlet'].tolist()

    import spacy
    try:
        nlp = spacy.load(args.spacy)
    except OSError:
        from spacy.cli import download
        download(args.spacy)
        nlp = spacy.load(args.spacy)
    print('spaCy', spacy.__version__, '| model', args.spacy)
    print(f'parsing {len(titles):,} clustered headlines ...\n')

    recs = []
    for k, doc in enumerate(nlp.pipe(titles, batch_size=256)):
        r = analyze_doc(doc, outlets[k], titles[k])
        if r is not None:
            recs.append(r)
    D = pd.DataFrame(recs)
    D.to_csv(os.path.join(out, 'agency_strict_diagnostic.csv'), index=False)

    n_orig = len(D)
    print('=' * 78)
    print(f'AGENTIVE (original rule): {n_orig} headlines  |  strict_A keeps {int(D.strict_A.sum())}'
          f'  |  strict_B keeps {int(D.strict_B.sum())}')
    print(f'strict_A demotes {int(D.demoted_A.sum())}  |  strict_B demotes {int(D.demoted_B.sum())}')
    print('=' * 78)

    dem = D[D.demoted_A == 1]
    print('\n--- what strict_A drops, by spaCy entity label of the subject ---')
    print(dem.subj_ent.value_counts().to_string())
    print('\n--- what strict_A drops, by POS ---')
    print(dem.subj_pos.value_counts().to_string())
    print('\n--- person-vs-institution estimate of the DROPPED subjects (spaCy-label driven) ---')
    print(dem.person_or_institution.value_counts().to_string())
    print('\n--- top 25 dropped subject surface forms ---')
    print(dem.subj_text.str.lower().value_counts().head(25).to_string())

    print('\n--- per-outlet agentive count, strict_A demotions, and demotion composition ---')
    g = D.groupby('outlet').agg(n_agentive=('title', 'size'),
                                demoted_A=('demoted_A', 'sum'),
                                demoted_B=('demoted_B', 'sum')).reset_index()
    g['demote_rate_A'] = (g.demoted_A / g.n_agentive).round(3)
    g['demote_rate_B'] = (g.demoted_B / g.n_agentive).round(3)
    print(g.sort_values('demote_rate_A', ascending=False).to_string(index=False))

    # BFMTV focus: is its demoted set people (NER-missed) or institutions?
    b = D[D.outlet == 'BFMTV']
    bd = b[b.demoted_A == 1]
    print('\n' + '=' * 78)
    print(f'BFMTV: {len(b)} agentive headlines | strict_A demotes {len(bd)} '
          f'({len(bd) / max(1, len(b)):.1%}) | strict_B demotes {int(b.demoted_B.sum())} '
          f'({int(b.demoted_B.sum()) / max(1, len(b)):.1%})')
    if len(bd):
        print('BFMTV dropped subjects, person-vs-institution:')
        print(bd.person_or_institution.value_counts().to_string())
        print(f'\n--- up to {args.dump} BFMTV headlines strict_A DROPS  [ent|POS] subject :: headline ---')
        for _, r in bd.head(args.dump).iterrows():
            print(f'  [{r.subj_ent:4}|{r.subj_pos:5}] {r.subj_text:22.22} :: {str(r.title)[:80]}')
    print('=' * 78)
    print('\nREAD: if BFMTV\'s dropped subjects are mostly "likely_person" / PROPN with none/MISC '
          'labels,\nthe strict null is an NER-recall artifact (my tool is subpar). If they are mostly '
          'ORG /\ninstitutional nouns, the null is real and the agency effect is institutional.')
    print(f'\nwrote -> {os.path.join(out, "agency_strict_diagnostic.csv")}')
    print('DONE')


if __name__ == '__main__':
    main()
