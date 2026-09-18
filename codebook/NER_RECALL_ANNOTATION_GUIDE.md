# NER Recall Hand-Check — Annotation Guide

**Goal:** measure how reliably the entity masker (`spaCy fr_core_news_lg`) actually removes named
entities from headlines. We already report *coverage* (the masker fires on 87.5% of headlines);
this measures *recall* — of the entities truly present, how many does the model catch? This is the
the strongest remaining check on the entity-masking numbers, and ~15–20 minutes of annotation addresses it.

**File to annotate:** `analysis_n11/ner_recall_packet.csv` — 110 headlines, 10 per outlet.

---

## What to do

For each row, read the `headline` and fill the **`gold_entities`** column with the named entities
that are actually present, **semicolon-separated**. Leave it blank if there are none.

You are annotating *gold truth* — what a careful human sees — **not** what any model output. The
script compares your gold list against the model's predictions to compute recall.

### Which entities count (spaCy PER / LOC / ORG / MISC)

| Type | Counts | Examples |
|------|--------|----------|
| **PER** person | named individuals | Mélenchon, Macron, Pierre Chaillan |
| **LOC** location | countries, cities, regions, geo-political places | France, Paris, Ukraine, Francfort |
| **ORG** organization | companies, institutions, parties, bodies | OMV, Gazprom, Assemblée, Porsche, BCE |
| **MISC** misc | nationalities, named events, adjectival proper nouns | Autrichien, Russe, Mondial, Jeux olympiques |

### What NOT to mark

- Common nouns and generic terms: *inflation, croissance, gouvernement, président* (without a name).
- Topic words, numbers, dates, currencies on their own.
- The masking placeholders `[OUTLET]` / `[URL]` if you see them (already handled upstream).

### Format rules

- Use the **surface string as it appears** in the headline (keep accents/casing): `Mélenchon`, not `melenchon`.
- Separate multiple entities with `;` → `OMV; Gazprom; Autrichien; Russe`
- One entity per mention; if the same name appears twice, list it once.
- Blank = no named entities in the headline.

---

## Worked examples

| headline | `gold_entities` |
|----------|-----------------|
| Gaz : l'Autrichien OMV met fin à son contrat avec le Russe Gazprom | `Autrichien; OMV; Russe; Gazprom` |
| Économie : la France va devoir composer avec une inflation forte… | `France` |
| Violences conjugales : l'Assemblée unanime pour une aide d'urgence aux victimes | `Assemblée` |
| Porsche, des voitures de sport à la Bourse de Francfort | `Porsche; Bourse de Francfort` (or `Porsche; Francfort`) |
| « Si vous pensez que je ne dois pas être Premier ministre… », affirme Mélenchon | `Mélenchon` |
| Comment bien préparer sa déclaration de revenus cette année | *(blank — no named entities)* |

Borderline calls (e.g. whether "Bourse de Francfort" is one ORG or "Francfort" a LOC) won't move the
recall number much — annotate consistently and move on. If genuinely unsure, note it in the `notes`
column.

---

## Scoring

Once the column is filled, run this **where `fr_core_news_lg` is installed** (a local environment):

```bash
python3 gen_ner_recall_packet.py score n11
```

It runs the masker's NER on the same headlines, compares to your gold, and prints:

- **Span-level recall** = gold entities the model caught / all gold entities
- **Headline-level** = share of headlines where *every* gold entity was caught

Send me those two numbers and I'll add one sentence to Methods/Limitations, converting the current
"coverage, not recall" caveat into a measured recall figure.

> Interpreting the result: high recall (e.g. ≥0.85) strengthens the "style not actors" claim —
> the masker really does remove the entities, so the surviving signal isn't residual names. Lower
> recall is still publishable: we'd state the entity-masked numbers as a conservative bound (some
> residual entity signal may remain), which is the honest framing already in the Limitations.
