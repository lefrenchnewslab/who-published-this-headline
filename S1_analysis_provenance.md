# Supplementary S1 — Analysis Provenance

*Companion to "Who Published This Headline? Editorial Fingerprinting in French News Under
Topic Control." Records the order in which the matched-event framing analysis was designed,
committed, and run, so that the confirmatory design and its directional hypotheses can be
seen to predate the confirmatory results. All times are UTC git committer timestamps (§5).*

---

## 1. What is documented

The paper's primary framing result is a single Bonferroni-surviving contrast (BFMTV assigns
agentive subjects more often within the same event). This note substantiates two claims:

1. the **framing measures, within-event estimator, and cluster/measure eligibility rules**
   were fixed *before* the confirmatory fits were run;
2. the **directional hypotheses** follow **prior Tier-0/1 findings**, not the confirmatory
   outcome — in particular, "BFMTV *more* agentive" follows a Tier-1 style observation
   committed two hours before the design was locked.

It does **not** claim a formally sealed, externally time-stamped preregistration (§5).

---

## 2. Commit timeline (2026-06-22, UTC)

| Time (UTC) | Commit | What it fixed |
|---|---|---|
| 12:15:58 | `88a3545` | Tier-1 interpretable style profile (`gen_style_profile.py` + results). |
| 12:18:27 | `bcc77d8` | Tier-1 style-profile session log. |
| **13:41:10** | **`39e8aea`** | **Tier-1 register finding: BFMTV 12% verbless vs. terse outlets 28–30% → BFMTV headlines are "full-clause, verb-driven."** |
| 15:33:28 | `1f17be6` | **Tier-2 design lock**: four framing measures, within-event estimator, clustering + measure eligibility rules. |
| 20:34:02 | `f43b586` | Tier-2 confirmatory results (full-pool within-event fits). |

Design lock → results: ~5 h. Tier-1 agency basis → lock: ~2 h.

---

## 3. What was fixed at the design lock (`1f17be6`)

Verbatim from `TIER2_DESIGN.md` as frozen at `1f17be6`:

- **Four framing measures**, with their Tier-0/1 directional priors:

  | Measure | Prior recorded at the lock |
  |---|---|
  | Valence | "Libération most negative, Ouest-France most positive" |
  | Attribution | "strongest tone cue; BFMTV/Franceinfo high" |
  | Agency / syntactic frame | "nominal-vs-clause split seen in Tier-1 style profile" |
  | Hedging | "Le Monde high, Ouest-France low" |

- **Estimator**: "each outlet's deviation from the cluster mean … tested whether per-outlet
  deviations are systematic across clusters (… mixed-effects with cluster as random
  effect) … with multiple-comparison correction."

- **Clustering eligibility** (§8 "Locked decisions (confirmed 2026-06-22)"): encoder
  `sentence-camembert-large`, ±2-day window, cosine threshold 0.70 (0.65/0.70/0.75 sweep),
  minimum 4 distinct outlets per cluster.

- **Confirmatory framing contrasts** (§8): "all four — valence, attribution, agency/voice,
  hedging," with valence and attribution flagged as having "the strongest Tier-0/1 priors."

---

## 4. Directional basis for the agency contrast (`39e8aea`, 13:41 UTC)

The design doc grounds the agency measure in "the nominal-vs-clause split seen in the Tier-1
style profile." That finding, committed at `39e8aea` in `README_PAPER_N10.md`, reads:

> "verbless/nominal-headline rate: Ouest-France 30% / Les Échos 28% (terse statement
> headlines, no finite verb) vs **BFMTV 12% / Le Parisien 15% (full-clause, verb-driven)**
> — consistent with the statement-vs-teaser axis."

Full-clause headlines place an actor as grammatical subject performing the action, so
"BFMTV *more* agentive" follows this prior observation rather than the confirmatory outcome.

---

## 5. Scope: what is and is not claimed

- **Claimed:** measures, estimator, and eligibility rules were frozen before the confirmatory
  fits; the directional hypotheses follow Tier-0/1 findings already on disk at the lock.
- **Not claimed:** a formally enumerated, externally time-stamped preregistration. The explicit
  one-sided contrast tuples (e.g. `BFMTV agent_subject, gt`) were written into the confirmatory
  notebook in the same commit as the results (`f43b586`); only their *substantive* directional
  basis predates the run.
- **Paper-inclusion threshold was set post hoc**, judged against pre-specified metrics
  (framing-only within-cluster attribution above chance; ≥3 robust interpretable offsets).
- **Timestamp status:** the hashes in §2 are git committer timestamps from the authors'
  development repository. That repository's history is **not** part of this public release,
  which is a snapshot without commit history, so the hashes do not resolve here. They are not a
  third-party immutable timestamp. §3 and §4 therefore quote the locked design and the Tier-1
  finding verbatim so their content can be read without repository access; the development
  history is available from the authors on request.
