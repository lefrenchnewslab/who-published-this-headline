# Supplementary Material — Reproduction Guide

**Paper:** *Who Published This Headline? Editorial Fingerprinting in French News Under Topic Control*
**Venue:** WaC-13 @ EMNLP 2026

French headlines are copyrighted, so raw text is **not** shipped. `data/corpus_manifest.csv.gz`
rebuilds the balanced benchmark from public URLs (§A).

**What needs what:**

- **Every number in the paper** is a cell in a released CSV — no code (Level 1).
- **Quantitative figures and attribution tables** regenerate from released tables and text-free
  predictions — no fetch (Level 2). Only the style heatmap needs headline text.
- **Character-level tables and model training** need the corpus rebuilt from the manifest (Level 3).
- **Matched-event framing outputs** are released and verifiable but cannot be re-derived: the
  478,110-title full-pool input is not shipped.

---

## 0. Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn statsmodels pyarrow matplotlib
python3 -m spacy download fr_core_news_lg     # only for entity-masking / NER / agency scripts
```
No build step. Scripts read from `data/derived_features/`, `data/model_predictions/`, and `llm/`
and run from anywhere. Level 3 additionally needs `results/n10/sampled_dataset_n10.parquet` (§A);
the framing scripts need `clusters_v2.parquet`, which the balanced manifest alone cannot rebuild.

---

## Level 1 — Verify any number (open a file)

| Paper item | Open |
|---|---|
| **Table 1** — model ladder accuracies | `data/derived_features/table2_main_results.csv` (+ `table2_bootstrap_ci_temporal.csv`) |
| Per-outlet F1 | `data/derived_features/table3_per_outlet_temporal.csv` |
| Char/word features | `data/derived_features/table4_features_*.csv` |
| Within-category accuracy | `data/derived_features/breakdown_by_category.csv` |
| **Decomposition** 43.5 / 13.6 / 43.0 (+CIs) | `data/derived_features/decomposition_bootstrap.csv` |
| Beat-vocabulary floor (0.234) | `data/derived_features/residual_beatstrip_curve.csv` |
| Typo-normalization ablation (0.427) | `data/derived_features/typo_normalization_ablation.csv` |
| Style-LR (0.269) | `data/derived_features/style_lr_results.csv` |
| **Tone analysis** (§5.3 / App. D: tone-only 0.13, +0.004 pooled / +0.010 within-cat lift; per-outlet z-scores) | `data/derived_features/tone_accuracy.csv`, `tone_zscores.csv` (+ `tone_profile.csv`) |
| McNemar outlet-mask null | `data/derived_features/mcnemar_temporal_mask_entities.csv` |
| **LLM baseline** | `llm/llm_baseline_aggregate.json` (+ `llm/*_predictions.csv`, `llm/llm_baseline_table.tex`) |
| Confirmatory framing βs (BFMTV agency +0.056, p_Bonf 0.008) | `data/derived_features/tier2_confirmatory.csv` |
| Gold gate — 5 of 6 candidate measures pass (`admitted=True`). The 6th, `verbless`, fails gold validation and is **absent** from this CSV; it is exploratory only in §5.4. | `data/derived_features/tier2_measure_validation.csv` |
| Agency robustness (LOO / jackknife) | `data/derived_features/agency_loo.csv` |
| Agency strict/animate null | `data/derived_features/agency_strict_robustness.csv` |
| Cluster validation (syndication / membership) | `data/derived_features/cluster_validation_*.csv` |
| Second gold rater / second same-event coder | `data/derived_features/second_rater_reliability.csv` / `gold/second_coder/` |

## Level 2 — Regenerate without fetching articles

**Figures.** Figure scripts write to `paper/figures/` at the package root; `gen_confusion_matrix.py`
writes CSV + PNG to `analysis/`.

| Paper figure | Script | Writes to | Level (needs) |
|---|---|---|---|
| `fig_accuracy_ladder` | `code/gen_paper_figures.py` | `paper/figures/` | 2 (shipped tables) |
| `fig_per_outlet_f1` | `code/gen_paper_figures.py` | `paper/figures/` | 2 (shipped tables) |
| `fig_beatstrip` | `code/gen_paper_figures.py` | `paper/figures/` | 2 (shipped table) |
| `confusion_camembert_temporal_maskent` | `code/gen_confusion_matrix.py` | `analysis/` (CSV+PNG) | 2 (shipped predictions) |
| `fig_decomp_waterfall` | `code/gen_decomp_waterfall_v2.py` | `paper/figures/` | 2 (paper summary values) |
| `fig_framing_effects` | `code/gen_framing_dotplot_v2.py` | `paper/figures/` | 2 (paper summary values) |
| `fig_tier1_tone` | `code/gen_tone_figure.py` | `paper/figures/` | 2 (shipped tone tables) |
| `fig_style_heatmap` | `code/gen_style_profile.py` | `analysis_n10/figures/` | 3 (rebuilt corpus + spaCy) |

```bash
python3 code/gen_paper_figures.py         # ladder, per-outlet F1, beat-strip
python3 code/gen_decomp_waterfall_v2.py   # decomposition waterfall
python3 code/gen_framing_dotplot_v2.py    # within-event framing dot-plot
python3 code/gen_tone_figure.py           # Appendix D tone analysis
```

**Attribution tables** (from text-free predictions in `data/model_predictions/`):

| Run | Produces |
|---|---|
| `python3 code/gen_confusion_matrix.py` | `confusion_camembert_temporal_maskent.csv/.png` |
| `python3 code/gen_maskoutlet_mcnemar.py` | prints per-seed McNemar stats for the §5.1 outlet-mask null (raw vs. mask_outlet, CamemBERT); writes no file. Distinct from the rung-gap `mcnemar_temporal_mask_entities.csv` shipped in `data/derived_features/` |
| `python3 code/gen_withincat_seed_stability.py n10` | `withincat_seed_stability.csv` |
| `python3 code/gen_drop_outlet_recompute.py n10 --drop "L'Humanité"` | leave-one-outlet-out (also needs class-probability `.npy`, available on request) |

## Level 3 — Re-run from the rebuilt corpus (needs §A first)

| Run | Produces |
|---|---|
| `python3 code/gen_decomposition_bootstrap.py n10 --spacy fr_core_news_lg` | `decomposition_bootstrap.csv` (44/14/43 + CIs) |
| `python3 code/gen_residual_beatstrip.py n10` | `residual_beatstrip_curve.csv` |
| `python3 code/gen_style_profile.py n10 --spacy fr_core_news_lg` | `style_profile_*.csv`, `style_lr_*.csv` |
| `python3 code/gen_typo_normalization_ablation.py n10` | `typo_normalization_ablation.csv` |
| `python3 code/gen_cleaner_validation.py` | `cleaner_validation_by_outlet.csv`, `cleaner_edits_sample.csv` |
| `python3 code/gen_ner_coverage.py n10` + `gen_ner_recall_packet.py build/score n10` | `ner_coverage.csv`, `ner_recall_packet.csv` (≈97% NER span recall) |
| `python3 code/gen_leakage_audit.py n10` | `leakage_audit.csv` |
| `python3 code/gen_topic_robustness.py n10` | `topic_robustness.csv` (within-category char-SVM ≈0.48, §5.1) |
| `python3 code/gen_template_audit.py` | `audit_testA/B/E_*.csv` (template-leakage check, §5.1) |
| `python3 code/gen_analysis.py` | `table4_features_*.csv`, confusion, qualitative examples |
| `code/phase1_train_n10.ipynb` (GPU) | trains the ladder → prediction parquets behind Table 1 |
| `code/xlmr_min_run.py` | XLM-R comparator row (three seeds) |
| `python3 code/gen_llm_baseline.py --model gpt-oss-120b=<id> --model llama-3.3-70b=<id> --model mistral-small=<id> --shots 0,5 --seeds 42,13,1234 --per-outlet 200` | `llm/*_predictions.csv`, `llm/llm_baseline_aggregate.json` (OpenAI-compatible endpoint; key via `CUSTOM_LLM_API_KEY`/`REGOLO_KEY`) |

**Matched-event framing (Tier 2):** the clustering notebook rebuilds `clusters_v2.parquet` from the
478,110-title full article pool (not shipped; needs that corpus + a GPU sentence encoder). The
released tables verify the reported outputs; this package alone does not support an independent
reconstruction of the matched-event analysis. Provenance: `S1_analysis_provenance.md`.

| Run | Produces |
|---|---|
| `code/tier2_matched_event_framing_v2.ipynb` (GPU) | `clusters_v2.parquet`, `tier2_confirmatory.csv`, `tier2_measure_validation.csv` |
| `python3 code/gen_agency_loo.py n10` | `agency_loo.csv` (LOO + `--jackknife`) |
| `python3 code/gen_agency_power.py n10` | binary-agency MDE / power (Appendix E) |
| `python3 code/gen_agency_strict_robustness.py n10 --spacy fr_core_news_lg` | `agency_strict_robustness.csv` (animate/institutional null) |
| `python3 code/gen_agency_strict_diagnostic.py n10 --spacy fr_core_news_lg` | `agency_strict_diagnostic.csv` |
| `python3 code/gen_cluster_validation.py n10 --spacy fr_core_news_lg` | `cluster_validation_*.csv` |
| `python3 code/gen_second_rater_reliability.py` | `second_rater_reliability.csv` |
| `python3 code/gen_secondcoder_sameevent.py build/score` | `gold/second_coder/` |

Shared modules (imported, not run): `code/split_utils.py`, `code/gen_residual_decomposition.py`.

---

## A. Rebuild the corpus
`gunzip data/corpus_manifest.csv.gz` → 170,160 rows:
`article_id, outlet, category, year, date, url, title_sha256, text_sha256`.

1. Fetch each `url`.
2. Verify headline and body against the SHA-256 hashes (UTF-8). If a page is unavailable or either
   hash differs, record it as unrecovered; do not substitute or accept changed content. No recovery
   percentage was measured or is claimed.
3. Save verified articles as `results/articles_pool.parquet`
   (`article_id, outlet, category, date, title, text, url, year`). `code/phase1_train_n10.ipynb`
   applies `clean_headline()` (validated by `code/gen_cleaner_validation.py`), balances the pool
   (5,672 per outlet×category; 17,016/outlet; 56,720/category), and writes
   `results/n10/sampled_dataset_n10.parquet`.

## B. Contents
```
README.md                     — this guide
S1_analysis_provenance.md     — matched-event pre-specification / provenance
code/                         — sampling, training, clustering, analysis, figure scripts
codebook/                     — framing codebook + NER-recall annotation guide
data/corpus_manifest.csv.gz   — corpus-rebuild manifest (ids, hashes, labels, public URLs)
data/derived_features/        — CSV tables backing every result (Level 1)
data/model_predictions/       — 88 text-free per-headline model predictions (Level 2)
gold/                         — expert gold set, second rater, second same-event coder
llm/                          — 18 LLM zero/5-shot per-headline predictions + aggregate JSON/TeX
                                 (harness/prompts live in code/gen_llm_baseline.py)
results/n10/                  — empty placeholder; §A writes the rebuilt corpus here
```
Level-2 scripts create `paper/figures/`, `analysis/`, and `analysis_n10/` at the package root.

## Note on individual headlines
The corpus is not redistributed (§A). A few hundred headlines appear in the annotation/example
files (240-item gold set, same-event coder sheets, NER-recall sample, `qualitative_examples.csv`)
because the codebook and measures cannot be audited without the text they were applied to. This is
a de-minimis annotated sample, not the corpus.
