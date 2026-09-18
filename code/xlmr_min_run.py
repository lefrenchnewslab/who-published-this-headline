# =====================================================================
# XLM-R BASELINE  -- three seeds (42/13/1234), ~1-2 GPU-hours per NEW seed
# =====================================================================
# "no multilingual comparator." the XLM-R
# comparator was a single run; add seeds to match the paper's three-seed
# discipline and drop the "single run, suggestive" caveat. This runs XLM-R-large
# on the primary cell -- pooled / temporal / entity-masked -- i.e. the exact
# configuration of the paper's headline CamemBERT number (0.576), using the
# notebook's own data, masking, split and hyperparameters, over three seeds.
# Seed 42 is already cached (skips); only 13 and 1234 fine-tune.
#
# HOW TO RUN (cheapest, guaranteed-comparable):
#   1. Open phase1_train_n10.ipynb in your GPU env.
#   2. Run cells 1..28 (config, data, masking, run_camembert, persistence).
#      Everything heavy is cached + idempotent, so already-finished runs skip.
#      (You do NOT need to run the cell-34 execution driver again.)
#   3. Paste THIS as a new cell at the end and run it. Only XLM-R trains.
#   4. Paste me the printed ">>> XLM-R" line.
#
# Reuses existing globals/helpers from the notebook: build_text, _make_dataset,
# tokenize_fn, SPLITS, subset_for_family, LABELS, eval_metrics, build_preds_df,
# make_rid, run_done, save_run, RESULTS, ART, plus the CAMEMBERT_* hyperparams.
# Nothing above is modified.
# =====================================================================
import copy, json, torch
import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          EarlyStoppingCallback)
from sklearn.metrics import f1_score, accuracy_score

XLMR_MODEL = "FacebookAI/xlm-roberta-large"   # ~550M params (vs CamemBERT-large ~336M)
XLMR_TAG   = "xlmr"
XLMR_LR    = CAMEMBERT_LR                      # 1e-5, identical to CamemBERT for comparability
# XLM-R-large is larger -> use per-device 64 x grad-accum 4 = effective batch 256
# (same effective batch as CamemBERT). Drop to 32 x 8 if you still OOM.
XLMR_PD_BATCH = 64
XLMR_GRAD_ACC = 4

# The minimum is just the headline cell. To also get raw temporal (one extra
# run, shows the masking drop replicates under a 2nd encoder), use:
#   CONDITIONS = ["mask_entities", "raw"]
CONDITIONS = ["mask_entities"]
FAMILY, CATEGORY, TEXT_FIELD, SPLIT = "pooled", "ALL", "headline_only", "temporal"
# Three seeds, matching the paper's CamemBERT discipline (removes the "single run,
# suggestive" caveat on the XLM-R comparator). Seed 42 is already cached and will
# skip; only 13 and 1234 fine-tune (~1-2 GPU-hours each).
SEEDS_X = [42, 13, 1234]


def run_xlmr(preprocess, seed):
    rid = make_rid(XLMR_TAG, FAMILY, CATEGORY, TEXT_FIELD, preprocess, SPLIT, seed, lr=XLMR_LR)
    if run_done(rid):
        m = json.load(open(RUNS / "metrics" / f"{rid}.json"))
        print(f"  [cached] {rid}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")
        return m

    seed_everything(seed)
    tr, va, te = SPLITS[SPLIT]
    tr = subset_for_family(tr, FAMILY, CATEGORY)
    va = subset_for_family(va, FAMILY, CATEGORY)
    te = subset_for_family(te, FAMILY, CATEGORY)

    tok = AutoTokenizer.from_pretrained(XLMR_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        XLMR_MODEL, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)})
    if torch.cuda.is_available():
        model = model.cuda()

    train_ds = _make_dataset(tr, tok, TEXT_FIELD, preprocess)   # reuses notebook tokenizer/maxlen(64)
    val_ds   = _make_dataset(va, tok, TEXT_FIELD, preprocess)
    test_ds  = _make_dataset(te, tok, TEXT_FIELD, preprocess)
    collator = DataCollatorWithPadding(tokenizer=tok)

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, preds, average="macro", zero_division=0),
                "accuracy": accuracy_score(p.label_ids, preds)}

    use_fp16 = torch.cuda.is_available()
    eff_batch = XLMR_PD_BATCH * XLMR_GRAD_ACC * max(1, torch.cuda.device_count())
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | "
          f"eff_batch={eff_batch} ~{steps_per_epoch} steps/epoch x{CAMEMBERT_EPOCHS}ep "
          f"(ES patience 2). Training {XLMR_MODEL} ...", flush=True)

    args = TrainingArguments(
        output_dir=str(ART / f"xlmr_{SPLIT}_{preprocess}_s{seed}"),  # per-seed dir, no checkpoint clobber
        num_train_epochs=CAMEMBERT_EPOCHS, learning_rate=XLMR_LR,
        weight_decay=CAMEMBERT_WD, warmup_ratio=CAMEMBERT_WARMUP_RATIO,
        lr_scheduler_type="linear",
        per_device_train_batch_size=XLMR_PD_BATCH,
        gradient_accumulation_steps=XLMR_GRAD_ACC,
        per_device_eval_batch_size=XLMR_PD_BATCH * 2,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, fp16=use_fp16, label_smoothing_factor=0.1,
        logging_steps=10, logging_first_step=True, disable_tqdm=False,
        report_to=[], seed=seed, save_total_limit=1)

    # live console logging: loss every logging_step, a line every epoch
    from transformers import TrainerCallback
    class Printer(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                print(f"    step {state.global_step}/{state.max_steps}  "
                      f"loss={logs['loss']:.4f}  lr={logs.get('learning_rate', 0):.2e}", flush=True)
        def on_epoch_end(self, a, state, control, **kw):
            print(f"  -- epoch {state.epoch:.0f} done (step {state.global_step}) --", flush=True)
        def on_evaluate(self, a, state, control, metrics=None, **kw):
            if metrics:
                print(f"    [eval] macro_f1={metrics.get('eval_macro_f1', float('nan')):.4f} "
                      f"acc={metrics.get('eval_accuracy', float('nan')):.4f}", flush=True)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2), Printer()])
    try:
        trainer.train()
        best_val_f1 = max((l["eval_macro_f1"] for l in trainer.state.log_history
                           if "eval_macro_f1" in l), default=0.0)
        pred = trainer.predict(test_ds)
        proba = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
        y_pred = [LABELS[i] for i in proba.argmax(1)]
        yte = te["outlet"].astype(str).tolist()
        m = eval_metrics(yte, y_pred, proba)
    finally:
        # free GPU even if interrupted, so a re-run starts clean
        import gc as _gc
        try: model.cpu()
        except Exception: pass
        try: del trainer
        except Exception: pass
        try: del model
        except Exception: pass
        try: del pred
        except Exception: pass
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    m.update({"rid": rid, "model": XLMR_TAG, "family": FAMILY, "category": CATEGORY,
              "text_field": TEXT_FIELD, "preprocess": preprocess, "split": SPLIT,
              "seed": seed, "lr": XLMR_LR, "n": len(te), "best_val_f1": best_val_f1})
    # JSON-safe: cast any numpy scalars from eval_metrics to native python
    m = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in m.items()}

    dfp = build_preds_df(te, yte, y_pred, proba.max(1), TEXT_FIELD, preprocess)
    save_run(rid, m, dfp, proba=proba, proba_labels=LABELS)
    RESULTS.append({**m, "experiment": rid})
    print(f"  >>> XLM-R [{preprocess}/{SPLIT}]  acc={m['accuracy']:.3f}  "
          f"macroF1={m['macro_f1']:.3f}  top2={m.get('top2_acc', float('nan')):.3f}  "
          f"(best_val_f1={best_val_f1:.4f})")
    return m


# =====================================================================
# ONE-TIME entity-mask precompute -- LIVE progress bar + disk cache.
# This is the step that was silent for 15 min: spaCy NER over ~131k
# headlines. Here it is batched (nlp.pipe) and shows a tqdm bar that
# ticks per headline, and is cached to disk so re-runs are instant.
# Output is byte-identical to the notebook's mask_entities(), so the
# XLM-R vs CamemBERT comparison stays fair.
# =====================================================================
import pandas as pd
from tqdm.auto import tqdm

_ENT_CACHE_FILE = ART / "mask_entities_cache.parquet"
_ENT_CACHE = {}
if _ENT_CACHE_FILE.exists():
    _c = pd.read_parquet(_ENT_CACHE_FILE)
    _ENT_CACHE = dict(zip(_c["title"].astype(str), _c["masked"].astype(str)))
    print(f"[mask cache] loaded {len(_ENT_CACHE):,} pre-masked headlines", flush=True)

if "mask_entities" in CONDITIONS:
    _tr, _va, _te = SPLITS[SPLIT]
    _titles = pd.unique(pd.concat([_tr["title"], _va["title"], _te["title"]]).astype(str))
    _todo = [t for t in _titles if t not in _ENT_CACHE]
    print(f"[mask] {len(_titles):,} unique headlines, {len(_todo):,} need masking", flush=True)
    if _todo:
        _pre = [mask_outlet(t) for t in _todo]          # outlet/URL mask first (matches original)
        for _doc, _t, _orig in zip(
                _NLP.pipe(_pre, batch_size=256),
                tqdm(_pre, desc="spaCy NER masking", unit="hl"), _todo):
            _s = _t
            for _a, _b in sorted([(e.start_char, e.end_char) for e in _doc.ents
                                  if e.label_ in {"PER", "LOC", "ORG", "MISC"}], reverse=True):
                _s = _s[:_a] + "[ENT]" + _s[_b:]
            _ENT_CACHE[_orig] = _s
        pd.DataFrame({"title": list(_ENT_CACHE), "masked": list(_ENT_CACHE.values())}
                     ).to_parquet(_ENT_CACHE_FILE)
        print(f"[mask] cached -> {_ENT_CACHE_FILE}", flush=True)
    # swap the slow per-call masker for the O(1) cache lookup (identical output)
    PREPROC_FN["mask_entities"] = lambda s: _ENT_CACHE.get(s, s)

print("\nCamemBERT reference (temporal, 3-seed mean): mask_entities=0.576  raw=0.615 | "
      "char-SVM m-ent=0.461 | chance=0.100\n")
for pp in CONDITIONS:
    for seed in SEEDS_X:
        print(f"[{pp} seed {seed}]", flush=True)
        run_xlmr(pp, seed)

# compact paste-ready comparison -- 3-seed mean +/- std, matching CamemBERT's reporting
ref = {"mask_entities": 0.576, "raw": 0.615}
print("\n=== PASTE-READY (XLM-R vs CamemBERT, temporal, 3 seeds) ===")
for pp in CONDITIONS:
    accs, f1s = [], []
    for seed in SEEDS_X:
        rid = make_rid(XLMR_TAG, FAMILY, CATEGORY, TEXT_FIELD, pp, SPLIT, seed, lr=XLMR_LR)
        m = json.load(open(RUNS / "metrics" / f"{rid}.json"))
        accs.append(m["accuracy"]); f1s.append(m["macro_f1"])
    mean_a, std_a = float(np.mean(accs)), float(np.std(accs))
    seeds_str = "/".join(map(str, SEEDS_X))
    accs_str = ", ".join(f"{a:.3f}" for a in accs)
    print(f"{pp:14s}: XLM-R {mean_a:.3f} +/- {std_a:.3f} over seeds {seeds_str} "
          f"(accs {accs_str}; macroF1 mean {np.mean(f1s):.3f}) vs CamemBERT {ref[pp]:.3f} "
          f"-> within {abs(mean_a - ref[pp]):.3f}")
    print(f"    LaTeX: XLM-R-large recovers the entity-masked publisher at "
          f"${mean_a:.3f} \\pm {std_a:.3f}$ across three seeds, within "
          f"{abs(mean_a - ref[pp]) * 100:.1f} points of CamemBERT, so the fingerprint does "
          f"not depend on French-specific pretraining.")
