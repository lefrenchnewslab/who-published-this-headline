"""
Recompute all results with one outlet removed, WITHOUT retraining (L'Humanité
carries a templated site-name suffix, a web-extraction artifact).

Method (zero-GPU, exact):
  - Models with saved class probabilities (CamemBERT, CharCNN pure/word, word-LR) are
    re-evaluated by DROPPING the removed outlet's probability column, re-argmax over the
    remaining classes, and dropping the removed outlet's test rows. This uses the SAVED
    predictions+proba (exact test split and true labels), so there is no split/masking to
    reproduce -- it cannot drift from the trained run.
  - char-SVM (LinearSVC, no saved proba) is refit on CPU over the remaining outlets for the
    TEMPORAL split (deterministic year split) on conditions that need no spaCy (raw, mask_outlet).
    char-SVM mask_entities and the random split are left for the pipeline env (need spaCy lg /
    exact stratified split) -- flagged, not guessed.
  - Majority = largest remaining-class share of the (restricted) test set.

Outputs to analysis_<panel>_drop<TAG>/.
Run:  python3 gen_drop_outlet_recompute.py n10 --drop "L'Humanité"
      python3 gen_drop_outlet_recompute.py n15 --drop "L'Humanité"
"""
import sys, re, json, glob, argparse
from pathlib import Path
from split_utils import temporal_split  # canonical paper split (carves 10% late-2024 val)
import numpy as np, pandas as pd, warnings
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent  # package root

PROBA_MODELS = ["camembert", "charcnn_pure", "charcnn_word", "word_lr"]
CONDS = ["raw", "mask_outlet", "mask_entities"]
SPLITS = ["random", "temporal"]


def _runs(panel, model):
    return glob.glob(str(ROOT / f"results/{panel}/runs/proba/{panel}_{model}_pooled_ALL_headline_only_*.npy"))


def _parse(fname):
    # n{P}_{model}_pooled_ALL_headline_only_{cond}_{split}_s{seed}[_lr...]
    b = Path(fname).stem
    m = re.search(r"headline_only_(raw|mask_outlet|mask_entities)_(random|temporal)_s(\d+)", b)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)


def proba_eval(panel, drop, keep_order):
    """Return long df: model,preprocess,split,seed,acc,macro_f1,top2 + per-row predictions for headline cell."""
    rows, headline_rows = [], None
    for model in PROBA_MODELS:
        for f in _runs(panel, model):
            cond, split, seed = _parse(f)
            if cond is None:
                continue
            rid = Path(f).stem
            meta = json.load(open(ROOT / f"results/{panel}/runs/metrics/{rid}.json"))
            labels = meta["proba_labels"]
            if drop not in labels:
                continue
            di = labels.index(drop)
            keep_idx = [i for i in range(len(labels)) if i != di]
            keep_labels = [labels[i] for i in keep_idx]
            proba = np.load(f)
            pr = pd.read_parquet(ROOT / f"results/{panel}/runs/preds/{rid}.parquet")
            true = pr["true"].to_numpy()
            keep_rows = true != drop
            P = proba[keep_rows][:, keep_idx]
            tsub = true[keep_rows]
            pred = np.array(keep_labels)[P.argmax(1)]
            # top-2 over restricted columns
            order = np.argsort(-P, axis=1)[:, :2]
            top2 = np.array([tsub[i] in [keep_labels[order[i, 0]], keep_labels[order[i, 1]]] for i in range(len(tsub))]).mean()
            acc = (tsub == pred).mean()
            mf1 = f1_score(tsub, pred, labels=keep_labels, average="macro", zero_division=0)
            rows.append(dict(model=model, preprocess=cond, split=split, seed=int(seed),
                             acc=acc, macro_f1=mf1, top2=top2, n=len(tsub)))
            # save per-row for the headline cell (camembert temporal mask_entities)
            if model == "camembert" and split == "temporal" and cond == "mask_entities":
                hr = pd.DataFrame({"true": tsub, "pred": pred,
                                   "category": pr.loc[keep_rows, "category"].to_numpy(),
                                   "correct": tsub == pred, "seed": int(seed)})
                headline_rows = hr if headline_rows is None else pd.concat([headline_rows, hr])
    return pd.DataFrame(rows), headline_rows


def majority(panel, drop):
    out = {}
    for split in SPLITS:
        f = glob.glob(str(ROOT / f"results/{panel}/runs/preds/{panel}_camembert_pooled_ALL_headline_only_raw_{split}_s42*"))
        if not f:
            f = glob.glob(str(ROOT / f"results/{panel}/runs/preds/{panel}_camembert_pooled_ALL_headline_only_mask_entities_{split}_s42*"))
        pr = pd.read_parquet(f[0])
        t = pr.loc[pr["true"] != drop, "true"]
        out[split] = t.value_counts(normalize=True).iloc[0]
    return out


def char_svm_temporal(panel, drop, keep_order):
    """Refit char-SVM on the 10/14 outlets, temporal split, raw + mask_outlet (no spaCy)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    OUTLET_DOMAINS = {  # identical to pipeline
        "Le Monde":["lemonde","le monde"],"Le Figaro":["lefigaro","le figaro"],
        "Les Echos":["lesechos","les echos","les Échos"],"La Croix":["la-croix","la croix"],
        "Ouest-France":["ouest-france","ouest france"],"Le Parisien":["leparisien","le parisien"],
        "TF1 INFO":["tf1info","tf1 info","tf1","lci"],"Libération":["liberation","libération","libe","libé"],
        "Franceinfo":["franceinfo","france info","francetvinfo"],"BFMTV":["bfmtv","bfm tv","bfm"],
        "L'Humanité":["l'humanité","l humanite","humanite","huma"],
        "20 Minutes":["20 minutes","20minutes","20 min"],"Le Point":["le point","lepoint"],
        "Le Nouvel Obs":["nouvel obs","nouvelobs","l'obs","le nouvel observateur"],
        "L'Express":["l'express","lexpress","express"],"Le HuffPost":["huffpost","huffington","le huffpost"],
        "RFI":["rfi","radio france internationale"],"CNEWS":["cnews","c news","cnews.fr"]}
    pats=[re.escape(v) for vs in OUTLET_DOMAINS.values() for v in vs]
    ORE=re.compile(r"(?i)\b("+"|".join(sorted(pats,key=len,reverse=True))+r")\b")
    URE=re.compile(r"https?://\S+|www\.\S+")
    mo=lambda t: ORE.sub("[OUTLET]",URE.sub("[URL]",t))

    d=pd.read_parquet(ROOT/f"results/{panel}/sampled_dataset_{panel}.parquet")[["title","outlet","year","dt"]].dropna(subset=["title"])
    d=d[d.outlet!=drop]
    tr,te=temporal_split(d)  # canonical paper split (carves 10% late-2024 val out of train)
    labels=keep_order
    res={}
    for cond,fn in [("raw",lambda s:s),("mask_outlet",mo)]:
        vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=200000,sublinear_tf=True,min_df=2)
        Xtr=vec.fit_transform(tr["title"].map(fn)); clf=LinearSVC(C=1.0,class_weight="balanced"); clf.fit(Xtr,tr["outlet"])
        pred=clf.predict(vec.transform(te["title"].map(fn)))
        res[cond]=dict(acc=(pred==te["outlet"].to_numpy()).mean(),
                       macro_f1=f1_score(te["outlet"],pred,labels=labels,average="macro",zero_division=0),n=len(te))
    return res


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("panel"); ap.add_argument("--drop",default="L'Humanité")
    ap.add_argument("--no-charsvm",action="store_true")
    a=ap.parse_args()
    panel,drop=a.panel,a.drop
    # remaining outlet order = panel order minus drop (read from any camembert run's proba_labels)
    runs=_runs(panel,"camembert")
    if not runs:
        sys.exit(f"No class-probability .npy files found under "
                  f"results/{panel}/runs/proba/ — these are not shipped in the supplementary "
                  f"package (available on request; see README Level-2 table). This script "
                  f"cannot run out of the box.")
    anyf=runs[0]; rid=Path(anyf).stem
    labels=json.load(open(ROOT/f"results/{panel}/runs/metrics/{rid}.json"))["proba_labels"]
    keep_order=[l for l in labels if l!=drop]
    chance=1/len(keep_order)
    tag=re.sub(r"[^A-Za-z]","",drop)[:4]
    out=ROOT/f"analysis_{panel}_drop{tag}"; out.mkdir(exist_ok=True)
    print(f"== panel {panel}: drop '{drop}' -> N={len(keep_order)} (chance={chance:.4f}) -> {out.name}/ ==")

    longdf,headline=proba_eval(panel,drop,keep_order)
    maj=majority(panel,drop)

    # aggregate main table (mean over seeds)
    agg=(longdf.groupby(["model","preprocess","split"])
         .agg(acc_mean=("acc","mean"),acc_std=("acc","std"),f1_mean=("macro_f1","mean"),
              f1_std=("macro_f1","std"),top2_mean=("top2","mean"),n_seeds=("acc","size")).reset_index().fillna(0))
    # majority + char-svm temporal
    extra=[]
    for split in SPLITS:
        extra.append(dict(model="majority",preprocess="raw",split=split,acc_mean=maj[split],acc_std=0,
                          f1_mean=np.nan,f1_std=0,top2_mean=np.nan,n_seeds=1))
    if not a.no_charsvm:
        try:
            cs=char_svm_temporal(panel,drop,keep_order)
            for cond,r in cs.items():
                extra.append(dict(model="char_svm",preprocess=cond,split="temporal",acc_mean=r["acc"],acc_std=0,
                                  f1_mean=r["macro_f1"],f1_std=0,top2_mean=np.nan,n_seeds=1))
            print(f"  char-SVM temporal refit OK: raw={cs['raw']['acc']:.3f} mask_outlet={cs['mask_outlet']['acc']:.3f}")
            print("  NOTE: char-SVM mask_entities + all char-SVM random NOT recomputed here (need spaCy lg / exact random split) -> run in pipeline env.")
        except Exception as e:
            print("  char-SVM refit skipped:",e)
    agg=pd.concat([agg,pd.DataFrame(extra)],ignore_index=True).sort_values(["split","acc_mean"],ascending=[True,False])
    agg.to_csv(out/"table2_main_results.csv",index=False)

    # headline-cell derived outputs (camembert temporal mask_entities, mean over seeds for accuracy; per-row from seed 42 for per-outlet/confusion)
    h42=headline[headline.seed==42]
    # per-outlet
    p,r,f1,sup=precision_recall_fscore_support(h42["true"],h42["pred"],labels=keep_order,zero_division=0)
    acc_o=[h42[h42.true==o]["correct"].mean() for o in keep_order]
    pd.DataFrame({"outlet":keep_order,"precision":p,"recall":r,"f1":f1,"accuracy":acc_o,"support":sup})\
      .sort_values("f1",ascending=False).to_csv(out/"table3_per_outlet_temporal.csv",index=False)
    # within-category
    wc=h42.groupby("category")["correct"].agg(["mean","size"]).reset_index().rename(columns={"mean":"accuracy","size":"support"})
    wc.to_csv(out/"breakdown_by_category.csv",index=False)
    # confusion
    order=pd.DataFrame({"o":keep_order,"f1":f1}).sort_values("f1",ascending=False)["o"].tolist()
    cm=confusion_matrix(h42["true"],h42["pred"],labels=order)
    pd.DataFrame(cm,index=order,columns=order).to_csv(out/"confusion_camembert_temporal_maskent.csv")

    # headline summary
    hcell=agg[(agg.model=="camembert")&(agg.split=="temporal")&(agg.preprocess=="mask_entities")].iloc[0]
    print(f"\n  HEADLINE (CamemBERT temporal mask_entities): acc={hcell.acc_mean:.3f}+/-{hcell.acc_std:.3f} "
          f"macroF1={hcell.f1_mean:.3f} top2={hcell.top2_mean:.3f}  => {hcell.acc_mean/chance:.1f}x chance")
    print(f"  wrote table2_main_results.csv, table3_per_outlet_temporal.csv, breakdown_by_category.csv, confusion_*.csv")
    print("\n=== MAIN TABLE (acc_mean) ===")
    piv=agg.pivot_table(index="model",columns=["split","preprocess"],values="acc_mean")
    print(piv.to_string(float_format=lambda x:f"{x:.3f}"))


if __name__ == "__main__":
    main()
