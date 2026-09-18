"""
Tier-0 residual decomposition: after stripping the NAMED surface form of a headline, does any
outlet signal survive? Whatever survives is, by construction, the "deeper" (lexical / word-choice /
semantic) part -- this bounds how much room there is for tone/framing/stance before investing in a
matched-event framing study. Zero new data; reuses the paper's exact char-SVM.

Surface-stripping ladder on the SAME char-SVM (TF-IDF char_wb 3-5, LinearSVC, balanced) and the SAME
temporal split (train <=2024, test 2025) as Table 2, so each rung is comparable to 0.496/0.427:

  raw                        anchor (~0.496)
  typo_normalized            quotes/dashes/ellipses -> ASCII, trailing punct stripped (~0.427)
  surface_stripped           lowercase + remove ALL punctuation + remove digits + collapse ws
                             -> residual = lexical word-choice; UPPER bound on non-surface signal
  surface_stripped_noproper  also drop tokens Capitalized in the original (crude proper-noun/entity
                             proxy) -> residual = common-word register; LOWER bound. Gap to the row
                             above ~ how much of the residual is proper-noun/topic leakage.

Plus a word-level lexical probe (TF-IDF word 1-2gram LR) on the stripped text + top distinctive words
per outlet, to NAME the residual (genuine register words vs place/actor names).

Run:    python3 gen_residual_decomposition.py n10
Writes: analysis_<panel>/residual_decomposition.csv
        analysis_<panel>/residual_top_words_by_outlet.csv
"""
import sys, re, argparse
from pathlib import Path
from split_utils import temporal_split  # canonical paper split (carves 10% late-2024 val)
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent  # package root
TRAIN_MAX_YEAR, TEST_YEAR = 2024, 2025  # split itself comes from split_utils.temporal_split

# --- reuse the typo-normalization map from gen_typo_normalization_ablation.py ---
_QUOTES = {"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‟": '"', "‹": '"', "›": '"'}
_APOS = {"’": "'", "‘": "'", "‛": "'", "ʼ": "'", "´": "'", "`": "'"}
_DASH = {"–": "-", "—": "-", "―": "-"}
_TRANS = {**_QUOTES, **_APOS, **_DASH, "…": "..."}
_TRAIL = re.compile(r"[\s\.\!\?:;,\|…-]+$")
_PUNCT = re.compile(r"[^\wÀ-ÿ\s]", re.UNICODE)   # anything not letter/number/underscore/space
_DIGIT = re.compile(r"\d+")
_CAPTOK = re.compile(r"\b[A-ZÀ-Þ][\wÀ-ÿ'’-]*", re.UNICODE)  # token starting with a capital


def normalize_typo(t: str) -> str:
    for k, v in _TRANS.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t).strip()
    return _TRAIL.sub("", t)


def surface_strip(t: str, drop_proper=False) -> str:
    if drop_proper:                      # remove Capitalized tokens BEFORE lowercasing (entity proxy)
        t = _CAPTOK.sub(" ", t)
    t = t.lower()
    t = _PUNCT.sub(" ", t)               # remove all punctuation
    t = _DIGIT.sub(" ", t)               # remove digits
    return re.sub(r"\s+", " ", t).strip()


def ner_mask_then_strip(texts, model):
    """Real entity mask: remove spaCy PER/LOC/ORG/MISC spans, THEN surface_strip. Returns list."""
    import spacy
    nlp = spacy.load(model, disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    out = []
    for doc in nlp.pipe((str(t) for t in texts), batch_size=512):
        s = doc.text
        for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
            s = s[:ent.start_char] + " " + s[ent.end_char:]   # drop the entity span
        out.append(surface_strip(s, drop_proper=False))
    return out


def fit_eval_char(train_txt, y_train, test_txt, y_test, labels, max_features, ngram_max):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, ngram_max),
                          max_features=max_features, sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(train_txt)
    clf = LinearSVC(class_weight="balanced")
    clf.fit(Xtr, y_train)
    pred = clf.predict(vec.transform(test_txt))
    return accuracy_score(y_test, pred), f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)


def word_probe(train_txt, y_train, test_txt, y_test, labels):
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=60000,
                          sublinear_tf=True, min_df=3)
    Xtr = vec.fit_transform(train_txt)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")  # multinomial is the default
    clf.fit(Xtr, y_train)
    pred = clf.predict(vec.transform(test_txt))
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)
    # top distinctive words per outlet
    feats = np.array(vec.get_feature_names_out())
    rows = []
    for i, o in enumerate(clf.classes_):
        top = feats[np.argsort(clf.coef_[i])[::-1][:12]]
        rows.append({"outlet": o, "top_words": ", ".join(top)})
    return acc, f1, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--spacy", default=None,
                    help="spaCy model (e.g. fr_core_news_lg) -> adds a real NER-mask residual rung")
    a = ap.parse_args()
    df = pd.read_parquet(ROOT / f"results/{a.panel}/sampled_dataset_{a.panel}.parquet")[
        ["title", "outlet", "year", "dt"]].dropna(subset=["title"])
    tr, te = temporal_split(df)  # canonical paper split (carves 10% late-2024 val out of train)
    labels = sorted(df["outlet"].unique()); chance = 1 / len(labels)
    out = ROOT / f"analysis_{a.panel}"; out.mkdir(exist_ok=True)
    (out / "residual_top_words_by_outlet.csv").unlink(missing_ok=True)  # fresh each run (avoid append dupes)
    print(f"panel {a.panel}: train={len(tr):,} test={len(te):,} outlets={len(labels)} chance={chance:.3f}",
          flush=True)

    conds = [
        ("raw", lambda s: s),
        ("typo_normalized", normalize_typo),
        ("surface_stripped", lambda s: surface_strip(s, drop_proper=False)),
        ("surface_stripped_noproper", lambda s: surface_strip(s, drop_proper=True)),
    ]
    rows = []
    raw_acc = None
    for cond, fn in conds:
        trx, tex = tr["title"].map(fn), te["title"].map(fn)
        acc, f1 = fit_eval_char(trx, tr["outlet"], tex, te["outlet"], labels, a.max_features, a.ngram_max)
        if cond == "raw":
            raw_acc = acc
        rows.append({"condition": cond, "model": "char_svm", "accuracy": round(acc, 4),
                     "macro_f1": round(f1, 4), "x_chance": round(acc / chance, 2),
                     "pct_raw_signal_retained": round(100 * (acc - chance) / (raw_acc - chance), 1) if raw_acc else None})
        pd.DataFrame(rows).to_csv(out / "residual_decomposition.csv", index=False)  # incremental
        print(f"  char-SVM {cond:26s} acc={acc:.4f} f1={f1:.4f} {acc/chance:.2f}x "
              f"({rows[-1]['pct_raw_signal_retained']}% of raw signal)", flush=True)

    # word-level lexical probe on the two stripped variants (names the residual)
    for cond, fn in [("surface_stripped", lambda s: surface_strip(s, False)),
                     ("surface_stripped_noproper", lambda s: surface_strip(s, True))]:
        trx, tex = tr["title"].map(fn), te["title"].map(fn)
        acc, f1, topdf = word_probe(trx, tr["outlet"], tex, te["outlet"], labels)
        rows.append({"condition": cond, "model": "word_lr", "accuracy": round(acc, 4),
                     "macro_f1": round(f1, 4), "x_chance": round(acc / chance, 2),
                     "pct_raw_signal_retained": round(100 * (acc - chance) / (raw_acc - chance), 1)})
        pd.DataFrame(rows).to_csv(out / "residual_decomposition.csv", index=False)
        topdf.insert(0, "condition", cond)
        hdr = not (out / "residual_top_words_by_outlet.csv").exists()
        topdf.to_csv(out / "residual_top_words_by_outlet.csv", mode="a", header=hdr, index=False)
        print(f"  word-LR  {cond:26s} acc={acc:.4f} f1={f1:.4f} {acc/chance:.2f}x", flush=True)

    # --- optional: real spaCy NER-mask residual (the clean residual, no brand/proper-noun leak) ---
    if a.spacy:
        print(f"  [spaCy {a.spacy}] masking entities (PER/LOC/ORG/MISC) then surface-stripping ...", flush=True)
        trx = ner_mask_then_strip(tr["title"], a.spacy)
        tex = ner_mask_then_strip(te["title"], a.spacy)
        acc, f1 = fit_eval_char(trx, tr["outlet"], tex, te["outlet"], labels, a.max_features, a.ngram_max)
        rows.append({"condition": "surface_stripped_nermask", "model": "char_svm", "accuracy": round(acc, 4),
                     "macro_f1": round(f1, 4), "x_chance": round(acc / chance, 2),
                     "pct_raw_signal_retained": round(100 * (acc - chance) / (raw_acc - chance), 1)})
        pd.DataFrame(rows).to_csv(out / "residual_decomposition.csv", index=False)
        print(f"  char-SVM surface_stripped_nermask   acc={acc:.4f} f1={f1:.4f} {acc/chance:.2f}x "
              f"({rows[-1]['pct_raw_signal_retained']}% of raw signal)", flush=True)
        wacc, wf1, topdf = word_probe(trx, tr["outlet"], tex, te["outlet"], labels)
        rows.append({"condition": "surface_stripped_nermask", "model": "word_lr", "accuracy": round(wacc, 4),
                     "macro_f1": round(wf1, 4), "x_chance": round(wacc / chance, 2),
                     "pct_raw_signal_retained": round(100 * (wacc - chance) / (raw_acc - chance), 1)})
        pd.DataFrame(rows).to_csv(out / "residual_decomposition.csv", index=False)
        topdf.insert(0, "condition", "surface_stripped_nermask")
        topdf.to_csv(out / "residual_top_words_by_outlet.csv", mode="a", header=False, index=False)
        print(f"  word-LR  surface_stripped_nermask   acc={wacc:.4f} f1={wf1:.4f} {wacc/chance:.2f}x", flush=True)

    print(f"\nWrote {out}/residual_decomposition.csv and residual_top_words_by_outlet.csv", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
