"""
What the editorial fingerprint is MADE OF -- an interpretable style profile.

Motivation (/ author "so what?" concern): the char-SVM and CamemBERT models
prove outlets are separable from headline *form*, but they are black boxes -- we never tell
the reader, in nameable human terms, WHAT the stylistic dimensions are. This script answers
that with two committed, reproducible, zero-GPU analyses computed from the headline text and
outlet labels we already have (results/<panel>/sampled_dataset_<panel>.parquet):

  (1) PROFILE.  ~20 named, countable features per headline (typography, structure, register).
      We report each outlet's rate/mean on each feature -> a human-readable per-outlet table and
      a z-scored heatmap figure. This is the explanation: e.g. "Les Echos headlines are short and
      numeral-dense; Ouest-France is colon/topic-prefixed; broadcasters carry all-caps format tags."

  (2) SIGNAL.  Train a multinomial logistic regression on ONLY those ~20 interpretable features
      (no n-grams, no words) and report its accuracy. The SAME temporal split as the paper
      (train 2021-2024, test 2025). If a handful of nameable dimensions recover the outlet well
      above chance, we can say those named dimensions account for a real share of the surface
      signal -- bridging the uninterpretable char-SVM (0.496) to human-readable style. We report
      the fraction of the char-SVM-over-chance gap that the interpretable features recover.

Optional (3): if spaCy + a French model (fr_core_news_lg/sm) is importable, we ADD three
linguistic register features (verbless/nominal-headline rate, proper-noun density, NER density)
to the profile and the LR. They are skipped cleanly if spaCy is unavailable, so the script always
runs; the paper number should be regenerated where fr_core_news_lg is installed for the full set.

Usage:   python3 gen_style_profile.py n10 [--spacy fr_core_news_lg]
Reads:   results/<panel>/sampled_dataset_<panel>.parquet  (cols: title, outlet, category, year)
Writes:  analysis_<panel>/style_profile_by_outlet.csv      (outlet x feature, raw rate/mean)
         analysis_<panel>/style_profile_zscores.csv        (outlet x feature, z across outlets)
         analysis_<panel>/style_lr_results.csv             (acc/macroF1/top2/xchance per split)
         analysis_<panel>/style_lr_coefficients.csv        (|coef| weight per feature, temporal)
         analysis_<panel>/style_outlet_signatures.csv      (top + / - distinctive features/outlet)
         analysis_<panel>/figures/fig_style_heatmap.pdf
"""
import sys, re, argparse, warnings
from pathlib import Path
from split_utils import temporal_split  # canonical paper split (carves 10% late-2024 val)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent  # package root

import matplotlib
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

# ----------------------------------------------------------------------------- features
# A small closed-class French function-word list (articles, prepositions, conjunctions,
# pronouns, common auxiliaries) -- used only for the function-word-ratio register feature.
FUNCTION_WORDS = set("""
le la les un une des du de d l au aux a à et ou ni mais or car donc que qui quoi dont où
ce cet cette ces son sa ses leur leurs mon ma mes ton ta tes notre nos votre vos
je tu il elle on nous vous ils elles me te se lui en y
dans sur sous par pour avec sans chez vers entre contre depuis pendant avant après
est sont être été a ont avoir avait sera seront ne pas plus très bien comme si
""".split())

GUILLEMETS = re.compile(r"[«»]")
ELLIPSIS = re.compile(r"(…|\.\.\.)")
EMDASH = re.compile(r"[–—]")           # en/em dash
SPACED_HYPHEN = re.compile(r"\s-\s")             # " - " separator
ALLCAPS_TOK = re.compile(r"\b[A-ZÀ-Þ]{2,}\b")    # ABC, EN, DIRECT, VIDEO, PSG ...
WORD = re.compile(r"\b[\wÀ-ÿ'’-]+\b", re.UNICODE)
DIGIT = re.compile(r"\d")
LISTICLE = re.compile(r"^\s*\d+\b")              # "3 choses ...", "5 raisons ..."

FEATURE_ORDER = [
    # typography
    "has_guillemets", "has_ellipsis", "has_colon", "colon_lead", "has_question",
    "has_exclaim", "has_percent", "has_dash", "has_dquote", "has_paren",
    # structure / length
    "char_len", "word_count", "mean_word_len", "digit_density", "has_number",
    "listicle_open", "n_allcaps_tok", "upper_ratio", "n_comma",
    # register (lexicon only)
    "function_word_ratio",
]
BINARY = {"has_guillemets", "has_ellipsis", "has_colon", "colon_lead", "has_question",
          "has_exclaim", "has_percent", "has_dash", "has_dquote", "has_paren",
          "has_number", "listicle_open"}


def headline_features(t: str) -> dict:
    t = "" if t is None else str(t)
    L = len(t)
    words = WORD.findall(t)
    nw = len(words)
    letters = [c for c in t if c.isalpha()]
    n_upper = sum(1 for c in letters if c.isupper())
    n_digit = sum(1 for c in t if c.isdigit())
    fw = sum(1 for w in words if w.lower() in FUNCTION_WORDS)
    head = t[:25]                                   # first ~25 chars for colon-lead detection
    return {
        "has_guillemets": int(bool(GUILLEMETS.search(t))),
        "has_ellipsis":   int(bool(ELLIPSIS.search(t))),
        "has_colon":      int(":" in t),
        "colon_lead":     int(":" in head),
        "has_question":   int("?" in t),
        "has_exclaim":    int("!" in t),
        "has_percent":    int("%" in t),
        "has_dash":       int(bool(EMDASH.search(t) or SPACED_HYPHEN.search(t))),
        "has_dquote":     int('"' in t or "“" in t or "”" in t),
        "has_paren":      int("(" in t),
        "char_len":       float(L),
        "word_count":     float(nw),
        "mean_word_len":  float(np.mean([len(w) for w in words])) if nw else 0.0,
        "digit_density":  (n_digit / L) if L else 0.0,
        "has_number":     int(bool(DIGIT.search(t))),
        "listicle_open":  int(bool(LISTICLE.search(t))),
        "n_allcaps_tok":  float(len(ALLCAPS_TOK.findall(t))),
        "upper_ratio":    (n_upper / len(letters)) if letters else 0.0,
        "n_comma":        float(t.count(",")),
        "function_word_ratio": (fw / nw) if nw else 0.0,
    }


def add_spacy_features(df, model):
    """Optional register features via spaCy: verbless/nominal rate, proper-noun & entity density."""
    try:
        import spacy
        nlp = spacy.load(model, disable=["lemmatizer"])
    except Exception as e:
        print(f"[spaCy skipped: {e}] -- regex features only.")
        return df, []
    print(f"[spaCy {model} loaded] adding verbless/propn/entity-density features ...")
    texts = df["title"].fillna("").astype(str).tolist()
    verbless, propn_den, ent_den = [], [], []
    for doc in nlp.pipe(texts, batch_size=512):
        toks = [t for t in doc if not t.is_space]
        n = len(toks) or 1
        has_finite_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
        verbless.append(int(not has_finite_verb))
        propn_den.append(sum(1 for t in doc if t.pos_ == "PROPN") / n)
        ent_den.append(sum(len(e) for e in doc.ents) / n)
    df["verbless_nominal"] = verbless
    df["propn_density"] = propn_den
    df["entity_density"] = ent_den
    return df, ["verbless_nominal", "propn_density", "entity_density"]


# ----------------------------------------------------------------------------- LR signal
def run_lr(df, feat_cols, split, panel):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    rng = np.random.RandomState(42)
    if split == "temporal":
        tr, te = temporal_split(df)  # canonical paper split (carves 10% late-2024 val out of train)
    else:  # random 80/20 stratified-ish by index shuffle
        idx = rng.permutation(len(df)); cut = int(0.8 * len(df))
        tr = df.iloc[idx[:cut]]; te = df.iloc[idx[cut:]]
    Xtr, Xte = tr[feat_cols].values, te[feat_cols].values
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    ytr, yte = tr["outlet"].values, te["outlet"].values
    clf = LogisticRegression(max_iter=2000, C=1.0)  # multinomial is the default in current sklearn
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    classes = clf.classes_
    pred = classes[proba.argmax(1)]
    top2 = np.mean([yte[i] in classes[np.argsort(proba[i])[-2:]] for i in range(len(yte))])
    acc = accuracy_score(yte, pred); mf1 = f1_score(yte, pred, average="macro")
    return dict(split=split, n_train=len(tr), n_test=len(te), accuracy=acc,
                macro_f1=mf1, top2_acc=float(top2)), clf, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="n10")
    ap.add_argument("--spacy", default=None, help="spaCy model, e.g. fr_core_news_lg (optional)")
    a = ap.parse_args()
    panel = a.panel
    out = ROOT / f"analysis_{panel}"; (out / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / f"results/{panel}/sampled_dataset_{panel}.parquet")[
        ["title", "outlet", "category", "year", "dt"]].copy()
    n_out = df["outlet"].nunique(); chance = 1 / n_out
    print(f"== Style profile, panel {panel}: {len(df):,} headlines, {n_out} outlets, chance={chance:.3f} ==\n")

    feats = pd.DataFrame([headline_features(t) for t in df["title"]], index=df.index)
    df = pd.concat([df, feats], axis=1)
    feat_cols = list(FEATURE_ORDER)
    if a.spacy:
        df, extra = add_spacy_features(df, a.spacy); feat_cols += extra

    # (1) per-outlet profile (mean/rate)
    prof = df.groupby("outlet")[feat_cols].mean().round(4)
    prof = prof.loc[prof.index.sort_values()]
    prof.to_csv(out / "style_profile_by_outlet.csv")
    # z-scores across outlets (how distinctive each outlet is on each feature)
    z = (prof - prof.mean()) / prof.std(ddof=0)
    z.round(3).to_csv(out / "style_profile_zscores.csv")

    # per-outlet signatures: top +/- distinctive features
    sig_rows = []
    for o in z.index:
        s = z.loc[o].sort_values()
        hi = ", ".join(f"{f}(+{z.loc[o,f]:.1f})" for f in s.index[::-1][:3])
        lo = ", ".join(f"{f}({z.loc[o,f]:.1f})" for f in s.index[:3])
        sig_rows.append({"outlet": o, "most_distinctive_high": hi, "most_distinctive_low": lo})
    pd.DataFrame(sig_rows).to_csv(out / "style_outlet_signatures.csv", index=False)

    print("(1) PROFILE -- per-outlet rates/means written to style_profile_by_outlet.csv")
    show = ["char_len", "word_count", "digit_density", "has_colon", "colon_lead",
            "has_guillemets", "n_allcaps_tok", "has_question", "function_word_ratio"]
    print(prof[show].to_string(float_format=lambda x: f"{x:.3f}"), "\n")

    # (2) interpretable-feature LR
    print("(2) SIGNAL -- multinomial LR on the interpretable features only:")
    res_rows = []
    charsvm_raw = {"temporal": 0.496, "random": None}  # paper char-SVM raw temporal (for gap %)
    coef_saved = False
    for split in ["temporal", "random"]:
        r, clf, classes = run_lr(df, feat_cols, split, panel)
        x = r["accuracy"] / chance
        r["x_chance"] = round(x, 2)
        cs = charsvm_raw.get(split)
        if cs:
            r["frac_charsvm_gap"] = round((r["accuracy"] - chance) / (cs - chance), 3)
        res_rows.append(r)
        msg = (f"  {split:8s}: acc {r['accuracy']:.3f}  macroF1 {r['macro_f1']:.3f}  "
               f"top2 {r['top2_acc']:.3f}  = {x:.1f}x chance")
        if cs:
            msg += f"  ({r['frac_charsvm_gap']*100:.0f}% of char-SVM-raw gap, {cs:.3f})"
        print(msg)
        if split == "temporal" and not coef_saved:
            mag = np.abs(clf.coef_).mean(0)  # mean |coef| across outlets per feature
            cf = pd.DataFrame({"feature": feat_cols, "mean_abs_coef": mag}).sort_values(
                "mean_abs_coef", ascending=False)
            cf.to_csv(out / "style_lr_coefficients.csv", index=False)
            coef_saved = True
            print("    top features by |coef| (temporal): " +
                  ", ".join(cf.feature.head(6).tolist()))
    pd.DataFrame(res_rows).to_csv(out / "style_lr_results.csv", index=False)

    # (3) heatmap figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        zc = z.clip(-2.5, 2.5)
        fig, ax = plt.subplots(figsize=(0.45 * len(feat_cols) + 2, 0.4 * len(zc) + 1.5))
        im = ax.imshow(zc.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_xticks(range(len(feat_cols))); ax.set_xticklabels(feat_cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(zc))); ax.set_yticklabels(zc.index, fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("z across outlets", fontsize=7)
        ax.set_title(f"Interpretable headline-style profile (z-scores), {panel}", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "figures" / "fig_style_heatmap.pdf")
        print(f"\n[figure] {out}/figures/fig_style_heatmap.pdf")
    except Exception as e:
        print(f"[figure skipped: {e}]")

    print(f"\nWrote outputs to {out}/")


if __name__ == "__main__":
    main()
