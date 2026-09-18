#!/usr/bin/env python3
"""
Unified LLM outlet-attribution baseline: N models x {zero-shot, k-shot} x seeds.

Supersedes gen_llm_zeroshot_baseline.py + gen_llm_fewshot_baseline.py. Same task
(given one 2025 headline, name the publishing outlet among ten), same balanced
sample and metrics, so every number is directly comparable to the paper's temporal
column (word-LR .326 | char-SVM .496 | CharCNN .493 | CamemBERT .576 entity-masked
/ .615 raw; chance .100).

This runs three model families under matched zero-shot AND few-shot in one harness, so
the LLM baseline is balanced across model families and shot settings.

Providers (OpenAI-compatible or custom), selected per --model via a small registry:
    regolo   -> api.regolo.ai            (gpt-oss-20b, gpt-oss-120b, ...)
    custom     -> your-openai-compatible-endpoint/v1/chat/completions    (custom endpoint; --model carries a pipeline id)
    hf       -> router.huggingface.co    (openai/gpt-oss-120b, meta-llama/..., mistralai/...)
    groq     -> api.groq.com             (llama-3.3-70b-versatile, ...)
    openrouter -> openrouter.ai

Keys are read from the environment (never hard-coded):
    REGOLO_API_KEY / REGOLO_KEY, CUSTOM_LLM_API_KEY, HF_TOKEN, GROQ_API_KEY, OPENROUTER_API_KEY

Models are addressed through an OpenAI-compatible endpoint by a model or pipeline id. Give each one a friendly label (label=pipeline_id) so it shows up readably in the paper
table. (Other providers exist in the code but are not needed; ignore them.)

Set the key once:
    export CUSTOM_LLM_API_KEY=sk-...

Examples:
    # three models, all custom-endpoint models, zero-shot + 5-shot, three seeds, 200 headlines/outlet
    # (<model_pipeline_id> is a placeholder model id)
    python3 gen_llm_baseline.py \
        --model mistral-small=<model_pipeline_id> \
        --model llama-3.3-70b=pipe_<llama_pipeline_id> \
        --model gpt-oss-120b=pipe_<gptoss_pipeline_id> \
        --shots 0,5 --seeds 42,13,1234 --per-outlet 200

    # single the custom endpoint model (bare pipeline id also works; defaults to custom)
    python3 gen_llm_baseline.py --model mistral-small=<model_pipeline_id> --shots 0,5,10 --seeds 42

Outputs (under llm/ at the package root):
    <tag>_<k>shot_seed<seed>_predictions.csv   raw per-headline predictions (resumable)
    llm_baseline_aggregate.json                mean/CI per (model, k) over seeds
    llm_baseline_table.tex                      paste-ready appendix table body

Nothing here needs a local DB or the raw corpus beyond results/n10/sampled_dataset_n10.parquet,
which is the released sampled panel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Task constants (identical to the two scripts this replaces)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent  # package root
DATA = ROOT / "results" / "n10" / "sampled_dataset_n10.parquet"
OUT_DIR = ROOT / "llm"
CATEGORIES = ["International", "Politique", "Économie"]
OUTLETS = ["Le Monde", "Le Figaro", "Les Echos", "Libération", "La Croix",
           "Le Parisien", "Franceinfo", "BFMTV", "TF1 INFO", "Ouest-France"]
CHANCE = 1.0 / len(OUTLETS)

ALIASES = {
    "les échos": "Les Echos", "les echos": "Les Echos", "lesechos": "Les Echos",
    "liberation": "Libération", "libé": "Libération", "libe": "Libération",
    "france info": "Franceinfo", "franceinfo": "Franceinfo", "france-info": "Franceinfo",
    "tf1": "TF1 INFO", "tf1info": "TF1 INFO", "tf1 info": "TF1 INFO",
    "ouest france": "Ouest-France", "ouest-france": "Ouest-France",
    "bfm": "BFMTV", "bfmtv": "BFMTV", "bfm tv": "BFMTV",
    "le monde": "Le Monde", "le figaro": "Le Figaro", "la croix": "La Croix",
    "le parisien": "Le Parisien",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


NORM_OUTLET = {norm(o): o for o in OUTLETS}
NORM_ALIAS = {norm(k): v for k, v in ALIASES.items()}


def parse_outlet(text: str | None) -> str | None:
    if not text:
        return None
    t = norm(text)
    if t in NORM_OUTLET:
        return NORM_OUTLET[t]
    if t in NORM_ALIAS:
        return NORM_ALIAS[t]
    hits = [(len(k), v) for k, v in {**NORM_OUTLET, **NORM_ALIAS}.items() if k in t]
    return sorted(hits, reverse=True)[0][1] if hits else None


# --------------------------------------------------------------------------- #
# Prompting (zero-shot when shots_block == "")
# --------------------------------------------------------------------------- #
ZERO = ("You are identifying which French news outlet published a headline.\n"
        "The outlet is exactly one of these ten:\n{opts}\n\n"
        "Given the headline below, reply with ONLY the exact outlet name from the "
        "list above and nothing else.\n\nHeadline: \"{title}\"\nOutlet:")

FEW = ("You are identifying which French news outlet published a headline.\n"
       "The outlet is exactly one of these ten:\n{opts}\n\n"
       "Here are labelled examples:\n{shots}\n"
       "Now classify this headline. Reply with ONLY the exact outlet name from the "
       "list above and nothing else.\n\nHeadline: \"{title}\"\nOutlet:")


def build_prompt(title: str, shots_block: str) -> str:
    opts = "\n".join(f"- {o}" for o in OUTLETS)
    if shots_block:
        return FEW.format(opts=opts, shots=shots_block, title=title)
    return ZERO.format(opts=opts, title=title)


def build_shots(df: pd.DataFrame, shots_per_outlet: int, seed: int) -> str:
    """Exactly k demonstrations per outlet from TRAIN years (<2025), balanced across
    categories as evenly as possible with backfill (so k=5 really gives 5, not 3),
    then shuffled so demonstration order carries no outlet signal.

    Category quota: floor(k/3) each, remainder handed to a seed-shuffled category
    order; any cell short of its quota is backfilled from that outlet's other cells,
    so we deliver k whenever the outlet has >= k train headlines total.
    """
    if shots_per_outlet <= 0:
        return ""
    train = df[df["year"] < 2025]
    rng = np.random.RandomState(seed)
    demos: list[tuple[str, str]] = []
    for o in OUTLETS:
        base, rem = divmod(shots_per_outlet, len(CATEGORIES))
        cats = CATEGORIES[:]
        rng.shuffle(cats)
        quota = {c: base + (1 if i < rem else 0) for i, c in enumerate(cats)}
        picked_ids: set = set()
        chosen: list = []
        # first pass: fill each cell to its quota
        for c in cats:
            sub = train[(train["outlet"] == o) & (train["category"] == c)]
            take = sub.sample(min(quota[c], len(sub)), random_state=rng)
            chosen.extend(take.itertuples())
            picked_ids.update(int(r.article_id) for r in take.itertuples())
        # backfill from any remaining train rows for this outlet up to k
        if len(chosen) < shots_per_outlet:
            pool = train[(train["outlet"] == o) & (~train["article_id"].isin(picked_ids))]
            need = shots_per_outlet - len(chosen)
            extra = pool.sample(min(need, len(pool)), random_state=rng)
            chosen.extend(extra.itertuples())
        for r in chosen[:shots_per_outlet]:
            demos.append((r.title, o))
    rng.shuffle(demos)
    return "\n".join(f'Headline: "{t}"\nOutlet: {o}' for t, o in demos)


def sample_test(df: pd.DataFrame, per_outlet: int, seed: int) -> pd.DataFrame:
    """Balanced 2025 test sample: per_outlet//3 per (outlet x category) so chance == 0.100."""
    test = df[df["year"] == 2025]
    per_cell = max(1, per_outlet // len(CATEGORIES))
    rows = []
    for o in OUTLETS:
        for c in CATEGORIES:
            sub = test[(test["outlet"] == o) & (test["category"] == c)]
            rows.append(sub.sample(min(per_cell, len(sub)), random_state=seed))
    return pd.concat(rows).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
KNOWN_PROVIDERS = {"custom", "regolo", "hf", "groq", "openrouter"}


@dataclass(frozen=True)
class ModelSpec:
    provider: str   # custom (default) | regolo | hf | groq | openrouter
    model: str      # for the custom endpoint: the model/pipeline id (e.g. <model_pipeline_id>)
    tag: str        # display/filesystem-safe label (used in the paper table)

    @staticmethod
    def parse(raw: str) -> "ModelSpec":
        """Accepts, in order of convenience:
            gpt-oss-120b=<model_pipeline_id>     -> custom-endpoint model with display label 'gpt-oss-120b'
            <model_pipeline_id>                  -> custom-endpoint model (bare id; custom is the default)
            custom:<model_pipeline_id>             -> explicit custom endpoint
            regolo:gpt-oss-120b           -> another provider only if you ask for it
        The custom endpoint is the default: anything without a known 'provider:' prefix is a model id.
        """
        raw = raw.strip()
        label: str | None = None
        if "=" in raw:  # optional friendly label -> 'label=spec'
            label, raw = (p.strip() for p in raw.split("=", 1))
        head = raw.split(":", 1)[0].strip().lower()
        if ":" in raw and head in KNOWN_PROVIDERS:
            provider, model = raw.split(":", 1)
            provider, model = provider.strip().lower(), model.strip()
        else:  # no known provider prefix -> custom-endpoint model id
            provider, model = "custom", raw
        tag = label or f"{provider}_{model}"
        tag = re.sub(r"[^A-Za-z0-9._-]", "_", tag)[:80]
        return ModelSpec(provider, model, tag)


PROVIDER_ENV = {
    "regolo": ("REGOLO_API_KEY", "REGOLO_KEY"),
    "custom": ("CUSTOM_LLM_API_KEY",),
    "hf": ("HF_TOKEN",),
    "groq": ("GROQ_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def _token_for(provider: str) -> str | None:
    return next((os.environ[k] for k in PROVIDER_ENV[provider] if os.environ.get(k)), None)


def _endpoint(provider: str) -> tuple[str, str]:
    """Return (url, bearer_token) for a provider; raise if the key is missing."""
    token = _token_for(provider)
    if not token:
        raise RuntimeError(f"provider={provider}: set one of {PROVIDER_ENV[provider]}")
    url = {
        "regolo": "https://api.regolo.ai/v1/chat/completions",
        "custom": "https://your-openai-compatible-endpoint/v1/chat/completions",
        "hf": "https://router.huggingface.co/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }[provider]
    return url, token


def _extract_custom(parsed: object) -> str | None:
    """Best-effort text extraction from the custom endpoint's non-OpenAI response shape."""
    try:
        return parsed["choices"][0]["message"]["content"]  # type: ignore[index]
    except Exception:
        pass
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("response", "output", "result", "answer", "completion",
                    "content", "text", "message", "data", "prediction"):
            if key in parsed:
                got = _extract_custom(parsed[key])
                if got:
                    return got
    if isinstance(parsed, list):
        for item in parsed:
            got = _extract_custom(item)
            if got:
                return got
    return None


def call_model(spec: ModelSpec, url: str, token: str, prompt: str,
               retries: int = 4) -> str:
    messages = [
        {"role": "system", "content": "Return exactly one outlet name and nothing else."},
        {"role": "user", "content": prompt},
    ]
    if spec.provider == "custom":
        payload = {"messages": messages, "pipeline_id": spec.model, "temperature": 0}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}",
                   "User-Agent": "curl/8.7.1", "Accept": "*/*"}  # CF 1010 blocks urllib UA
    else:
        payload = {"model": spec.model, "messages": messages, "temperature": 0}
        if spec.provider in ("regolo", "groq"):
            payload["reasoning_effort"] = "low"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    data = json.dumps(payload).encode("utf-8")
    for a in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            if spec.provider == "custom":
                return (_extract_custom(parsed) or "").strip()
            return (parsed["choices"][0]["message"]["content"] or "").strip()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503):
                time.sleep(2 * (a + 1)); continue
            return f"__HTTP_{exc.code}__"
        except Exception:
            time.sleep(2 * (a + 1))
    return "__ERR__"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def score(res: pd.DataFrame) -> dict:
    res = res.copy()
    res["correct"] = res["gold"] == res["pred"]
    acc = float(res["correct"].mean())
    parsed = float((res["pred"] != "__NONE__").mean())
    f1s = []
    for o in OUTLETS:
        tp = int(((res.pred == o) & (res.gold == o)).sum())
        fp = int(((res.pred == o) & (res.gold != o)).sum())
        fn = int(((res.pred != o) & (res.gold == o)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    macro_f1 = float(np.mean(f1s))
    rng = np.random.default_rng(0)
    c = res["correct"].to_numpy()
    boot = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(2000)]
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {"n": int(len(res)), "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
            "macro_f1": macro_f1, "parse_rate": parsed, "x_chance": acc / CHANCE}


# --------------------------------------------------------------------------- #
# Run one (model, k, seed)
# --------------------------------------------------------------------------- #
def run_cell(spec: ModelSpec, df: pd.DataFrame, shots: int, seed: int,
             per_outlet: int, workers: int, fresh: bool) -> dict:
    url, token = _endpoint(spec.provider)
    shots_block = build_shots(df, shots, seed)
    test = sample_test(df, per_outlet, seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{spec.tag}_{shots}shot_seed{seed}_predictions.csv"

    done: dict[int, tuple[str, str]] = {}
    if out_csv.exists() and not fresh:
        prev = pd.read_csv(out_csv)
        good = prev[prev["pred"] != "__NONE__"]
        done = {int(a): (str(p), str(r)) for a, p, r in
                zip(good["article_id"], good["pred"], good.get("raw", good["pred"]))}
        print(f"    resume {spec.tag} {shots}shot s{seed}: {len(done)} cached")

    todo = [r for r in test.itertuples() if int(r.article_id) not in done]
    fresh_rec: dict[int, tuple[str, str]] = {}
    n_demos = shots_block.count("Outlet:")
    print(f"  {spec.tag} | {shots}-shot (demos={n_demos}) | seed {seed} | "
          f"N={len(test)} | to-do={len(todo)}")

    def infer(row) -> tuple[int, str, str]:
        raw = call_model(spec, url, token, build_prompt(row.title, shots_block))
        return int(row.article_id), (parse_outlet(raw) or "__NONE__"), raw

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(infer, r): r for r in todo}
            for i, f in enumerate(as_completed(futs), 1):
                aid, pred, raw = f.result()
                fresh_rec[aid] = (pred, raw)
                if i % 100 == 0 or i == len(todo):
                    print(f"      {i}/{len(todo)}")

    merged = {**done, **fresh_rec}
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["article_id", "gold", "category", "pred", "raw"])
        for r in test.itertuples():
            pred, raw = merged.get(int(r.article_id), ("__NONE__", ""))
            w.writerow([int(r.article_id), r.outlet, r.category, pred, raw])

    res = pd.read_csv(out_csv)
    s = score(res)
    print(f"    -> acc {s['accuracy']:.3f} [{s['acc_lo']:.3f},{s['acc_hi']:.3f}] "
          f"macro-F1 {s['macro_f1']:.3f} {s['x_chance']:.1f}x (parsed {s['parse_rate']:.0%})")
    return {"provider": spec.provider, "model": spec.model, "tag": spec.tag,
            "shots": shots, "seed": seed, **s}


# --------------------------------------------------------------------------- #
def aggregate(rows: list[dict]) -> list[dict]:
    agg: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        agg.setdefault((r["tag"], r["shots"]), []).append(r)
    out = []
    for (tag, shots), items in sorted(agg.items()):
        accs = [i["accuracy"] for i in items]
        out.append({
            "tag": tag, "model": items[0]["model"], "provider": items[0]["provider"],
            "shots": shots, "n_seeds": len(items),
            "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
            "mean_macro_f1": float(np.mean([i["macro_f1"] for i in items])),
            "mean_x_chance": float(np.mean([i["x_chance"] for i in items])),
            "mean_parse_rate": float(np.mean([i["parse_rate"] for i in items])),
            "seeds": [i["seed"] for i in items],
        })
    return out


def _display_name(a: dict) -> str:
    """Prefer the friendly label (tag); fall back to the model/pipeline id."""
    tag = a.get("tag", "")
    # tags for bare/explicit-provider specs look like 'custom_...'; keep real labels
    if tag and not tag.startswith(("custom_", "regolo_", "hf_", "groq_", "openrouter_")):
        return tag
    return a.get("model", tag).split("/")[-1]


def write_latex(agg: list[dict], path: Path) -> None:
    """Paste-ready appendix table body (rows only)."""
    lines = []
    for a in agg:
        shot = "0-shot" if a["shots"] == 0 else f"{a['shots']}-shot"
        lines.append(f"{_display_name(a)} & {shot} & {a['mean_accuracy']:.3f} & "
                     f"{a['mean_macro_f1']:.3f} & {a['mean_x_chance']:.1f}$\\times$ \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="custom-endpoint model, repeatable. Use 'label=pipeline_id' for a readable "
                         "name (e.g. gpt-oss-120b=<model_pipeline_id>); a bare "
                         "pipeline id defaults to the custom endpoint.")
    ap.add_argument("--shots", default="0,5", help="comma list of shots-per-outlet (0 = zero-shot)")
    ap.add_argument("--seeds", default="42,13,1234", help="comma list of seeds")
    ap.add_argument("--per-outlet", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="build sample/prompts and print one prompt per (model,k); no API calls")
    args = ap.parse_args()

    specs = [ModelSpec.parse(m) for m in args.model]
    shots_list = [int(x) for x in args.shots.split(",") if x.strip() != ""]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    # Skip (don't abort on) any model whose provider key is missing, so a single
    # unset key never kills the whole batch. --dry-run needs no keys.
    if not args.dry_run:
        usable, skipped = [], []
        for spec in specs:
            (usable if _token_for(spec.provider) else skipped).append(spec)
        for spec in skipped:
            print(f"  WARNING: skipping {spec.tag} -- set one of "
                  f"{PROVIDER_ENV[spec.provider]} to run it")
        if not usable:
            envs = sorted({e for s in specs for e in PROVIDER_ENV[s.provider]})
            raise SystemExit(f"No usable models: export at least one of {envs} and re-run.")
        specs = usable

    df = pd.read_parquet(DATA)
    print(f"data={DATA.name}  2025-test rows={int((df['year']==2025).sum())}  "
          f"models={[s.tag for s in specs]}  shots={shots_list}  seeds={seeds}  chance={CHANCE:.3f}")

    if args.dry_run:
        for spec in specs:
            for k in shots_list:
                block = build_shots(df, k, seeds[0])
                test = sample_test(df, args.per_outlet, seeds[0])
                ex = test.iloc[0]
                print(f"\n--- {spec.tag} {k}-shot (demos={block.count('Outlet:')}) "
                      f"N={len(test)} ---")
                print(build_prompt(ex.title, block)[:800])
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        for k in shots_list:
            for seed in seeds:
                rows.append(run_cell(spec, df, k, seed, args.per_outlet,
                                     args.workers, args.fresh))

    agg = aggregate(rows)
    (OUT_DIR / "llm_baseline_aggregate.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_latex(agg, OUT_DIR / "llm_baseline_table.tex")

    print("\n=== AGGREGATE (mean over seeds) ===")
    for a in agg:
        shot = "0-shot" if a["shots"] == 0 else f"{a['shots']}-shot"
        print(f"  {_display_name(a):28s} {shot:8s} "
              f"acc {a['mean_accuracy']:.3f}±{a['std_accuracy']:.3f}  "
              f"macroF1 {a['mean_macro_f1']:.3f}  {a['mean_x_chance']:.1f}x  "
              f"(parsed {a['mean_parse_rate']:.0%}, {a['n_seeds']} seeds)")
    print(f"\nwrote {OUT_DIR/'llm_baseline_aggregate.json'}")
    print(f"wrote {OUT_DIR/'llm_baseline_table.tex'}  (paste into the appendix table)")


if __name__ == "__main__":
    main()
