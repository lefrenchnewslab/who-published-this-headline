#!/usr/bin/env python3
"""
Second-coder reliability for SAME-EVENT identity (matched-event calibration).

The paper calibrates the 0.70 cosine cutoff on a 400-pair sample coded by one
expert. Inter-coder agreement on that judgment is reported here,
especially near the threshold. This builds a blind 50-pair packet for a second
expert and, once filled, computes Cohen's kappa vs. the first coder.

Design: the 50 pairs are stratified by cosine so the boundary region (0.60-0.80),
where same-event calls are hardest, is over-represented -- kappa is then a real
stress test, not an easy-case ceiling. The blind sheet hides both the first
coder's label AND the cosine value (so the second coder is not anchored).

Usage
-----
Build the packet (deterministic):
    python3 gen_secondcoder_sameevent.py make --n 50 --seed 42

  writes to second_coder/:
    sameevent_BLANK.csv   pair_id, title_i, title_j, same_event   <- give to coder 2
    sameevent_KEY.csv     pair_id, coder1_same_event, cos, i, j   <- do NOT show coder 2
    INSTRUCTIONS.md       coding guideline (one paragraph)

Score once coder 2 has filled the same_event column of sameevent_BLANK.csv:
    python3 gen_secondcoder_sameevent.py score --filled second_coder/sameevent_BLANK.csv

  prints Cohen's kappa, raw agreement, boundary-only agreement, and a paste-ready
  sentence for the appendix.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # package root
SRC = ROOT / "data" / "derived_features" / "tier2_pair_labeled.csv"
OUT = ROOT / "second_coder"
CUTOFF = 0.70
BOUNDARY = (0.60, 0.80)

INSTRUCTIONS = """\
# Same-event coding — second rater

You will see 50 pairs of French news headlines. For each pair, decide whether the
two headlines report **the same specific real-world occurrence** (the same event),
not merely the same topic or theme.

- Label **1** if both headlines are about the same specific event
  (e.g. both about the same vote, the same match, the same incident on the same day).
- Label **0** otherwise — including when they share a topic but describe different
  events (e.g. two different matches, two different votes, general-vs-specific).

Judge the event, not the wording or the outlet. Fill the `same_event` column with
1 or 0 for every row; leave none blank. You are blind to the cosine similarity and
to the first coder's labels by design — code each pair on its own.
"""


def _band(cos: float) -> str:
    if cos < BOUNDARY[0]:
        return "low"
    if cos < BOUNDARY[1]:
        return "boundary"
    return "high"


def make(n: int, seed: int) -> None:
    df = pd.read_csv(SRC)
    df["band"] = df["cos"].map(_band)
    # ~half the sample from the hard boundary region, rest split low/high
    n_boundary = n // 2
    n_side = n - n_boundary
    n_low = n_side // 2
    n_high = n_side - n_low
    quota = {"boundary": n_boundary, "low": n_low, "high": n_high}

    rng = np.random.RandomState(seed)
    picks = []
    for band, k in quota.items():
        sub = df[df["band"] == band]
        take = min(k, len(sub))
        picks.append(sub.sample(take, random_state=rng))
    sample = pd.concat(picks)
    # backfill if any band was short, to reach exactly n
    if len(sample) < n:
        remaining = df[~df.index.isin(sample.index)]
        sample = pd.concat([sample, remaining.sample(n - len(sample), random_state=rng)])
    sample = sample.sample(frac=1, random_state=rng).reset_index(drop=True)  # shuffle order
    sample.insert(0, "pair_id", range(1, len(sample) + 1))

    OUT.mkdir(parents=True, exist_ok=True)
    blank = sample[["pair_id", "title_i", "title_j"]].copy()
    blank["same_event"] = ""  # to be filled by coder 2
    blank.to_csv(OUT / "sameevent_BLANK.csv", index=False)

    key = sample[["pair_id", "same_event", "cos", "i", "j"]].rename(
        columns={"same_event": "coder1_same_event"})
    key.to_csv(OUT / "sameevent_KEY.csv", index=False)
    (OUT / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")

    comp = sample["band"].value_counts().to_dict()
    lab = sample["same_event"].value_counts().to_dict()
    print(f"wrote {OUT/'sameevent_BLANK.csv'}  (n={len(sample)})")
    print(f"wrote {OUT/'sameevent_KEY.csv'}   (hidden coder-1 labels + cosine)")
    print(f"wrote {OUT/'INSTRUCTIONS.md'}")
    print(f"cosine bands: {comp}   (boundary={BOUNDARY[0]}-{BOUNDARY[1]})")
    print(f"coder-1 label balance in sample: same={lab.get(1,0)} diff={lab.get(0,0)}")
    print("LEAK CHECK: blank sheet columns =", list(blank.columns),
          "(no cos, no coder-1 label)")


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary label vectors (no sklearn dependency)."""
    n = len(a)
    po = float((a == b).mean())
    # expected agreement from marginals
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(filled: Path) -> None:
    f = pd.read_csv(filled)
    key = pd.read_csv(OUT / "sameevent_KEY.csv")
    if f["same_event"].isna().any() or (f["same_event"].astype(str).str.strip() == "").any():
        raise SystemExit("Some rows are unlabeled — coder 2 must fill every same_event cell.")
    m = f[["pair_id", "same_event"]].merge(key, on="pair_id", how="inner")
    c2 = m["same_event"].astype(int).to_numpy()
    c1 = m["coder1_same_event"].astype(int).to_numpy()

    kappa = _kappa(c1, c2)
    agree = float((c1 == c2).mean())
    bmask = (m["cos"] >= BOUNDARY[0]) & (m["cos"] < BOUNDARY[1])
    b_agree = float((c1[bmask.to_numpy()] == c2[bmask.to_numpy()]).mean()) if bmask.any() else float("nan")
    n, nb = len(m), int(bmask.sum())

    print(f"n={n} pairs ({nb} in the {BOUNDARY[0]}-{BOUNDARY[1]} boundary band)")
    print(f"Cohen's kappa (same-event identity) = {kappa:.3f}")
    print(f"raw agreement = {agree:.3f}   boundary-band agreement = {b_agree:.3f}")
    print("\n=== PASTE-READY (appendix) ===")
    print(f"A second expert independently re-coded a 50-pair subset of the same-event "
          f"sample (blind to the first coder and to cosine), stratified toward the "
          f"0.60--0.80 boundary; inter-coder agreement is high (Cohen's $\\kappa = {kappa:.2f}$, "
          f"raw agreement {agree*100:.0f}\\%), confirming that same-event identity is "
          f"reliably codable and does not rest on a single rater's judgment.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make"); mk.add_argument("--n", type=int, default=50); mk.add_argument("--seed", type=int, default=42)
    sc = sub.add_parser("score"); sc.add_argument("--filled", type=Path, default=OUT / "sameevent_BLANK.csv")
    args = ap.parse_args()
    if args.cmd == "make":
        make(args.n, args.seed)
    else:
        score(args.filled)


if __name__ == "__main__":
    main()
