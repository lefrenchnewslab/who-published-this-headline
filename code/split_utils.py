"""Canonical dataset splits — SINGLE SOURCE OF TRUTH so every standalone script matches the paper.

THE BUG THIS PREVENTS
---------------------
The paper's temporal split (phase1/phase2 `temporal_split`, TEMPORAL_VAL_FRACTION=0.10) carves the
**latest 10% of 2024 headlines (by `dt`) as a validation slice and EXCLUDES it from training**. A
standalone script that builds its training set as `df[df.year <= 2024]` instead trains on ~3,463 extra
rows and inflates every metric by ~0.5pt (e.g. char-SVM raw 0.496 -> 0.501, typo 0.427 -> 0.431). This
is exactly the 0.501-vs-0.496 inconsistency found on 2026-06-22.

RULE: never write `df[df.year <= 2024]` for a training set in this repo. Always call `temporal_split`.

Requires columns: `year` (int) and `dt` (datetime) — `dt` is needed to order 2024 for the val carve.
"""
TRAIN_MAX_YEAR = 2024
TEST_YEAR = 2025
VAL_FRACTION = 0.10  # matches phase1/phase2 TEMPORAL_VAL_FRACTION


def temporal_split(df, return_val=False):
    """Canonical paper temporal split.

    train = years <= 2024 MINUS the latest 10% of 2024 (held-out validation, by `dt`);
    val   = that latest 10% of 2024;
    test  = year == 2025.

    Returns (train, test) by default, or (train, val, test) if return_val=True.
    """
    if "dt" not in df.columns or "year" not in df.columns:
        raise ValueError("temporal_split requires both 'year' and 'dt' columns "
                         "(load them from sampled_dataset_<panel>.parquet).")
    pool = df[df.year <= TRAIN_MAX_YEAR]
    test = df[df.year == TEST_YEAR]
    y2024 = pool[pool.year == TRAIN_MAX_YEAR].sort_values("dt")
    val = y2024.tail(int(len(y2024) * VAL_FRACTION))
    train = pool.drop(index=val.index)
    return (train, val, test) if return_val else (train, test)
