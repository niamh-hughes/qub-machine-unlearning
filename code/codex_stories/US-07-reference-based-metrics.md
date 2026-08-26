# US-07 — Implement and validate reference-based forgetting metrics

---

You are implementing **US-07 only: paired probability and TOFU-inspired
truth-ratio metrics**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- existing `src/echoforge/metrics.py`
- saved original and full-retraining prediction schemas

US-06 must have produced four valid references.

## Goal

Implement the two primary forgetting metrics once, test them independently, and
produce a reference-gap audit showing whether each scenario is measurable.

## Required API in `src/echoforge/metrics.py`

- `align_prediction_frames(left, right)` merging one-to-one by `recording_id`,
  validating labels and rejecting missing/duplicate IDs;
- `paired_probability_distance(unlearned, reference) -> float` as mean absolute
  probability difference on aligned rows;
- `true_class_probability(y_true, p_positive)`;
- `truth_ratio(y_true, p_positive, epsilon=1e-7)` as
  `(1 - p_true) / (p_true + epsilon)` with clipped valid probabilities;
- `truth_ratio_ks(unlearned, reference) -> dict` returning at least KS D,
  p-value and compared sample counts using `scipy.stats.ks_2samp`;
- `original_reference_gap(original, reference)` returning raw original-to-
  retrained probability distance and KS D.

Do not implement a composite forgetting score or MIA.

## Tests

Cover:

- identical probabilities give zero probability distance and KS D;
- known toy arrays produce a manually checked MAD;
- row order does not affect aligned results;
- duplicate/missing IDs and label mismatches fail clearly;
- true-class probability handles both binary labels;
- extreme probabilities remain finite due to clipping;
- KS outputs are within valid ranges;
- empty and one-class inputs receive explicit behaviour.

## Reference-gap audit

Create `scripts/build_echoforge_reference_gap.py`. It must load saved original
and matching full-retraining forget predictions for S1/S3/S4/S5, calculate raw
gaps and write
`results/echoforge/metrics/reference_gap_seed_42.csv` once.

Include a `weakly_identifiable` Boolean based on a predeclared small probability-
distance threshold stored in the script/config and documented in its output.
Do not change scenarios to make gaps larger.

## Acceptance criteria

- Metrics operate only on ID-aligned predictions.
- Reference-gap output has exactly one row per core scenario.
- Raw effect statistics are saved; p-values are not described as proof.
- No existing model is retrained.

## Verification

```bash
python -m pytest tests/echoforge/test_metrics.py -q
python scripts/build_echoforge_reference_gap.py
python -m pytest tests/echoforge -q
```

Report the four raw reference gaps and any weakly identifiable flag. Do not
begin an approximate method.

