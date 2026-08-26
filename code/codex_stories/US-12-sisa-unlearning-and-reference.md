# US-12 — Implement SISA unlearning and full SISA references

---

You are implementing **US-12 only: SISA deletion for S1/S3/S4/S5 and matching
full SISA retraining references**.

Read completely:

- `AGENTS.md`, especially SISA rules
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- `src/echoforge/sisa.py`
- the immutable partition/configuration from US-11

US-11 must pass. Do not recreate, rebalance or tune the partition.

## Goal

For each deletion request, retrain only the required SISA shard suffix and
compare it with a fresh SISA ensemble trained on the retain data using the same
algorithm.

## Required implementation

Extend `src/echoforge/sisa.py` with:

- `affected_shards_and_slices(forget_ids, partition)` returning every affected
  shard and its earliest affected slice;
- `sisa_unlearn(...)` which:
  - leaves unaffected shard checkpoints byte/state unchanged;
  - loads the checkpoint immediately before the earliest affected slice, or
    fresh deterministic weights when the earliest slice is zero;
  - retrains subsequent cumulative slices while excluding every forget ID;
  - returns final shard models, histories, affected counts and runtime;
- `train_full_sisa_reference(...)` which trains a fresh retain-only ensemble
  using the same original shard/slice mapping after removing forgotten IDs;
- helpers for final probability aggregation and audited training-ID logs.

If a slice becomes empty, skip its update explicitly while preserving the
checkpoint sequence. Never move rows between shards/slices after deletion.

## Mandatory tests

- affected-shard/slice detection on a toy mapping;
- forgotten IDs never enter retraining batches;
- unaffected final shard parameters/predictions remain exactly unchanged;
- an earliest-slice-zero deletion starts that shard from fresh weights;
- a later-slice deletion loads the preceding checkpoint;
- full SISA reference contains all and only retain IDs;
- aggregated probabilities are aligned and finite.

## Notebook completion

Extend and execute `notebooks/echoforge/06_sisa.ipynb` with a new deletion
section looping S1/S3/S4/S5.

For each scenario:

1. list affected shards and earliest slices;
2. perform targeted SISA retraining;
3. perform full SISA retain-only retraining;
4. evaluate both on global test;
5. compare targeted SISA forget predictions with full SISA reference using the
   two core forgetting metrics;
6. record targeted runtime, full reference runtime and speed-up;
7. save one `sisa.csv` row and selected prediction/checkpoint artifacts.

Use one fixed SISA configuration, so four main result rows are expected. Clearly
separate SISA's method-matched reference from the monolithic full-retraining
reference.

## Acceptance criteria

- Four SISA result rows use method-matched full SISA references.
- Forgotten IDs are absent from every affected retraining log.
- Unaffected shard states are unchanged.
- Affected-shard counts, runtimes and speed-ups are saved without changing the
  original partition.

## Verification

```bash
python -m pytest tests/echoforge -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/06_sisa.ipynb --output 06_sisa.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=10800
```

Report affected shard counts, speed-ups and utility/forgetting metrics. If a
scenario touches every shard, state that SISA loses its expected efficiency;
do not modify the partition.
