# US-11 — Build the SISA partition and original ensemble

---

You are implementing **US-11 only: the immutable SISA partition, cumulative
slice training and original SISA ensemble**.

Read completely:

- `AGENTS.md`, especially SISA rules
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- original monolithic configuration and current shared model/training code

US-05 must be complete. SISA is a separate training path and must not reuse the
trained monolithic checkpoint.

## Goal

Create a genuine five-shard, three-slice SISA training structure before any
SISA deletion experiment is attempted.

## Required implementation in `src/echoforge/sisa.py`

Implement explicit functions for:

- creating a stable actor-based shard assignment from raw recording metadata;
- assigning training rows to three chronological slices within each shard by
  sorting `recording_date`, then `recording_id`, then splitting as evenly as
  possible;
- validating that every training ID occurs in exactly one shard/slice and no
  validation/test ID occurs;
- training one independent shard MLP cumulatively through slices;
- saving/loading a checkpoint after each cumulative slice;
- predicting with every final shard model;
- aggregating binary probabilities by unweighted mean.

Use the frozen preprocessor and shared MLP architecture, but initialise each
shard model independently and deterministically from seed 42 plus shard ID.
Use one simple declared `epochs_per_slice` value of 5. Do not tune shard/slice
counts, partition strategies or aggregation rules in this story.

Save the immutable mapping to:

`results/echoforge/configurations/sisa_partition_seed_42.csv`

with columns `recording_id`, `actor_id`, `shard_id`, `slice_id`,
`recording_date`. Save a JSON configuration including partition formula, sort
rule, number of shards/slices, epochs per slice and aggregation rule.

## Tests

- deterministic partition recreation;
- exactly one assignment per train ID;
- no non-train IDs;
- all records for one actor map to one shard;
- slice IDs are chronological within a shard;
- shard models are independent objects/initialisations;
- mean aggregation matches a manual toy example;
- checkpoint reload preserves shard probabilities.

## Notebook

Create the initial version of `notebooks/echoforge/06_sisa.ipynb` containing:

1. partition validation and shard/slice size tables;
2. cumulative training of the original SISA ensemble;
3. global-test F1/AUROC and runtime;
4. saved final shard and slice checkpoints;
5. a compact comparison with original monolithic utility;
6. no deletion/unlearning yet.

Save original ensemble metrics to
`results/echoforge/metrics/sisa_original_seed_42.csv`; US-12 will build the main
`sisa.csv`.

## Acceptance criteria

- Every training row has one stable shard/slice assignment.
- Fifteen cumulative shard/slice checkpoints are valid or an explicit documented
  count explains empty slices.
- Original SISA ensemble predictions reload reproducibly.
- No deletion or partition tuning has occurred.

## Verification

```bash
python -m pytest tests/echoforge -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/06_sisa.ipynb --output 06_sisa.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=7200
```

Report shard/slice counts, ensemble utility, total training time and checkpoints.
Do not perform SISA deletion or change the partition after seeing utility.
