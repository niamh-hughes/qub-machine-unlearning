# US-08 — Implement retain fine-tuning

---

You are implementing **US-08 only: retain fine-tuning and
`03_retain_finetuning.ipynb`**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- current training, metrics and data modules

US-07 must pass.

## Goal

Start from the seed-42 original MLP and continue training only on each
scenario's retain set. Evaluate every configuration against the matching fresh
full-retraining reference.

## Required implementation

Create `src/echoforge/unlearning.py` with an explicit
`retain_finetune(...)` function that:

- accepts an already-loaded model copy, retain arrays, learning rate, epochs,
  batch size, weight decay, device and seed;
- never mutates the saved original checkpoint;
- uses BCE/AdamW and returns the updated model, tidy history and runtime;
- supports `epochs=0` for the zero-step control.

Add tests that:

- a zero-step run gives exactly the original predictions;
- training batches contain only retain IDs in a controlled integration fixture;
- a small retain run changes parameters and remains finite;
- the saved original checkpoint hash/state is unchanged.

## Notebook

Create and execute `notebooks/echoforge/03_retain_finetuning.ipynb` using the
standard method layout and looping S1/S3/S4/S5.

Use the compact fixed grid:

- learning rate 0.0001;
- epochs 1 and 3;
- original batch size and selected weight decay.

For every scenario/configuration:

1. Reload a fresh copy of the original checkpoint.
2. Construct the retain set from immutable manifests.
3. Run fine-tuning.
4. Calculate global-test F1/AUROC.
5. Calculate paired probability distance and truth-ratio KS against the matching
   full-retraining forget predictions.
6. Record runtime, counts, device, status and deterministic config ID.
7. Save selected prediction files named with scenario, seed and config ID.

Write `retain_finetuning.csv` once after validating unique run keys. Save the
notebook configuration JSON. Do not select a different hyperparameter grid for
each scenario.

## Acceptance criteria

- Eight method rows exist: four scenarios times two epoch settings.
- Zero-step control is tested but need not appear in the main eight rows.
- Every run begins from identical original weights.
- No forget row is used for updating.
- Original/reference artifacts remain unchanged.

## Verification

```bash
python -m pytest tests/echoforge -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/03_retain_finetuning.ipynb --output 03_retain_finetuning.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=3600
```

Report the eight rows compactly. Do not begin gradient ascent.

