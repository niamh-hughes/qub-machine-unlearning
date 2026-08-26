# US-09 — Implement gradient-ascent unlearning

---

You are implementing **US-09 only: bounded gradient-ascent unlearning and
`04_gradient_ascent.ipynb`**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- current `src/echoforge/unlearning.py`

US-07 must pass. US-08 may be complete but its implementation must not be copied
into this method.

## Goal

Update an original checkpoint by increasing BCE loss on forget records while
controlling numerical instability and recording failures honestly.

## Required implementation

Add `gradient_ascent_unlearn(...)` to `src/echoforge/unlearning.py`.

- The optimiser must minimise `-forget_loss`; do not accidentally minimise the
  normal loss.
- Use only forget batches for the update.
- Accept learning rate, epochs, batch size, gradient-clip norm, device and seed.
- Return updated model, per-epoch history and runtime.
- Record normal forget BCE before/after each epoch for interpretability.
- Stop on non-finite loss/parameters and return a failed/diverged status rather
  than a misleading checkpoint.
- Support zero steps as an exact original-model control.

## Mandatory tests

- On a deterministic toy batch, one small update increases normal forget BCE.
- The update gradient has the opposite sign to ordinary loss minimisation.
- Zero steps reproduce original predictions.
- Gradient clipping is applied.
- Non-finite updates are detected and represented as invalid/diverged.
- Saved original weights remain unchanged.

## Notebook

Create and execute `notebooks/echoforge/04_gradient_ascent.ipynb` with the same
outer scenario loop and result format as retain fine-tuning.

Fixed grid:

- learning rates 0.00001 and 0.00005;
- epochs 1 and 3;
- one documented gradient-clip norm.

This creates 16 planned rows: four scenarios times four configurations. For
each row calculate global utility, the two reference-based forgetting metrics,
runtime and status. Failed runs remain in the CSV with metric fields `NaN` and a
concise error message.

Save method configuration JSON, necessary selected predictions and
`results/echoforge/metrics/gradient_ascent.csv` once. Limit notebook output to a
compact trajectory figure and result table.

## Acceptance criteria

- The gradient-direction unit test proves ascent behaviour.
- Every valid run begins from the same original checkpoint.
- No retain rows are used in the ascent update.
- Exactly 16 completed/failed result keys are present without duplicates.
- Divergence is visible rather than silently excluded.

## Verification

```bash
python -m pytest tests/echoforge -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/04_gradient_ascent.ipynb --output 04_gradient_ascent.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=3600
```

Report completed versus failed configurations and the observed utility/
forgetting direction. Do not begin gradient difference.

