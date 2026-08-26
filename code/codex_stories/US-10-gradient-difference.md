# US-10 — Implement gradient-difference unlearning

---

You are implementing **US-10 only: retain-supported gradient difference and
`05_gradient_difference.ipynb`**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- current `src/echoforge/unlearning.py`

US-07 must pass. Do not alter completed retain-fine-tuning or gradient-ascent
results.

## Goal

Train from the original checkpoint using the declared objective:

```text
lambda_retain * retain_loss - forget_loss
```

The method should increase forget loss while using retain loss to constrain
collateral damage.

## Required implementation

Add `gradient_difference_unlearn(...)` to
`src/echoforge/unlearning.py`.

- Accept separate retain and forget arrays/loaders.
- Use matched-size batches when practical. If loaders differ in length, cycle
  the retain loader without silently discarding forget batches.
- Calculate both normal component losses before combining them.
- Apply the exact objective above and gradient clipping.
- Return updated model, history containing retain loss, forget loss and combined
  objective, runtime and status.
- Support zero steps and protect the original checkpoint.

## Mandatory tests

- With `lambda_retain=0`, the update direction matches gradient ascent.
- On a controlled batch, the forget component raises forget BCE.
- Increasing the retain weight changes the gradient in the expected retain-loss
  direction.
- Both loaders are used and batch-size mismatch is handled.
- Zero steps reproduce original predictions.
- Divergence/non-finite values are detected.

## Notebook

Create and execute `notebooks/echoforge/05_gradient_difference.ipynb` using the
same sections, scenario order and output schema as the previous method
notebooks.

Use this fixed compact grid:

- learning rate 0.00001;
- epochs 1 and 3;
- `lambda_retain` 0.5 and 1.0;
- one documented clip norm.

This creates 16 planned result keys. For every valid configuration calculate
global F1/AUROC, paired probability distance, truth-ratio KS and runtime. Save
method configuration JSON, necessary predictions and
`gradient_difference.csv` once after uniqueness validation.

Do not introduce another balancing coefficient, scheduler or per-scenario grid.

## Acceptance criteria

- Objective-sign tests pass.
- Both forget and retain data use only their intended IDs.
- All runs start from identical original weights.
- Sixteen completed/failed keys exist with no duplicates.
- Components and combined objective are visible in saved history.

## Verification

```bash
python -m pytest tests/echoforge -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/05_gradient_difference.ipynb --output 05_gradient_difference.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=3600
```

Report whether retain support reduced utility damage at comparable forgetting.
Do not begin SISA.

