# US-06 — Train full-retraining references

---

You are implementing **US-06 only: seed-42 monolithic full-retraining
references for S1, S3, S4 and S5**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- the saved original configuration and all current `src/echoforge/` modules

US-05 must be complete and its checkpoint reload test must pass.

## Goal

For each core scenario, train a fresh MLP on the retain set and save the
reference probabilities needed by every approximate method.

## Required work

Create and execute `notebooks/echoforge/02_full_retraining.ipynb`.

For each scenario in the fixed order S1, S3, S4, S5:

1. Load original train rows and the immutable forget manifest.
2. Construct retain rows as train minus forget IDs.
3. Assert exact counts, disjointness and complete train reconstruction.
4. Load the frozen original preprocessor; do not refit it.
5. Initialise a fresh MLP with seed 42. Do not warm-start from the original
   checkpoint. Add an explicit check that initial parameters are not identical
   to the trained original parameters.
6. Train with the selected original architecture/optimizer configuration and
   the same early-stopping rule. Only training rows differ.
7. Time training with explicit start/end boundaries on the chosen device.
8. Evaluate F1/AUROC on the global test set.
9. Predict on the exact scenario forget records in immutable ID order.
10. Save:
    - one reference checkpoint per scenario under
      `models/echoforge/full_retraining/`;
    - global test and forget prediction CSVs under the reference prediction
      path;
    - per-scenario training histories;
    - one complete `results/echoforge/metrics/full_retraining.csv`;
    - one seed-42 reference configuration JSON.

Add any small shared helper only when it avoids genuine duplication. Do not
create a general experiment framework.

## Required checks

- No forget ID appears in reference training IDs.
- Every expected retain ID appears once.
- Saved reference predictions contain exactly the expected forget IDs.
- All probabilities/losses are finite.
- Each saved checkpoint reloads with identical predictions.
- Rerunning replaces method outputs without duplication.

## Acceptance criteria

- Four completed reference rows and four valid checkpoints exist.
- Each result row follows `docs/RESULTS_SCHEMA.md`.
- Full retraining runtime is present and positive.
- The notebook does not calculate approximate unlearning or MIA.

## Verification

```bash
python -m pytest tests/echoforge -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/02_full_retraining.ipynb --output 02_full_retraining.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=3600
```

Report retain/forget counts, F1, AUROC and runtime for all four scenarios. Do not
begin US-07.

