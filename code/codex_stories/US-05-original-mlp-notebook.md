# US-05 — Train the original EchoForge MLP

---

You are implementing **US-05 only: the original-model notebook and its minimal
utility metrics**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- all current `src/echoforge/` modules

US-03 and US-04 must pass before editing.

## Goal

Fit the original seed-42 MLP on all 14,000 training records, select the baseline
configuration using validation data only, and save deterministic artifacts that
all later monolithic methods will reuse.

## Required work

1. Create or extend `src/echoforge/metrics.py` with only:
   - binary F1 and AUROC calculation;
   - input validation for aligned labels/probabilities;
   - no forgetting or MIA metrics yet.
2. Add metric tests, including a single-class AUROC case returning `NaN` with a
   documented reason rather than crashing.
3. Create and execute `notebooks/echoforge/01_original_mlp.ipynb` using the
   standard layout.
4. Use seed 42. Fit the preprocessor on original training data and save it to
   `models/echoforge/original/preprocessor.joblib`.
5. Run the compact validation search:
   - learning rate: 0.001 and 0.0003;
   - weight decay: 0 and 0.0001;
   - maximum 50 epochs with fixed documented early-stopping patience.
6. Select by validation loss, with AUROC as a reported check, not by test score.
7. Retrain/select consistently according to a declared rule and save:
   - `models/echoforge/original/original_seed_42.pt`;
   - `results/echoforge/configurations/original_seed_42.json`;
   - training/search history CSV;
   - global test predictions;
   - forget-record predictions for S1/S3/S4/S5 from the same original model;
   - `results/echoforge/metrics/original.csv` following the shared schema.
8. Reload the saved checkpoint and assert prediction equality before finishing.

The notebook may show one loss curve, one ROC curve and one compact summary
table. Do not add confusion matrices or MIA.

## Acceptance criteria

- Preprocessor and model are fitted only from the fixed train/validation policy.
- Global test data never influences selection.
- Prediction files merge by `recording_id` and follow the schema.
- All probabilities are finite and in [0, 1].
- Rerunning replaces, rather than duplicates, outputs.
- The saved checkpoint reloads exactly.

## Verification

```bash
python -m pytest tests/echoforge -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/01_original_mlp.ipynb --output 01_original_mlp.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=1800
```

Report selected configuration, selected epoch, F1, AUROC and all artifact paths.
Do not begin full retraining.

