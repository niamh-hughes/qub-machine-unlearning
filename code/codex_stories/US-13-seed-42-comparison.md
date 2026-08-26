# US-13 — Compare seed-42 methods and freeze final configurations

---

You are implementing **US-13 only: the seed-42 comparison notebook and frozen
configuration selection**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- every existing EchoForge method result/configuration file

US-08, US-09, US-10 and US-12 must be complete. Stop with a clear missing-file
list if any required result is absent or has failed key validation.

## Goal

Create a training-free comparison that answers which configurations provide the
best utility/forgetting balance at seed 42 and freezes the small set to repeat
with seeds 43 and 44.

## Required work

Create and execute `notebooks/echoforge/07_results_comparison.ipynb`.

The notebook must:

1. Load method CSVs; never train or modify model checkpoints.
2. Validate required columns, stable keys, metric ranges and expected row counts.
3. Keep `monolithic_mlp` and `sisa_ensemble` references explicit. Do not compare
   SISA forget predictions with the monolithic reference as though they were
   the same learning algorithm.
4. Report raw F1, AUROC, probability distance, KS D/p, runtime and status.
5. Calculate relative F1/AUROC to the matching reference and the harmonic
   plotting utility score, while retaining raw metrics.
6. Use S1 and S5 as the predeclared pilot scenarios for configuration freezing.
7. For each approximate monolithic method, select the configuration with the
   lowest mean paired probability distance across S1/S5 subject to:
   - relative harmonic utility at least 0.98 on both pilot scenarios;
   - no raw F1 or AUROC loss greater than 0.02 versus matching retraining.
   Use mean KS D as a tie-breaker. If no configuration is feasible, choose the
   least-damaging Pareto point and mark `constraint_met=false`.
8. Freeze one selected config plus one neighbouring strength per approximate
   method in
   `results/echoforge/configurations/frozen_final_configs.json`. SISA has one
   fixed configuration.
9. Produce and save only these core figures:
   - utility versus forgetting scatter;
   - scenario difficulty at the utility constraint;
   - runtime/speed-up comparison;
   - method trajectory plot for the approximate methods.
10. Save a concise seed-42 comparison table and Pareto table.

Do not invent missing metrics, drop failed runs or choose configurations by
looking separately for the most flattering result in each scenario.

## Acceptance criteria

- Notebook executes without invoking training functions.
- Configuration-selection logic is visible, deterministic and tested on a toy
  table in `tests/echoforge/test_selection.py`.
- Frozen JSON contains exact config IDs/parameters, selection rule and
  constraint flag.
- All figures derive from saved CSVs.
- SISA architecture/reference difference is clearly labelled.

## Verification

```bash
python -m pytest tests/echoforge -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/07_results_comparison.ipynb --output 07_results_comparison.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=900
```

Report selected configs and constraint status. Do not start multi-seed runs.

