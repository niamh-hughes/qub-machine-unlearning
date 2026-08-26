# US-14 — Repeat only frozen configurations for seeds 43 and 44

---

You are implementing **US-14 only: final multi-seed repetition using the frozen
US-13 configurations**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `results/echoforge/configurations/frozen_final_configs.json`
- all EchoForge notebooks and shared modules

US-13 must be complete. Stop if the frozen configuration file is missing,
ambiguous or contains parameters not represented in seed-42 results.

## Goal

Repeat the final selected configurations with model seeds 43 and 44, preserving
the seed-42 exploratory grid but not repeating that whole grid for new seeds.

## Required work

1. Add a small explicit `run_phase` field to result generation:
   `pilot_grid` for exploratory seed-42 rows and `final_repeat` for selected
   multi-seed rows.
2. Update the original, full-retraining and method notebooks so their
   configuration cell supports:
   - development/pilot mode: seed 42 and original compact grid;
   - final mode: seeds 42, 43 and 44 using only frozen selected configurations
     (and the explicitly frozen neighbour where requested).
3. Do not use Papermill or introduce a workflow framework. A simple
   `NOTEBOOK_CONFIG["run_mode"]` and seed loop is sufficient.
4. For every seed:
   - initialise original and full-retraining models deterministically;
   - save seed-specific checkpoints/predictions;
   - run only frozen retain fine-tuning, gradient ascent and gradient difference
     configurations;
   - train/evaluate the fixed SISA configuration with a seed-specific immutable
     partition file using the same partition rule;
   - retain failed/divergent rows.
5. Rebuild each method CSV deterministically from its intended pilot and final
   rows. Validate unique keys before writing.
6. Execute notebooks 01 through 06 in dependency order in final mode.
7. Re-execute notebook 07 so it reports:
   - individual seed values;
   - mean and standard deviation;
   - whether the seed-42 conclusion remains stable;
   - no claim of statistical certainty from only three seeds.
8. Save a compact `multi_seed_summary.csv` and seed-variation figure.

## Scope controls

- Do not repeat the full seed-42 hyperparameter grids for seeds 43/44.
- Do not change a frozen parameter after seeing a new seed result.
- Do not regenerate the dataset for each model seed.
- Do not add S2, S6 or MIA.

## Acceptance criteria

- Every selected method/scenario has seed 42/43/44 rows or an explicit failure.
- Checkpoints and predictions are seed-specific and never overwritten by another
  seed.
- Summary mean/std reconciles with individual rows.
- All notebooks still support the original development mode.

## Verification

```bash
python -m pytest tests/echoforge -q
```

Then execute notebooks 01–07 in documented final mode with appropriate timeouts.
Report every command and total failed/completed run count. Do not begin cleanup
until all discrepancies are explained.

