# EchoForge Machine-Unlearning Study

This repository contains a reproducible study of machine unlearning on the
synthetic EchoForge voice-production dataset. The study evaluates one binary
MLP with full retraining, retain fine-tuning, gradient ascent, gradient
difference and SISA methods.

## Protected existing work

The existing Adult Income notebooks, models, data and results are protected
research artefacts. EchoForge work is isolated under `data/echoforge/`,
`src/echoforge/`, `tests/echoforge/`, `notebooks/echoforge/`,
`models/echoforge/` and `results/echoforge/`.

## Notebook plan

1. `00_data_validation.ipynb` — validate installed data and scenarios.
2. `01_original_mlp.ipynb` — train the seed-42 original model.
3. `02_full_retraining.ipynb` — create full-retraining references.
4. `03_retain_finetuning.ipynb` — evaluate retain-set fine-tuning.
5. `04_gradient_ascent.ipynb` — evaluate gradient-ascent unlearning.
6. `05_gradient_difference.ipynb` — evaluate retain-supported gradient difference.
7. `06_sisa.ipynb` — train and evaluate the SISA ensemble.
8. `07_results_comparison.ipynb` — compare saved results without training.

Run one Codex story at a time, in the order in `CODEX_STORY_INDEX.md`. Do not
start the next story until the current story's verification has passed.

## Study documentation

- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Results schema](docs/RESULTS_SCHEMA.md)
- [Notebook standard](docs/NOTEBOOK_STANDARD.md)
