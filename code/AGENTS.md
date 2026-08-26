# EchoForge research project instructions

## Mission

Implement a small, reproducible machine-unlearning study using the synthetic
EchoForge voice-production dataset. Prefer simple, inspectable and correct code
over general frameworks or premature abstractions.

## Read before changing code

For every task:

1. Read this file completely.
2. Read the active user story in `codex_stories/`.
3. Read every document explicitly listed by that story.
4. Inspect the files that the story proposes to change.
5. Check the current worktree and preserve unrelated user changes.

Implement only the active story. Do not continue into the next story merely
because it appears straightforward.

## Fixed research scope

- Dataset: EchoForge synthetic dataset only.
- Core scenarios: S1, S3, S4 and S5.
- Extension scenarios S2 and S6 are out of scope until the core study is done.
- Standard architecture: one small binary PyTorch MLP.
- Standard methods: original model, full retraining, retain fine-tuning,
  gradient ascent and gradient difference.
- SISA is a separate sharded/sliced ensemble training path. Never treat a
  monolithic MLP checkpoint as a SISA model.
- Core utility metrics: F1 and AUROC.
- Core forgetting metrics: paired probability distance and TOFU-inspired
  truth-ratio KS statistic/p-value.
- Supporting metric: runtime. Do not implement MIA unless explicitly requested
  by a future story.
- Development seed: 42. Seeds 43 and 44 are used only after seed 42 runs
  correctly and configurations are frozen.

## Protected content

- Do not edit, move, rename or delete anything beneath `notebooks/adult/`,
  `models/adult/`, `results/**/adult/` or `data/**/adult/`.
- Do not rewrite `unlearning_evaluation.py` unless a story explicitly names it.
- Treat every file beneath `data/echoforge/data/raw/` and
  `data/echoforge/data/manifests/` as immutable after installation.
- Never silently regenerate the dataset, redefine a scenario or change a fixed
  train/validation/test split.
- Never delete existing research outputs to make a test pass.

If a requested change conflicts with these protections, stop and explain the
conflict.

## Intended project layout

Follow `docs/PROJECT_STRUCTURE.md`. New reusable logic belongs in the
`src/echoforge/` package. New notebooks belong in `notebooks/echoforge/`. New
tests belong in `tests/echoforge/`. New outputs belong in
`results/echoforge/` or `models/echoforge/`.

## Engineering rules

- Use plain Python, pandas, scikit-learn, SciPy, PyTorch and matplotlib/seaborn.
- Do not introduce Hydra, MLflow, Weights & Biases, a database, a web service,
  a plugin system or an experiment-management framework.
- Avoid inheritance hierarchies and generic trainer abstractions.
- Keep notebooks thin: configuration, function calls, compact tables/plots and
  interpretation. Put reusable logic in `src/echoforge/`.
- Use one shared MLP definition and one shared prediction/evaluation path.
- Use type hints and short docstrings for public functions.
- Use `pathlib.Path`; do not hard-code a username or absolute machine path.
- Do not add a dependency when the standard library or an existing dependency
  is sufficient.
- Use deterministic seeds for Python, NumPy and PyTorch.
- Select Apple MPS, CUDA or CPU through one shared device helper; tests must run
  on CPU.

## Data and preprocessing rules

- `quality_pass` is the target and positive class.
- `recording_id` is for joins/auditing, never a feature.
- `split` controls the fixed split, never a feature.
- Direct entity IDs, consent/licence/rights metadata, row hashes and
  generator-only latent effects must not be model features.
- Fit the main preprocessor on the original training split once and save it.
- Reuse that frozen preprocessor for the monolithic model-only unlearning study.
  Do not refit it separately for each approximate method.
- Every forget set must be a subset of the original training split.
- Define retain IDs as original train IDs minus the scenario forget IDs.
- Do not use the global test set for training, early stopping or configuration
  selection.

## Model and unlearning rules

- Initialise every approximate method from the matching original checkpoint.
- Initialise every full-retraining reference from fresh weights.
- A zero-step approximate run must reproduce original predictions.
- Gradient ascent minimises negative forget loss. Test the gradient direction.
- Gradient difference minimises `lambda_retain * retain_loss - forget_loss`.
- Apply gradient clipping to aggressive unlearning updates.
- Record divergent/invalid runs with `status` and `error_message`; do not silently
  remove them.

## SISA rules

- Create and save one immutable `recording_id -> shard_id -> slice_id` mapping.
- Training records must belong to exactly one shard and one slice.
- Train shard models independently.
- Save a checkpoint after each cumulative slice.
- Aggregate binary probabilities by an explicitly documented rule.
- For a deletion, identify every affected shard and the earliest affected slice,
  then retrain only the necessary suffix without forgotten IDs.
- Compare SISA unlearning with a full SISA retraining reference using the same
  partitioning and aggregation algorithm.
- Report the number of affected shards and slices. If all shards are affected,
  report the lost speed advantage rather than altering the partition after seeing
  results.

## Result rules

- Follow `docs/RESULTS_SCHEMA.md` exactly.
- Each method notebook creates its result table in memory and writes its own CSV
  once. Do not append blindly to an existing CSV.
- One row represents one completed or failed run/configuration.
- Use stable keys: dataset version, model seed, method, scenario and config ID.
- Save raw F1, AUROC, probability distance and KS statistic. Composite scores
  may be derived later but never replace raw metrics.
- Save only predictions needed to reproduce reference comparisons.
- Final comparison notebooks read saved results and never train models.

## Notebook rules

- Follow `docs/NOTEBOOK_STANDARD.md`.
- One notebook per model/method, not one notebook per scenario.
- Method notebooks loop through S1, S3, S4 and S5 in a fixed order.
- Notebooks must restart and run top-to-bottom without hidden state.
- Do not embed large reusable function definitions in notebooks.
- Keep output focused: a training curve when useful, one summary table and only
  plots required by the active story.

## Verification

For each story:

- Run the story-specific commands.
- Run the relevant focused tests.
- If shared code changed, run `python -m pytest tests/echoforge -q`.
- Execute or smoke-test the affected notebook when the story requires it.
- Confirm no Adult Income files changed.
- Report commands run, important outputs and any remaining blocker.

Do not state that work is complete when tests or required notebook execution
failed.

## Final response for each Codex story

Return:

1. Outcome in one sentence.
2. Files created or changed.
3. Verification commands and results.
4. Generated artifact paths.
5. Assumptions, limitations or blockers.
