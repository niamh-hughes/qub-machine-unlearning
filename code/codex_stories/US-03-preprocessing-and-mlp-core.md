# US-03 — Implement preprocessing, MLP and shared training code

---

You are implementing **US-03 only: preprocessing, the shared MLP and common
training/prediction functions**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`
- existing `src/echoforge/data.py`

US-02 tests must pass before editing.

## Goal

Create one transparent preprocessing path and one shared binary MLP used by all
monolithic experiments. Do not train the full dataset or create notebooks in
this story.

## Required work

In `src/echoforge/data.py`, add:

- `infer_feature_columns(train_features)` returning ordered numeric/categorical
  column lists;
- `build_preprocessor(train_features) -> ColumnTransformer` using
  `StandardScaler` and `OneHotEncoder(handle_unknown="ignore")`;
- `fit_preprocessor(train_features)`;
- `transform_features(preprocessor, features) -> np.ndarray` with float32,
  finite-value and row-count validation;
- `save_preprocessor` and `load_preprocessor` using joblib.

The preprocessor is fitted on the original training split once and later reused.

Create `src/echoforge/model.py` with:

- `EchoForgeMLP(input_dim: int, dropout: float = 0.2)` implementing
  input -> 64 -> ReLU -> Dropout -> 32 -> ReLU -> output;
- `build_model(input_dim, seed, dropout=0.2)`.

Create `src/echoforge/training.py` with:

- `set_global_seed(seed)` for Python, NumPy and PyTorch;
- `select_device(preference="auto")` supporting MPS, CUDA and CPU;
- tensor/dataloader helpers;
- `train_binary_model(...)` with AdamW, `BCEWithLogitsLoss`, validation loss,
  optional early stopping and a tidy history DataFrame;
- `predict_binary_model(...)` returning probabilities, predicted classes,
  per-record BCE and true-class probabilities;
- checkpoint save/load functions that store model dimensions, state dict,
  seed and selected training configuration.

Keep signatures explicit. Do not create a generic Trainer class.

## Tests

Add focused tests for:

- excluded columns never appearing in model features;
- deterministic preprocessing shape/order;
- transformed values finite and float32;
- MLP output shape `(batch, 1)`;
- deterministic CPU initialisation;
- a small synthetic batch can be overfitted;
- checkpoint reload produces identical probabilities;
- selected device is valid without requiring a GPU.

Use a tiny fixture/sample, not the complete dataset, for model tests.

## Files allowed to change

- `src/echoforge/data.py`
- `src/echoforge/model.py`
- `src/echoforge/training.py`
- public exports in `src/echoforge/__init__.py`
- `tests/echoforge/test_preprocessing.py`
- `tests/echoforge/test_model_training.py`

## Acceptance criteria

- One shared deterministic preprocessing/model/training path exists.
- Input dimensions and feature order are stable.
- CPU tests prove train, predict, save and reload behaviour.
- No full-dataset model or project preprocessor has been created.

## Verification

```bash
python -m pytest tests/echoforge -q
python -m compileall -q src/echoforge
```

All tests must run on CPU. Do not save a fitted project preprocessor, train the
real baseline or begin US-04.
