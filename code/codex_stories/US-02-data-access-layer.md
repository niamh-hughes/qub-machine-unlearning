# US-02 — Implement the EchoForge data access layer

---

You are implementing **US-02 only: the reusable EchoForge data access layer**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/RESULTS_SCHEMA.md`

US-01 must already pass. Run its verification command before editing.

## Goal

Provide small, tested functions for loading the fixed modelling table, metadata,
splits and deletion manifests. Do not implement preprocessing, neural networks
or notebooks in this story.

## Required API

Create `src/echoforge/paths.py` with:

- `find_project_root(start: Path | None = None) -> Path`
- `data_root() -> Path`
- `results_root() -> Path`
- `models_root() -> Path`

The root is the nearest parent containing both `AGENTS.md` and `src/`. Never
hard-code an absolute path.

Create `src/echoforge/data.py` with:

- `CORE_SCENARIOS = ("S1", "S3", "S4", "S5")`
- `load_model_table() -> pd.DataFrame`
- `load_recording_metadata() -> pd.DataFrame`
- `load_scenario_definitions() -> pd.DataFrame`
- `load_forget_manifest(scenario_id: str) -> pd.DataFrame`
- `split_model_table(df) -> dict[str, pd.DataFrame]`
- `scenario_train_frames(model_df, scenario_id) -> tuple[pd.DataFrame, pd.DataFrame]`
- `feature_target_split(df) -> tuple[pd.DataFrame, pd.Series]`

## Behaviour requirements

- Validate scenario IDs and reject S2/S6 unless an explicit
  `allow_extension=True` argument is supplied to the manifest loader.
- Preserve `recording_id` while joining and checking, but remove it from the
  returned feature matrix.
- Remove `recording_id`, `split` and `quality_pass` from X.
- Return `quality_pass` as integer y.
- Ensure scenario forget rows are training rows only.
- Define retain rows as original training rows minus forget IDs.
- Raise clear `ValueError`/`FileNotFoundError` messages rather than returning an
  empty DataFrame.
- Do not mutate caller DataFrames in place.
- Do not load all relational tables when only the model table is needed.

## Tests

Create `tests/echoforge/test_data.py` covering:

- root/path resolution from the repository and notebook directory;
- fixed split counts;
- exact core forget counts;
- retain/forget disjointness and complete reconstruction of train;
- target/feature separation;
- rejection of unknown/extension scenarios;
- preservation of source DataFrames.

## Files allowed to change

- `src/echoforge/paths.py`
- `src/echoforge/data.py`
- `src/echoforge/__init__.py` for public exports only
- `tests/echoforge/test_data.py`

Do not create notebooks, preprocessors, models or results.

## Acceptance criteria

- All required loaders return validated, non-empty copies.
- Exact split and scenario counts match the protocol.
- Features contain no target, split or recording identifier.
- Focused tests cover failure paths as well as successful loading.
- No dataset, model, notebook or Adult Income file changed.

## Verification

```bash
python scripts/verify_echoforge_dataset.py
python -m pytest tests/echoforge/test_dataset_installation.py tests/echoforge/test_data.py -q
python -m compileall -q src/echoforge
```

Include a brief example in the final response showing the returned train,
retain and forget counts for S1. Do not begin US-03.
