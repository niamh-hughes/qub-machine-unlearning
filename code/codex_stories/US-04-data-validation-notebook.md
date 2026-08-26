# US-04 — Create and execute the data-validation notebook

---

You are implementing **US-04 only: `00_data_validation.ipynb`**.

Read completely:

- `AGENTS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/NOTEBOOK_STANDARD.md`
- existing `src/echoforge/data.py`

US-02 must be complete. US-03 may exist, but this notebook must not train a
model or fit preprocessing.

## Goal

Create a concise, executed notebook that proves the installed dataset and four
core scenario manifests are safe to use.

## Required notebook content

Create `notebooks/echoforge/00_data_validation.ipynb` using the standard
notebook headings. It must:

1. Resolve the project root without a machine-specific path.
2. Display the fixed configuration and dataset version.
3. Load the model table, recording metadata and scenario definitions through
   `src.echoforge` functions.
4. Report:
   - table shape;
   - unique recording count;
   - target distribution/pass rate;
   - split counts and split target rates;
   - missing values and duplicated IDs;
   - numeric feature ranges and categorical cardinalities;
   - exact S1/S3/S4/S5 forget counts and pass rates;
   - retain/forget disjointness;
   - pairwise forget-set overlap counts.
5. Assert the protocol invariants. A failed assertion must stop execution.
6. Save one compact summary CSV to
   `results/echoforge/logs/data_validation_summary.csv` by replacing the file,
   never appending.
7. Include only useful displays: compact tables and at most two small plots
   (target by split and scenario size).
8. End with factual observations and the statement that no model was trained.

Do not duplicate loader/validation functions in notebook cells. If shared code
has a genuine bug, fix it minimally and add a regression test.

## Execution

Execute from a clean kernel using `nbclient`, `jupyter nbconvert --execute` or
the project's available equivalent. Save the executed notebook in place. Do
not rely on manually run hidden state.

## Acceptance criteria

- Clean top-to-bottom execution succeeds.
- Counts match 20,000 and 14,000/3,000/3,000.
- Core forget counts match 478/1,287/2,068/678.
- The saved summary has unique metric/scenario keys.
- No model/preprocessor/checkpoint exists as a side effect.

## Verification

```bash
python -m pytest tests/echoforge/test_dataset_installation.py tests/echoforge/test_data.py -q
jupyter nbconvert --to notebook --execute notebooks/echoforge/00_data_validation.ipynb --output 00_data_validation.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=600
```

Report the notebook path and summary CSV path. Do not begin US-05.

