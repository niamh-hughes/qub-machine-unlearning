# Standard notebook structure

Every notebook in `notebooks/echoforge/` must restart and run top-to-bottom.
Use numbered Markdown headings in this order where applicable.

1. **Purpose and expected output**
2. **Imports and reproducibility configuration**
3. **Paths and fixed experiment configuration**
4. **Load and validate data/artifacts**
5. **Run the model or method**
6. **Evaluate**
7. **Save deterministic artifacts**
8. **Compact result table/plot**
9. **Checks, observations and limitations**

## Common configuration cell

Use an explicit small dictionary rather than hidden globals or a configuration
framework:

```python
NOTEBOOK_CONFIG = {
    "dataset_version": "echoforge_seed_42_v1",
    "model_seed": 42,
    "scenario_ids": ["S1", "S3", "S4", "S5"],
    "device_preference": "auto",
}
```

Method-specific parameters may extend this dictionary. Display it near the top
and save a JSON copy beside method results.

## Finding the project root

Do not hard-code a username or machine path. Find the nearest parent containing
`AGENTS.md`, then add that root to `sys.path` only once:

```python
from pathlib import Path
import sys

candidate_paths = [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next(path for path in candidate_paths if (path / "AGENTS.md").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

## Display policy

- Show a small configuration table and compact validation summary.
- Show a loss curve only when training occurs.
- Show one main result table per notebook.
- Avoid printing thousands of rows, model state dictionaries or full
  classification reports.
- Save detailed rows to CSV instead of leaving them only in notebook output.
- Add two to five factual observations at the end; do not write conclusions
  before results exist.

## Shared method layout

Notebooks 03–06 should use the same outer loop:

```python
result_rows = []
for scenario_id in NOTEBOOK_CONFIG["scenario_ids"]:
    # load scenario artifacts
    # run method/configurations
    # calculate matching metrics
    # append validated result rows
```

Reusable implementation must be imported from `src/echoforge/`. A notebook
should orchestrate functions, not define a second version of them.

