# US-00 — Audit and scaffold the EchoForge workspace

Paste the prompt below into Codex while the existing `code/` directory is the
workspace root.

---

You are implementing **US-00 only: repository audit and safe EchoForge
scaffolding**.

Read completely before editing:

- `AGENTS.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/NOTEBOOK_STANDARD.md`

## Goal

Prepare a clean, minimal EchoForge area inside the existing research codebase
without changing or deleting any Adult Income work. This story creates folders,
package markers and basic project documentation only. It must not install the
dataset, train a model, generate results or create notebooks with fake content.

## Required work

1. Inspect the workspace, including current git status when available. Report
   existing unrelated modifications before editing, but work around them.
2. Confirm that `AGENTS.md` is at the workspace root. Stop if it is missing.
3. Create the intended EchoForge directories from
   `docs/PROJECT_STRUCTURE.md`, including:
   - `incoming/`
   - `data/echoforge/`
   - `src/echoforge/`
   - `tests/echoforge/`
   - `notebooks/echoforge/`
   - `models/echoforge/{original,full_retraining,sisa}/`
   - `results/echoforge/{metrics,predictions,figures,configurations,logs}/`
   - `scripts/`
4. Add `src/echoforge/__init__.py` containing only a short package docstring.
5. Use `.gitkeep` only where an otherwise empty output directory must be
   preserved. Do not create placeholder CSV, model or notebook files.
6. If `.gitignore` exists, extend it minimally; otherwise create it. Preserve
   all existing entries. Ignore `.DS_Store`, `__pycache__/`, `*.pyc`,
   `.ipynb_checkpoints/`, `.venv/`, temporary logs and
   `results/echoforge/tmp/`. Do not ignore final result CSVs or figures.
7. Update the empty or minimal root `README.md` with:
   - project purpose;
   - the protected status of Adult Income work;
   - the numbered notebook plan;
   - the instruction to run one Codex story at a time;
   - links to the four documentation files above.
8. Update `requirements.txt` only if it is empty. Use a minimal unpinned list:
   `numpy`, `pandas`, `scikit-learn`, `scipy`, `torch`, `matplotlib`,
   `seaborn`, `joblib`, `jupyter`, `nbformat`, `nbclient`, and `pytest`.
   Do not install packages during this story.

## Files allowed to change

- `.gitignore`
- `README.md`
- `requirements.txt`
- new EchoForge directories/package markers

Do not alter anything under an Adult path or `unlearning_evaluation.py`.

## Acceptance criteria

- The intended folders exist with no fake experiment outputs.
- `src/echoforge` imports successfully.
- Existing Adult Income files are byte-for-byte untouched.
- README describes the execution order accurately.
- No training or data-extraction code has been added.

## Verification

Run:

```bash
python -m compileall -q src/echoforge
python -c "import src.echoforge; print('import ok')"
find src/echoforge notebooks/echoforge tests/echoforge results/echoforge models/echoforge -maxdepth 2 -type d | sort
```

When git is available, also show `git diff --stat` and confirm no protected
Adult paths changed.

Finish with the response format required by `AGENTS.md`. Do not begin US-01.

