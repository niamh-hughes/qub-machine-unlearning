# US-15 — Reproduction, cleanup and research handoff

---

You are implementing **US-15 only: final reproducibility and handoff**.

Read completely:

- `AGENTS.md`
- all files in `docs/`
- `CODEX_STORY_INDEX.md`
- all EchoForge code, tests, notebook configurations and final result schemas

US-14 must be complete. This story does not add a new model, metric, scenario or
experiment.

## Goal

Make the completed EchoForge study understandable and safely reproducible by a
fresh reviewer without hiding failed runs or disturbing prior research work.

## Required work

1. Audit the EchoForge tree for:
   - duplicate/dead implementation paths;
   - notebook-defined reusable functions that belong in `src/echoforge/`;
   - absolute paths or usernames;
   - blind CSV append operations;
   - feature leakage from IDs/metadata;
   - missing configuration, seed or status fields;
   - result/checkpoint filename collisions;
   - changes under protected Adult paths.
2. Fix only demonstrated issues. Do not broadly refactor working code for style.
3. Create `REPRODUCING_ECHOFORGE.md` containing:
   - environment setup;
   - dataset ZIP placement and verification;
   - exact story/notebook execution order;
   - development versus final run modes;
   - expected artifact locations;
   - approximate runtime/device notes;
   - troubleshooting for MPS/CUDA/CPU and notebook timeouts;
   - limitations: synthetic data, frozen preprocessing, behavioural rather than
     formal deletion evidence, three-seed uncertainty and SISA reference rules.
4. Create `scripts/check_echoforge_artifacts.py` that performs read-only checks:
   - required files exist;
   - result schemas and keys are valid;
   - all expected seed/scenario/method combinations are present or failed
     explicitly;
   - prediction IDs reconcile with manifests;
   - probabilities and finite metrics are valid when status is completed;
   - figures are non-empty files;
   - no test/global IDs entered training logs where ID logs exist.
5. Update the root README with a short final EchoForge status and link to the
   reproduction guide. Preserve existing research context.
6. Execute the full focused test suite, dataset verifier and artifact checker.
7. Smoke-execute notebook 00 and notebook 07 from clean kernels. Do not rerun
   expensive training merely for cleanup unless a required artifact is invalid.
8. Produce a final inventory CSV under `results/echoforge/logs/` containing
   relative artifact path, size and SHA-256 hash.

## Prohibited actions

- No destructive cleanup commands.
- No removal of failed/negative results.
- No regeneration of data to repair a model result.
- No new hyperparameter choice.
- No Adult Income edits.
- No claim that the study proves legal compliance or exact parameter erasure.

## Acceptance criteria

- The dataset, tests and artifact checker pass.
- Reproduction instructions work from the project root.
- Required outputs have valid schemas, hashes and provenance.
- Final notebooks execute without hidden state.
- No new experiment or protected Adult change was introduced.

## Verification

```bash
python scripts/verify_echoforge_dataset.py
python -m pytest tests/echoforge -q
python scripts/check_echoforge_artifacts.py
jupyter nbconvert --to notebook --execute notebooks/echoforge/00_data_validation.ipynb --output 00_data_validation.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute notebooks/echoforge/07_results_comparison.ipynb --output 07_results_comparison.ipynb --output-dir notebooks/echoforge --ExecutePreprocessor.timeout=900
```

Return a final inventory summary, verification results, known limitations and
exact paths for paper-ready tables/figures. Do not start additional research.
