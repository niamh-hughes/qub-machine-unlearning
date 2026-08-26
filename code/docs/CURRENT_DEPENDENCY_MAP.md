# Current Dependency Map

## Audit scope

This map describes the repository as inspected without executing notebooks or modifying existing artifacts. The accepted classifier dataset is the v3.2 final assembly. The repository-level `AGENTS.md`, `README.md`, `INSTALL.md`, `CODEX_STORY_INDEX.md`, and existing `docs/` files describe an unfinished EchoForge project rather than the current kidney-transplant study; they therefore cannot serve as the final kidney submission documentation without a later, explicitly authorised rewrite.

## 1. Authoritative current kidney-transplant files

The authoritative assessment dataset is:

`data/raw/KidneyTransplant/qwen_v3_2_production_001/final_assembly_001/kidney_transplant_assessments.csv`

It contains 60,000 assessments, 10,000 recipients, six scheduled assessments per recipient, the 18 approved classifier features, and target `acute_rejection_within_30_days`.

The separate identity table is:

`data/raw/KidneyTransplant/qwen_v3_2_production_001/final_assembly_001/kidney_transplant_identity.csv`

It contains 18,000 synthetic recipient/donor identities and must remain separate from classifier inputs. The same directory's `classifier_feature_list.json`, `data_dictionary.csv`, `README.md`, deletion audit, integrity/quality/reconciliation/assembly reports, and hash manifest form the compact authoritative release package.

No existing model or result was found to have been trained from this final assembly.

## 2. Notebook 00 dependencies

`notebooks/KidneyTransplant/00_generate_and_validate_dataset.ipynb` is a large historical generation notebook. A full replay or forensic review depends on:

- `data/raw/KidneyTransplant/qwen_v3_1_design/` for the 31-column assessment schema, identity schema, feature list, canonical ranges, early controls and prompt material;
- `data/raw/KidneyTransplant/qwen_v3_2_design/` for the final prompt, API configuration, anchor specification and confirmation criteria;
- `data/raw/KidneyTransplant/qwen_v3_2_production_design_001/` for the frozen 10,000-recipient skeleton, assessment/event schedules, identities, anchors and production reports;
- `data/raw/KidneyTransplant/qwen_v3_2_production_001/chunk_001/` through `chunk_100/` for requests, complete responses, manifests, validated checkpoints and chunk reports;
- progress checkpoints, validation amendments and manual review under `qwen_v3_2_production_001/`;
- historical pilots, diagnostic batches, confirmation evidence and the protected old dataset for earlier notebook sections and fingerprint checks;
- the external files `../docs/references/USAGE.md` and `../docs/references/TOFU.pdf`, which sit outside this repository root. These are broken dependencies for a standalone submission of `code/` unless separately supplied or cited in a portable way;
- the `openai` client and hosted endpoint only for historical generation cells. The accepted final assembly itself can be audited locally without API access.

Moving or deleting any of those evidence directories would break full Notebook 00 replay and its stored fingerprint assumptions. They are evidence dependencies, not classifier inputs.

## 3. Dependencies of future baseline and unlearning notebooks

The future current-study pipeline should begin from the authoritative assessment CSV, the separate identity CSV for deletion joins only, `classifier_feature_list.json`, `data_dictionary.csv`, and `deletion_scenario_audit.csv`. It will also need a new deterministic grouped split keyed by `recipient_id`, a fitted preprocessor, baseline checkpoint, scenario forget/retain indices, full-retraining references, and method-specific checkpoints/results. None of those current-v3.2 downstream artifacts exists yet.

The existing Notebooks 01–05 do **not** meet that dependency contract:

- Notebook 01 reads `data/raw/kidney_transplant_unlearning_dataset.csv` and writes `data/processed/KidneyTransplant/kidney_transplant_assessments_clean.csv`.
- Notebooks 02–05 read that processed CSV, baseline artifacts under `models/KidneyTransplant/baseline/` and `results/KidneyTransplant/baseline/`, then consume the older full-retraining and method outputs.
- None of Notebooks 01–05 references the v3.2 final assessment or identity CSV.

They must therefore be revised or replaced in a separately authorised stage before any current v3.2 training occurs.

## 4. Older kidney-transplant experiment

The older chain consists of `data/raw/kidney_transplant_unlearning_dataset.csv`, `data/processed/KidneyTransplant/`, Notebooks 01–05 and their HTML exports, `models/KidneyTransplant/`, and `results/KidneyTransplant/`. The outputs are internally linked through baseline manifests, splits, preprocessing objects, scenario indices and checkpoints, but they belong to the earlier dataset. They should be archived together rather than partially mixed into the new submission.

The `qwen_v3_design/` directory and early Qwen pilot/batch/diagnostic directories are also superseded development history. They remain useful evidence but are not the authoritative dataset.

## 5. Evidence directories versus classifier data

- Classifier data: only the 31-column final assessment CSV, restricted to the feature list and target at modelling time.
- Identity/audit data: the separate identity CSV, deletion audit, dates, consent and identifiers; never classifier features.
- Compact final evidence: reports and hash manifest in `final_assembly_001/`.
- Frozen design evidence: `qwen_v3_1_design/`, `qwen_v3_2_design/`, and `qwen_v3_2_production_design_001/`.
- Raw generation archive: 100 production chunk directories containing request/response/manifest/checkpoint records.
- Historical development evidence: pilots, corrected-ten, diagnostic-100 and confirmation batches.

## 6. Consequences of moving or deleting major directories

| Directory/file | Consequence |
|---|---|
| `final_assembly_001/` | Removes the authoritative current dataset, identity separation, feature contract and final validation evidence. |
| `chunk_001/`–`chunk_100/` | Prevents complete response reconciliation and breaks Notebook 00's immutable evidence/fingerprint checks. |
| v3.1/v3.2 design and production-design directories | Removes schemas, prompts, anchors, event schedules and deterministic skeleton needed to explain or replay generation. |
| old raw/processed kidney dataset | Breaks existing Notebooks 01–05 and all old kidney checkpoints/results, but does not invalidate the already assembled v3.2 CSV. |
| `models/KidneyTransplant/` or `results/KidneyTransplant/` | Breaks the old unlearning notebooks' reload/reconciliation steps; these artifacts are not proven current-v3.2 outputs. |
| Adult directories | Breaks protected Adult notebooks; exclude only from a separate submission copy, never alter the working repository. |
| CastMe directories | Impact depends on Niamh's submission-scope decision; do not remove automatically. |
| EchoForge scaffold/docs | Removes a separate planned project; exclude from a kidney-only copy only after confirming it is not part of the assessed submission. |

## Broken and non-portable references

- Notebook 00 computes `ROOT.parent / "docs/references"`; `USAGE.md` and `TOFU.pdf` are outside the stated repository root.
- The conformance-addendum JSON files store absolute `/Users/niamhhughes/...` paths to those references.
- Stored outputs in several notebooks contain absolute local paths. Kidney notebook source cells themselves use root discovery rather than hard-coded `/Users/...` strings, but output clearing or path sanitisation should be considered in a future submission copy.
- Repository documentation names missing intended EchoForge files such as `data/echoforge/data/processed/echoforge_model_table.csv` and an absent `notebooks/echoforge/` tree. Those are broken for EchoForge, although unrelated to the kidney dataset.
