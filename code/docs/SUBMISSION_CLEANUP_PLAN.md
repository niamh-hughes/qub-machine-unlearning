# Submission Cleanup Plan

## Scope and safety language

This is a proposal only. No cleanup was performed.

- **Exclude from the submission copy** means omit a file while copying into a new submission directory; the working repository remains untouched.
- **Archive outside the submission** means preserve the original in a separate, checksummed archive and omit it from the lightweight submission.
- **Permanently delete** means destroy the only copy. This audit does not recommend permanent deletion for any research artifact; exclusion or archiving is sufficient.

## Repository totals

- Files inspected at audit start: **2,472**.
- Repository size at audit start: **690,242,620 bytes (658.27 MiB)**.
- Audit files added: **3**; repository after audit: **2,475 files, 691,224,858 bytes (659.20 MiB)**.
- Estimated clean submission from files marked `INCLUDE`: **18,889,735 bytes (18.01 MiB)**. This excludes raw archives and assumes that `INCLUDE_SUMMARY_ONLY` material is represented by compact summaries rather than copied wholesale.
- Raw production directory: approximately **196.64 MB**; all raw KidneyTransplant material: approximately **224.19 MB**.
- Exact-content duplicate scan: **150 groups / 407 files**, with approximately **30.75 MB** of redundant copies. Many are legitimate checkpoint-versus-final-checkpoint audit duplicates, so hash duplication alone is not a deletion instruction.

## Counts by classification

| Classification | Files | Bytes | Human size |
|---|---:|---:|---:|
| REQUIRED_CURRENT | 7 | 17,814,516 | 16.99 MiB |
| CURRENT_OUTPUT | 9 | 1,075,219 | 1.03 MiB |
| REPRODUCIBILITY_EVIDENCE | 99 | 33,198,428 | 31.66 MiB |
| RAW_GENERATION_ARCHIVE | 1,300 | 173,845,859 | 165.79 MiB |
| LEGACY_KIDNEY | 228 | 50,103,877 | 47.78 MiB |
| UNRELATED_PROJECT | 541 | 364,212,565 | 347.34 MiB |
| TEMPORARY_OR_CACHE | 10 | 102,913 | 100.50 KiB |
| DUPLICATE_CANDIDATE | 18 | 17,382,498 | 16.58 MiB |
| REVIEW_REQUIRED | 263 | 33,488,983 | 31.94 MiB |

## Counts by recommended action

| Action | Files | Bytes | Human size |
|---|---:|---:|---:|
| ARCHIVE_OUTSIDE_SUBMISSION | 1,528 | 223,949,736 | 213.58 MiB |
| EXCLUDE_FROM_SUBMISSION | 569 | 381,697,976 | 364.02 MiB |
| INCLUDE | 16 | 18,889,735 | 18.01 MiB |
| INCLUDE_SUMMARY_ONLY | 99 | 33,198,428 | 31.66 MiB |
| REVIEW_BEFORE_DECISION | 263 | 33,488,983 | 31.94 MiB |

## 1. Proposed clean submission tree

```text
submission/
├── README.md                         # new kidney-specific submission README, authorised later
├── requirements.txt
├── docs/
│   ├── SUBMISSION_FILE_AUDIT.csv
│   ├── SUBMISSION_CLEANUP_PLAN.md
│   ├── CURRENT_DEPENDENCY_MAP.md
│   └── method_and_provenance_summary.md   # distilled later from Notebook 00/evidence
├── data/
│   └── KidneyTransplant/
│       └── final_assembly_001/
│           ├── kidney_transplant_assessments.csv
│           ├── kidney_transplant_identity.csv
│           ├── classifier_feature_list.json
│           ├── data_dictionary.csv
│           ├── deletion_scenario_audit.csv
│           ├── final_*_report.json
│           ├── file_hash_manifest.json
│           └── README.md
├── notebooks/
│   └── KidneyTransplant/
│       ├── 00_generate_and_validate_dataset.ipynb  # optional source-only evidence copy
│       └── 01–05 current-v3.2 notebooks            # do not use existing legacy versions unchanged
├── src/                              # future current-v3.2 reusable training/evaluation code
├── models/                           # only selected current-v3.2 checkpoints, once created
└── results/                          # compact current-v3.2 metrics/tables, once created
```

The proposed tree is conceptual. Do not create it until the notebook migration and CastMe scope decisions are complete.

## 2. Files confidently excludable from a kidney-only submission copy

- `.DS_Store`, `__pycache__/` and `.pyc` files.
- Adult Income, bank, hotel and EchoForge files, subject to preservation in the working repository.
- Static HTML notebook exports when the `.ipynb` is retained; especially filenames containing ` 2.html`.
- The duplicate Qwen test script under `notebooks/KidneyTransplant/` after choosing the canonical `scripts/test_qwen_api.py` copy.
- Old top-level Adult result tables/figures that do not belong to the kidney study.

These are exclusions from a new copy, not deletion instructions.

## 3. Files to archive rather than submit

- All 100 `qwen_v3_2_production_001/chunk_NNN/` directories, with their current hashes and directory structure preserved.
- Qwen pilots, corrected-ten, diagnostic-100, confirmation run, design iterations, amendments, manual review and progress checkpoints. Submit concise final reports; archive full evidence.
- `data/raw/kidney_transplant_unlearning_dataset.csv`, `data/processed/KidneyTransplant/`, existing Kidney Notebooks 01–05, `models/KidneyTransplant/` and `results/KidneyTransplant/` as one internally linked legacy package.
- Superseded HTML exports and earlier Qwen design versions if historical presentation evidence is required.

## 4. Items requiring Niamh's decision

1. Whether CastMe is part of the assessed submission. Every CastMe file is `REVIEW_REQUIRED`, not automatically removable.
2. Whether EchoForge is a separate future project or part of this submission. Current root documentation describes EchoForge and conflicts with the kidney scope.
3. Whether the submission needs a source-only Notebook 00 or only a method/provenance summary plus archived raw evidence.
4. Whether to include the separate synthetic identity table in the submitted package or retain it in a controlled audit appendix. It is authoritative but must remain outside classifier inputs.
5. Whether old Kidney Notebooks 01–05 and results are needed as historical comparison. They must not be presented as outputs of the v3.2 dataset.
6. Whether supervisor `USAGE.md` and TOFU reference material may be redistributed. Notebook 00 currently depends on copies outside this repository.
7. Whether `requirements.txt` should be frozen with versions and include `openai` for generation reproducibility; current training dependencies are unpinned and `openai` is absent.

## 5. Largest submission obstacles

The largest individual files are mostly unrelated Adult results: `mia_record_scores.csv` (52.09 MB), `mia_scores.csv` (36.92 MB), and a rebuilt direct-MIA table (23.79 MB). The largest current/legacy kidney files are the old processed assessment CSV (19.89 MB), old protected raw CSV (19.82 MB), authoritative final assessment CSV (14.63 MB), and production assessment metadata (12.48 MB). Raw Qwen evidence comprises many smaller files but approximately 196.64 MB in aggregate.

## 6. Staged cleanup procedure for a future authorised copy

1. Freeze the working repository and record a content-hash manifest.
2. Create a new sibling submission directory; never clean in place.
3. Copy only `INCLUDE` files from this audit.
4. Resolve every `REVIEW_REQUIRED` row, especially CastMe and root EchoForge documentation.
5. Produce compact summaries for selected `INCLUDE_SUMMARY_ONLY` evidence; keep originals in a checksummed archive.
6. Copy archive-designated material into a separate archive, preserving paths and manifests.
7. Rewrite or replace Notebooks 01–05 so they consume the v3.2 final assessment CSV and use the identity table only for deletion joins.
8. Strip notebook outputs and local paths only in the submission copy; retain original executed notebooks in the archive.
9. Add current-v3.2 models/results only after they have been generated and validated.
10. Run the clean-copy verification below before packaging.

## 7. Rollback strategy

Because cleanup should occur only in a new directory, rollback is simply discarding the unaccepted submission copy. The working repository and external evidence archive remain unchanged. Record the source-tree hash manifest before copying and the submission-tree manifest after copying. Never use in-place deletion as rollback.

## 8. Final clean-copy verification

1. Confirm the assessment CSV SHA-256 matches the final assembly manifest and has 60,000 rows/31 columns.
2. Confirm the identity CSV has 18,000 rows and is not referenced as a classifier feature source.
3. Verify the classifier feature list contains exactly the approved 18 features and target.
4. Search all submission text/notebook sources for `/Users/`, `Desktop - Niamh`, parent-directory supervisor references, API keys and endpoint URLs.
5. Confirm no Adult, bank, hotel, unresolved CastMe or EchoForge files remain unless explicitly approved.
6. Confirm no `.DS_Store`, `__pycache__`, `.pyc`, duplicate HTML or ` 2.html` files remain.
7. Confirm each included notebook has cleared outputs, portable root discovery and no broken inputs.
8. Confirm revised Notebooks 01–05 load the v3.2 final assessment dataset, not the old protected/processed CSV.
9. Recompute a clean-copy SHA-256 manifest and compare every copied file with its approved source.
10. Open the package on a different path or clean environment and perform read-only dependency checks before submission.
