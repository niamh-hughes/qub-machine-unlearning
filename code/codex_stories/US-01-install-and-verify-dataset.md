# US-01 — Install and verify the EchoForge dataset

---

You are implementing **US-01 only: install and verify the supplied EchoForge
dataset**.

Read completely:

- `AGENTS.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `codex_stories/US-00-repository-audit-and-scaffold.md`

## Preconditions

- US-00 is complete.
- `incoming/EchoForge_Synthetic_Dataset_CSV_Package.zip` exists.

If the ZIP is missing, stop and report the exact expected path. Do not download,
recreate or substitute another dataset.

## Goal

Safely extract the supplied package into `data/echoforge/`, verify its file
checksums and validate its fixed scientific invariants. Dataset source files
become immutable after this story.

## Required work

1. Inspect the ZIP listing before extraction. Reject absolute paths, `..` path
   traversal or symbolic-link entries.
2. The ZIP contains a top-level package folder. Copy its contents—not the
   redundant wrapper—into `data/echoforge/`. Preserve the incoming ZIP.
3. Do not overwrite an existing non-identical dataset. If
   `data/echoforge/config.json` already exists, compare the installed manifest
   and stop on any conflict.
4. Create `scripts/verify_echoforge_dataset.py`. It must:
   - resolve paths relative to the repository root;
   - verify SHA-256 entries from `data/echoforge/file_manifest.csv` for all
     generator-owned files listed there;
   - load the model table, raw recording table, scenario definitions and all
     forget manifests;
   - verify 20,000 unique recording IDs;
   - verify train/validation/test counts 14,000/3,000/3,000;
   - verify binary target and pass rate between 0.55 and 0.65;
   - verify S1/S3/S4/S5 train forget counts 478/1,287/2,068/678;
   - verify every forget ID exists exactly once and belongs to the training
     split;
   - verify each retain set would be disjoint from its forget set;
   - return non-zero and a clear message on failure.
5. Add focused tests in `tests/echoforge/test_dataset_installation.py` for the
   same critical invariants. Tests must read the installed data and never mutate
   it.
6. Add a short `data/echoforge/LOCAL_DATASET_NOTES.md` identifying the installed
   package, immutable folders and verification command. Do not edit the supplied
   dataset README/config/manifests merely to change wording.

## Files allowed to change

- `data/echoforge/` through safe installation
- `scripts/verify_echoforge_dataset.py`
- `tests/echoforge/test_dataset_installation.py`

## Acceptance criteria

- Verification passes with no warnings hidden.
- All required dataset files are present.
- Exact core scenario counts match the protocol.
- The incoming ZIP remains available.
- No dataset-generation script has been run.
- No Adult Income file changed.

## Verification

```bash
python scripts/verify_echoforge_dataset.py
python -m pytest tests/echoforge/test_dataset_installation.py -q
```

Report the observed row count, split counts, pass rate and four forget counts.
Do not begin US-02.

