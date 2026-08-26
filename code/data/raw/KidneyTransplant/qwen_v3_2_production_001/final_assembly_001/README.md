# Qwen v3.2 kidney-transplant final assembly

This directory is an immutable, local-only assembly of the completed synthetic Qwen v3.2 production run. All data are entirely synthetic and must not be interpreted as real patient data.

- `kidney_transplant_assessments.csv` is the 60,000-row longitudinal assessment dataset.
- `kidney_transplant_identity.csv` contains the separate synthetic recipient and donor identities. It must never be joined into classifier inputs.
- `classifier_feature_list.json` is the authoritative list of fields the classifier may use.
- The target column is `acute_rejection_within_30_days`.
- `data_dictionary.csv` documents every assessment and identity field.
- `deletion_scenario_audit.csv` identifies future forget-set scenarios without deleting records.
- The JSON reports document reconciliation, validation, quality, integrity and assembly.
- Raw requests, complete responses, manifests and per-chunk validation evidence remain archived unchanged in `../chunk_001/` through `../chunk_100/`.

Do not manually edit any file in this directory. Create a separately authorised, reproducible successor stage for any future transformation. No data split, model training, classifier execution, or forget/retain dataset creation occurred in this stage.
