# Kidney-Transplant Machine-Unlearning Submission

This directory is the clean, self-contained submission workspace for the current kidney-transplant machine-unlearning study. Everything outside `final_submission/` is working material or archived evidence and is not part of this submission copy.

## Study scope

The study predicts whether acute rejection will occur within 30 days after a kidney-transplant assessment. All recipient, donor, identity and clinical data in this workspace are entirely synthetic; no real patient records are included.

The modelling target is:

`acute_rejection_within_30_days`

## Authoritative data

- Final assessment dataset: `data/final/kidney_transplant_assessments.csv`
- Separate identity table: `data/final/kidney_transplant_identity.csv`
- Authoritative classifier feature list: `data/final/classifier_feature_list.json`
- Data dictionary: `data/final/data_dictionary.csv`
- Integrity, quality, reconciliation, deletion-scenario and assembly evidence: `data/final/`

The identity table is retained separately for audit and deletion-scenario joins. It is not a classifier-feature source.

## Data-generation documentation

`notebooks/00_dataset_generation.ipynb` is a source-only copy of the generation and validation notebook. Its outputs and execution counts have been cleared. It documents how the accepted dataset was produced and should not normally be rerun.

The frozen v3.2 prompt, API configuration and TOFU-inspired synthesis-method report are under `docs/data_generation/`. The copied synthesis report contains a historical absolute path identifying the external TOFU reference used during conformance review; the external paper itself is not included.

## Modelling status and future work

Notebook 01 is complete at `notebooks/01_baseline_model.ipynb`. It is organised into the approved 12-section MSc presentation, with concise explanations, focused code cells and recalculated findings. It reproduces the permanent seed-42 donor-grouped train, validation and test splits, fits training-only preprocessing, trains the weighted PyTorch baseline, selects its threshold using validation predictions and evaluates the frozen model once on the untouched test set.

The first execution stopped during its final CSV round-trip audit because of an overly strict floating-point tolerance. That attempt is preserved at `results/baseline/baseline_attempt_001_failure_report.json`. The corrected baseline and its presentation-only refactor both completed successfully: split membership remained exact, the preprocessor and model reloaded, probabilities reproduced within the approved tolerance, and saved metrics were independently recalculated.

Baseline processed data, checkpoint, fitted preprocessor, reports, predictions and figures are under `processed_data/`, `models/baseline/` and `results/baseline/`.

Notebook 02 is complete at `notebooks/02_deletion_scenarios_and_full_retraining.ipynb`. All five deletion scenarios passed structural, membership, metric, truth-ratio, artifact, configuration and model-output verification. The five full-retraining reference models, scenario-specific preprocessors and comparison outputs are under `models/full_retraining/` and `results/full_retraining/`.

Notebook 03 is complete at `notebooks/03_retain_set_fine_tuning.ipynb`. All seven final acceptance requirements passed. Five independently reloaded fine-tuned models reproduced their retained-test and training-forget probabilities within numerical tolerance. Utility, truth-ratio and KS outputs are under `results/retain_set_fine_tuning/`, while the saved models are under `models/retain_set_fine_tuning/`. Runtime findings in Sections 7.4 and 9.1 are generated directly from the execution's efficiency table because wall-clock time is hardware-dependent; a descriptive `1.10×` threshold distinguishes a meaningful runtime advantage and does not affect training, model selection or scientific metrics.

Notebook 04 is complete at `notebooks/04_gradient_ascent.ipynb`, with a review export at `notebooks/04_gradient_ascent.html`. It evaluates one predefined controlled gradient-ascent configuration across the same five deletion scenarios, using the frozen baseline preprocessor and a retained-validation safety guardrail. All nine final acceptance requirements passed; five independently reloaded checkpoints reproduced retained-test and training-forget probabilities, and all retained-utility metrics and composite scores were independently recalculated from saved predictions. Models and results are under `models/gradient_ascent/` and `results/gradient_ascent/`.

`notebooks/05_sisa.ipynb` is the main **Standard request-agnostic SISA** experiment, with its review export at `notebooks/05_sisa.html`. Recipients are assigned to five shards and five slices using a deterministic hash without knowledge of future deletion requests. Its clean execution completed 75 sequential code cells and all 12 final acceptance requirements. Every scenario affected all five shards from slice 1, replayed all 25 stages and reused no final shard model. The recalculated mean Composite Model Utility is 0.322200 and the mean KS statistic is 0.150551. Current hash-based artifacts are isolated under `processed_data/sisa/hash_based/`, `models/sisa/hash_based/` and `results/sisa/hash_based/`. Historical pre-renaming notebook evidence is retained outside the presentation notebook sequence under `docs/data_generation/sisa_historical_evidence/`.

`notebooks/05_policy_aware_sisa.ipynb` is the additional **Policy-Aware SISA Sensitivity Experiment**, with its review export at `notebooks/05_policy_aware_sisa.html`. It describes an intentionally favourable configuration that used predefined deletion-request memberships to concentrate affected records into fewer shards and later slices. Its most recent execution failed at code cell 10 because it checked the standard SISA prerequisite while that prerequisite's run summary was temporarily not yet marked completed; the policy-aware run summary therefore remains `running`, and its remaining cells did not execute in that attempt. The numerical policy-aware tables currently present are preserved historical results, byte-identical to the archived evidence under `docs/data_generation/sisa_historical_evidence/artifacts/`; they must not be described as products of the failed rerun. The byte-identical pre-title notebook is preserved at `docs/data_generation/sisa_historical_evidence/05_policy_aware_sisa_hash_reference.ipynb`. This optimistic historical sensitivity result does not replace standard request-agnostic SISA or establish universal efficiency, erasure or privacy.

`notebooks/05_provider_aware_sisa.ipynb` is the additional **Provider-Aware SISA Experiment**, with its review export at `notebooks/05_provider_aware_sisa.html`. Before any deletion membership was loaded, it froze five hospital-pair shards using deterministic greedy balancing of predeclared hospital provenance and then assigned donor-connected recipient components to five chronological slices. The clean execution completed 43 sequential code cells and all 11 acceptance requirements. The assignment exposed a valid negative structural result: every deletion scenario still affected all five shards from slice 1, so every scenario replayed all 25 stages and reused no final shard model. Mean Composite Model Utility was 0.323358, mean KS statistic was 0.150056 and total selective-retraining time was 20.0405 seconds (hardware-dependent). Nine donor components spanned hospitals, producing 6,630 cross-hospital spillover assessment rows under the primary-hospital assignment rule. Provider-aware models and results are isolated under `models/sisa/provider_aware/` and `results/sisa/provider_aware/`; the frozen assignments, pre-deletion audit, reconciliation evidence and integrity reports are retained with the results. This experiment neither replaces standard request-agnostic SISA nor treats its negative locality outcome as a failed run.

Notebook 06 is complete at `notebooks/06_gradient_difference.ipynb`, with its review export at `notebooks/06_gradient_difference.html`. It implements the TOFU Gradient Difference objective `-L_forget + L_retain` as a binary-classification adaptation: each forget assessment is paired with one reproducibly sampled retained-training assessment, every scenario starts from a fresh baseline copy, and five fixed epochs are run with epoch 5 as the primary result. All 12 final acceptance requirements passed. Mean epoch-5 Composite Model Utility was 0.340444, mean truth-ratio KS statistic was 0.050937, and total measured update time was 0.9452 seconds. All final models reloaded with zero probability difference, and saved utility, truth-ratio and KS results reproduced. Models and results are under `models/gradient_difference/` and `results/gradient_difference/`. The saved preprocessor was created with scikit-learn 1.9.0 and loaded here with 1.8.0; output reproduction passed, but the version mismatch remains an explicit environment limitation.

Notebooks 01–06 and the two additional SISA sensitivity/structural experiments now cover the current method-specific work.

The root-level `unlearning_evaluation.py` was not copied. The submission-file audit classified it as Adult/shared legacy code with no dependency from the current kidney notebooks, so its suitability requires review before any later inclusion.

CastMe is not included at this stage. It may be added later only if it is confirmed as part of the final dissertation.

## Integrity

`submission_manifest.csv` records the source and submission-copy hashes for copied files. The authoritative final-assembly manifest remains at `data/final/file_hash_manifest.json`.
