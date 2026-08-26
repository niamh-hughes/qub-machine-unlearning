# EchoForge fixed experiment protocol

## Dataset

- Synthetic EchoForge voice-production dataset, generation seed 42.
- 20,000 recordings.
- Target: `quality_pass`; 1 is the positive/pass class.
- Fixed split: 14,000 train, 3,000 validation and 3,000 test.
- Observed pass rate: approximately 0.6021.
- Default modelling file:
  `data/echoforge/data/processed/echoforge_model_table.csv`.
- Record/scenario metadata:
  `data/echoforge/data/raw/recordings.csv` and
  `data/echoforge/data/manifests/`.

## Core deletion scenarios

| ID | Reason | Forget rule | Expected train count | Structure |
|---|---|---|---:|---|
| S1 | Actor consent withdrawal | `actor_id == A07` | 478 | coherent actor cluster |
| S3 | Studio withdrawal | `studio_id == ST02` | 1,287 | source/domain cluster |
| S4 | Retention expiry | date before 2022-08-01 | 2,068 | temporal cohort |
| S5 | Invalid consent | `consent_batch_id == B17` | 678 | distributed audit batch |

S2 and S6 are documented extensions. Do not include them in the core run matrix.

## Standard model

- Binary PyTorch MLP.
- Shape: input -> 64 -> ReLU -> Dropout(0.2) -> 32 -> ReLU -> output.
- Loss: `BCEWithLogitsLoss`.
- Optimiser: AdamW.
- Apply sigmoid only for prediction/evaluation.
- Development seed: 42.
- Maximum baseline epochs: 50 with early stopping on validation loss.
- Save the selected epoch and training history.

## Preprocessing policy

This is a model-only unlearning study. Fit one preprocessor on the original
training split and save it. Reuse it for original, monolithic full-retraining
and approximate-method experiments. This keeps input dimensions fixed and
isolates neural-model updates. State this limitation in the report.

Never use `recording_id`, `split`, entity IDs, consent/licence/rights metadata,
row hashes or latent generator effects as model features.

## Reference models

- Monolithic approximate methods are compared with a fresh monolithic MLP
  trained on the scenario retain set.
- SISA unlearning is compared with a fresh SISA ensemble trained on the same
  retain data using the identical shard/slice/aggregation algorithm.
- Original-to-reference differences must be reported. If the difference is
  extremely small, label the scenario weakly identifiable rather than inflating
  a normalised score.

## Approximate methods

- Retain fine-tuning: learning rate 0.0001; 1 and 3 epochs.
- Gradient ascent: learning rates 0.00001 and 0.00005; 1 and 3 epochs;
  minimise `-forget_loss`; use gradient clipping.
- Gradient difference: the same conservative learning rates; 1 and 3 epochs;
  `lambda_retain` 0.5 and 1.0; minimise
  `lambda_retain * retain_loss - forget_loss`.

Pilot with S1 and S5 using model seed 42. Freeze selected configurations plus
one neighbouring strength before running every scenario. Do not tune a separate
favourable configuration for every scenario.

## SISA configuration

- Five shards.
- Three chronological slices per shard.
- Stable actor-based shard mapping using metadata only.
- One independent MLP per shard.
- Save checkpoints after every cumulative slice.
- Aggregate shard probabilities using an unweighted mean unless a documented
  predeclared alternative is required.
- Never change the partition after observing unlearning results.

## Metrics

Primary utility:

- F1 on the global test set.
- AUROC on the global test set.

Primary forgetting:

- Mean absolute paired probability distance between unlearned and matching
  full-retraining predictions on the same forget records.
- TOFU-inspired truth ratio
  `(1 - p_true) / (p_true + epsilon)` followed by a two-sample KS comparison
  between the unlearned and matching reference distributions. Report KS D and
  p-value; do not claim that a high p-value proves deletion.

Supporting:

- Wall-clock runtime measured on the same device with explicit timing bounds.
- SISA affected-shard count and speed-up over full SISA retraining.

## Seed policy

Implement and debug every stage with seed 42. Repeat only frozen selected
configurations with seeds 43 and 44. Report mean, standard deviation and all
three individual values.
