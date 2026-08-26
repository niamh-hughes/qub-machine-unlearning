# Results and prediction schemas

## Method result files

Each model/method writes exactly one deterministic CSV beneath
`results/echoforge/metrics/`:

```text
original.csv
full_retraining.csv
retain_finetuning.csv
gradient_ascent.csv
gradient_difference.csv
sisa.csv
```

The comparison notebook creates `all_results.csv` from these files. Method
notebooks must not append to `all_results.csv`.

## Required result columns

| Column | Meaning |
|---|---|
| `dataset_version` | fixed dataset name/version |
| `model_seed` | 42, 43 or 44 |
| `scenario_id` | S1, S3, S4 or S5; `GLOBAL` for original utility |
| `method` | stable method identifier |
| `architecture` | `monolithic_mlp` or `sisa_ensemble` |
| `run_phase` | `pilot_grid` or `final_repeat` |
| `config_id` | deterministic short configuration name |
| `checkpoint_stage` | selected epoch/slice/checkpoint |
| `train_count` | rows used in the update/training run |
| `forget_count` | scenario forget rows |
| `retain_count` | scenario retain rows |
| `affected_shards` | integer for SISA; blank otherwise |
| `f1` | global-test F1 for the pass class |
| `auroc` | global-test AUROC |
| `probability_distance` | paired forget probability MAD to reference |
| `truth_ratio_ks_d` | KS effect statistic |
| `truth_ratio_ks_p` | KS p-value |
| `runtime_seconds` | measured method/training runtime |
| `reference_runtime_seconds` | matching full-retraining runtime |
| `speedup` | reference runtime divided by method runtime |
| `device` | `cpu`, `mps` or `cuda` |
| `status` | `completed`, `diverged`, `invalid` or `failed` |
| `error_message` | blank on success; concise failure description otherwise |

Use `NaN` only when a field is genuinely not applicable. Do not substitute
zero for a missing metric.

## Prediction files

Save selected prediction files under:

```text
results/echoforge/predictions/<method>/<scenario>_seed_<seed>_<set>.csv
```

Required columns:

```text
recording_id
quality_pass
predicted_probability
predicted_class
per_record_bce
true_class_probability
```

Reference prediction files may additionally include `model_seed`,
`scenario_id` and `method`. Preserve row order or merge comparisons explicitly
by `recording_id`; never assume two prediction files already have the same
order.

## Deterministic write policy

Each notebook builds rows in a Python list, converts the complete list to a
DataFrame, validates its key uniqueness, sorts it and writes its method CSV
once. Rerunning a notebook replaces that notebook's outputs. Blind CSV append is
prohibited.

Suggested unique key:

```text
dataset_version, model_seed, scenario_id, method, config_id, checkpoint_stage
```
