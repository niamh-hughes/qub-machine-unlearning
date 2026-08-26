---
title: "Adult Income Machine-Unlearning"
subtitle: "Structured Research Results Report"
---

# Executive Summary

The baseline Adult Income MLP achieved **85.60% test accuracy**, **F1 = 0.675**, and **AUROC = 0.910**. Full retraining on each retain set is the empirical reference: it represents how the model behaves when the requested records are excluded from training.

- **Retain-set fine-tuning** preserved utility and provided partial, scenario-dependent evidence of approximate unlearning. The retention-expiry scenario (S5) showed the clearest movement toward full retraining.
- **Gradient ascent** preserved ordinary predictive utility and was fast, but selected checkpoints moved away from full retraining on forget-set probability and loss metrics.
- **SISA** has been rerun using the same current scenarios. All requests affected every shard in this five-shard configuration, so it did not provide a request-time advantage and had lower utility than the single-model baseline.
- **Membership-inference results are inconclusive** because the positive controls were invalid in the approximate-method notebooks.

# Experimental Framework

Each approximate method begins with the same baseline model. For each deletion request, it is compared with a separately retrained model that was trained without the relevant forget set.

| Scenario | Forget set | Purpose |
|---|---|---|
| S1 | Random 1% | Small random deletion |
| S2 | Random 5% | Moderate random deletion |
| S3 | Provider withdrawal (~5%) | Structured withdrawal |
| S4 | Random 10% | Larger random deletion |
| S5 | Retention expiry (16.3%) | Structured cohort deletion |

## Measures

| Dimension | Main measures | Desired result |
|---|---|---|
| Utility | Test F1, AUROC, accuracy deltas | High F1/AUROC; deltas near zero |
| Forgetfulness | Wasserstein distance, probability MAD, agreement with retraining | Low distances; high agreement |
| Efficiency | Deletion runtime compared with full retraining | Lower runtime ratio |
| MIA diagnostic | Attack AUROC and positive-control validity | Interpret only if control is valid |

# Reference Performance

| Model | Accuracy | F1 | AUROC | Interpretation |
|---|---:|---:|---:|---|
| Baseline MLP | 0.856 | 0.675 | 0.910 | Original model |
| Full retraining | 0.856–0.858 | 0.668–0.677 | 0.908–0.910 | Empirical unlearning reference |

Random deletions produce fully retrained models that are already close to the original baseline. This makes forgetting difficult to demonstrate because an approximate method must reproduce a small difference reliably.

# Retain-Set Fine-Tuning

| Scenario | Test F1 | AUROC | Forget W | Baseline W | Forget agreement | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 0.6762 | 0.9098 | 0.0107 | 0.0151 | 98.83% | 7.87 |
| S2 | 0.6731 | 0.9096 | 0.0097 | 0.0086 | 98.65% | 6.14 |
| S3 | 0.6780 | 0.9099 | 0.0075 | 0.0132 | 98.70% | 10.82 |
| S4 | 0.6782 | 0.9097 | 0.0137 | 0.0147 | 98.36% | 8.60 |
| S5 | 0.6758 | 0.9096 | 0.0266 | 0.0382 | 96.04% | 5.56 |

**Interpretation.** Utility remains stable. Forget-set Wasserstein distance improves over the untouched baseline in S1, S3, S4 and S5; S5 is the clearest structured-deletion result. The method does not exactly reproduce full retraining and is not consistently faster.

*Source: `results/retain_finetuning/adult/metrics/unlearning_evaluation.csv`.*

# Gradient Ascent

| Scenario | F1 vs baseline | AUROC vs baseline | Probability improvement | Loss improvement | Forget agreement |
|---|---:|---:|---:|---:|---:|
| S1 | -0.0009 | -0.0001 | -0.0890 | -0.0698 | 97.95% |
| S2 | +0.0018 | -0.0001 | -0.0292 | -0.0585 | 98.01% |
| S3 | +0.0015 | -0.0001 | -0.4250 | -0.3687 | 98.23% |
| S4 | +0.0015 | +0.0000 | -0.0614 | -0.0642 | 98.39% |
| S5 | +0.0139 | -0.0006 | -0.6654 | -0.5794 | 91.37% |

**Interpretation.** Utility is preserved: F1 and AUROC remain close to the baseline. Every selected checkpoint has negative probability and loss improvement, however, so gradient ascent does not provide convincing approximate-unlearning evidence under the selected configuration.

*Source: `results/gradient_ascent/adult/tables/behavioural_interpretation.csv` and Notebook 4 utility summary.*

# SISA Using Current Shared Scenarios

| Scenario | Test F1 | Delta F1 | Forget W | Baseline W | SISA runtime (s) | Full retrain (s) |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 0.6516 | -0.0250 | 0.0257 | 0.0151 | 14.13 | 6.16 |
| S2 | 0.6455 | -0.0304 | 0.0277 | 0.0086 | 14.12 | 6.31 |
| S3 | 0.6479 | -0.0201 | 0.0315 | 0.0132 | 13.25 | 4.58 |
| S4 | 0.6437 | -0.0301 | 0.0362 | 0.0147 | 12.84 | 5.08 |
| S5 | 0.6406 | -0.0269 | 0.0683 | 0.0382 | 13.78 | 5.08 |

**Interpretation.** SISA has lower utility in every scenario and its forget-set distance is higher than the untouched baseline. Every request affected all five shards and retrained all 25 slices, making it slower than full retraining in this configuration.

*Source: refreshed `results/sisa/adult/metrics/sisa_utility_table.csv`, `sisa_forget_set_fidelity_table.csv`, and `sisa_efficiency_summary.csv`.*

# Cross-Method Interpretation

| Method | Utility | Forgetfulness | Efficiency | Defensible conclusion |
|---|---|---|---|---|
| Retain fine-tuning | Preserved | Partial; strongest in S5 | Not consistently faster | Useful approximate baseline; scenario-dependent |
| Gradient ascent | Preserved | Not demonstrated | Very fast | Fast update, but not a retraining surrogate |
| SISA | Reduced | Not demonstrated | Slower in this partition | Partitioning did not isolate requests |

# Limitations and Next Steps

- Full retraining is an empirical reference, not a formal deletion guarantee.
- Results use one dataset, one model architecture and a fixed seed; repeat across multiple seeds before making general claims.
- Do not treat the MIA metrics as evidence of successful forgetting: positive controls were invalid in the approximate-method experiments.
- For SISA, test a partitioning strategy that reduces the likelihood of a deletion request affecting every shard.
- For gradient ascent, tune against a retraining-similarity objective while retaining validation-utility constraints.
