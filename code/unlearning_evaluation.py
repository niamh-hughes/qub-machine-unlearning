"""Reference-Based Unlearning Evaluation helpers for Adult Income experiments.

The functions in this module are intentionally method-agnostic.  They compare
an approximate unlearning method against the full-retraining reference for one
forget scenario and return a single tidy row that can later be appended to SISA
or other methods.

The framework separates three dimensions:
1. Predictive utility.
2. Forget-set fidelity to full retraining.
3. Membership-inference diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class PredictionBundle:
    probabilities: np.ndarray
    predictions: np.ndarray
    losses: np.ndarray
    true_class_confidence: np.ndarray
    correctness: np.ndarray


PredictionSource = Union[torch.nn.Module, Callable[..., np.ndarray]]


def _device_for_model(model: torch.nn.Module) -> torch.device:
    """Infer the device used by a model."""
    return next(model.parameters()).device


def _tensor_dataset(features: pd.DataFrame, labels: pd.Series) -> TensorDataset:
    """Convert pandas features and labels into a PyTorch dataset."""
    feature_tensor = torch.tensor(
        features.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    label_tensor = torch.tensor(
        labels.to_numpy(dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
    )

    return TensorDataset(feature_tensor, label_tensor)


def _predict_with_loss(
    model: PredictionSource,
    features: pd.DataFrame,
    labels: pd.Series,
    batch_size: int = 256,
) -> PredictionBundle:
    """Generate probabilities, predictions, per-record BCE losses and confidence."""
    if not isinstance(model, torch.nn.Module) and callable(model):
        return _predict_with_loss_from_callable(
            predictor=model,
            features=features,
            labels=labels,
            batch_size=batch_size,
        )

    device = _device_for_model(model)
    data_loader = DataLoader(
        _tensor_dataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    probabilities = []
    losses = []

    model.eval()

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            batch_losses = criterion(logits, batch_labels)
            batch_probabilities = torch.sigmoid(logits)

            losses.extend(batch_losses.cpu().numpy().ravel())
            probabilities.extend(batch_probabilities.cpu().numpy().ravel())

    probabilities_array = np.array(probabilities)
    losses_array = np.array(losses)
    labels_array = labels.to_numpy()
    predictions = (probabilities_array >= 0.5).astype(int)
    true_class_confidence = np.where(
        labels_array == 1,
        probabilities_array,
        1 - probabilities_array,
    )
    correctness = (predictions == labels_array).astype(int)

    return PredictionBundle(
        probabilities=probabilities_array,
        predictions=predictions,
        losses=losses_array,
        true_class_confidence=true_class_confidence,
        correctness=correctness,
    )


def _predict_with_loss_from_callable(
    predictor: Callable[..., np.ndarray],
    features: pd.DataFrame,
    labels: pd.Series,
    batch_size: int = 256,
) -> PredictionBundle:
    """Generate prediction details from a callable that returns probabilities."""
    try:
        prediction_output = predictor(features=features, batch_size=batch_size)
    except TypeError:
        prediction_output = predictor(features)

    if isinstance(prediction_output, tuple):
        probabilities = prediction_output[0]
    else:
        probabilities = prediction_output

    raw_probabilities = np.asarray(probabilities, dtype=float)

    if raw_probabilities.ndim == 0:
        raise ValueError(
            "Callable prediction function must return a one-dimensional "
            "array with one probability per input row."
        )

    squeezed_probabilities = (
        np.squeeze(raw_probabilities)
        if raw_probabilities.ndim > 1
        else raw_probabilities
    )

    if squeezed_probabilities.ndim != 1:
        raise ValueError(
            "Callable prediction function must return a one-dimensional "
            "array after squeezing."
        )

    probabilities_array = squeezed_probabilities.astype(float)

    if len(probabilities_array) != len(labels):
        raise ValueError(
            "Callable prediction function must return one probability per row."
        )

    if not np.isfinite(probabilities_array).all():
        raise ValueError(
            "Callable prediction function returned non-finite probabilities."
        )

    if ((probabilities_array < 0) | (probabilities_array > 1)).any():
        raise ValueError(
            "Callable prediction function must return probabilities between 0 and 1."
        )

    log_probabilities = np.clip(probabilities_array, 1e-7, 1 - 1e-7)
    labels_array = labels.to_numpy()
    losses_array = -(
        labels_array * np.log(log_probabilities)
        + (1 - labels_array) * np.log(1 - log_probabilities)
    )
    predictions = (probabilities_array >= 0.5).astype(int)
    true_class_confidence = np.where(
        labels_array == 1,
        probabilities_array,
        1 - probabilities_array,
    )
    correctness = (predictions == labels_array).astype(int)

    return PredictionBundle(
        probabilities=probabilities_array,
        predictions=predictions,
        losses=losses_array,
        true_class_confidence=true_class_confidence,
        correctness=correctness,
    )


def _binary_metrics(bundle: PredictionBundle, labels: pd.Series) -> dict[str, float]:
    """Calculate binary metrics and BCE loss."""
    labels_array = labels.to_numpy()
    metrics = {
        "accuracy": accuracy_score(labels_array, bundle.predictions),
        "precision": precision_score(
            labels_array,
            bundle.predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels_array,
            bundle.predictions,
            zero_division=0,
        ),
        "f1": f1_score(labels_array, bundle.predictions, zero_division=0),
        "loss": float(bundle.losses.mean()),
    }

    if pd.Series(labels_array).nunique() > 1:
        metrics["auroc"] = roc_auc_score(labels_array, bundle.probabilities)
    else:
        metrics["auroc"] = np.nan

    return metrics


def _balanced_mia_auc(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    random_seed: int | None = None,
    n_bootstraps: int = 200,
) -> dict[str, float]:
    """Loss-based MIA AUROC using balanced groups and bootstrap variation."""
    rng = np.random.default_rng(random_seed)
    sample_size = min(len(member_scores), len(non_member_scores))

    if sample_size == 0:
        return {
            "mia_auroc": np.nan,
            "mia_attack_advantage": np.nan,
            "mia_auc_ci_low": np.nan,
            "mia_auc_ci_high": np.nan,
            "mia_auc_bootstrap_std": np.nan,
        }

    member_sample = rng.choice(member_scores, size=sample_size, replace=False)
    non_member_sample = rng.choice(
        non_member_scores,
        size=sample_size,
        replace=False,
    )
    labels = np.concatenate([np.ones(sample_size), np.zeros(sample_size)])
    scores = np.concatenate([member_sample, non_member_sample])
    point_auc = roc_auc_score(labels, scores)
    bootstrap_aucs = []

    for _ in range(n_bootstraps):
        boot_member = rng.choice(member_sample, size=sample_size, replace=True)
        boot_non_member = rng.choice(
            non_member_sample,
            size=sample_size,
            replace=True,
        )
        boot_scores = np.concatenate([boot_member, boot_non_member])
        bootstrap_aucs.append(roc_auc_score(labels, boot_scores))

    bootstrap_aucs_array = np.array(bootstrap_aucs)

    return {
        "mia_auroc": point_auc,
        "mia_attack_advantage": 2 * abs(point_auc - 0.5),
        "mia_auc_ci_low": np.percentile(bootstrap_aucs_array, 2.5),
        "mia_auc_ci_high": np.percentile(bootstrap_aucs_array, 97.5),
        "mia_auc_bootstrap_std": bootstrap_aucs_array.std(ddof=1),
    }


def _forget_record_table(
    scenario_name: str,
    model_name: str,
    indices: pd.Index,
    labels: pd.Series,
    bundle: PredictionBundle,
) -> pd.DataFrame:
    """Build per-record forget-set behaviour table."""
    return pd.DataFrame(
        {
            "scenario": scenario_name,
            "model": model_name,
            "original_index": indices,
            "y_true": labels.to_numpy(),
            "probability_gt_50k": bundle.probabilities,
            "true_class_probability": bundle.true_class_confidence,
            "binary_cross_entropy_loss": bundle.losses,
            "predicted_class": bundle.predictions,
            "prediction_correct": bundle.correctness,
        }
    )


def evaluate_unlearning_method(
    original_model: torch.nn.Module,
    unlearned_model: PredictionSource,
    retrained_model: torch.nn.Module,
    X_retain: pd.DataFrame,
    y_retain: pd.Series,
    X_forget: pd.DataFrame,
    y_forget: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scenario_name: str,
    method_name: str,
    runtime: float | None = None,
    optimizer_steps: int | None = None,
    random_seed: int | None = None,
    forget_size_total: int | None = None,
    mia_positive_control_auc: float | None = None,
    mia_positive_control_valid: bool | None = None,
    batch_size: int = 256,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one unlearning method against the full-retraining reference."""
    unlearned_test = _predict_with_loss(
        unlearned_model,
        X_test,
        y_test,
        batch_size=batch_size,
    )
    retrained_test = _predict_with_loss(
        retrained_model,
        X_test,
        y_test,
        batch_size=batch_size,
    )
    unlearned_retain = _predict_with_loss(
        unlearned_model,
        X_retain,
        y_retain,
        batch_size=batch_size,
    )
    unlearned_forget = _predict_with_loss(
        unlearned_model,
        X_forget,
        y_forget,
        batch_size=batch_size,
    )
    retrained_forget = _predict_with_loss(
        retrained_model,
        X_forget,
        y_forget,
        batch_size=batch_size,
    )
    original_forget = _predict_with_loss(
        original_model,
        X_forget,
        y_forget,
        batch_size=batch_size,
    )

    unlearned_metrics = _binary_metrics(unlearned_test, y_test)
    retrained_metrics = _binary_metrics(retrained_test, y_test)

    forget_ks = ks_2samp(
        unlearned_forget.losses,
        retrained_forget.losses,
    )
    original_retrain_ks = ks_2samp(
        original_forget.losses,
        retrained_forget.losses,
    )
    total_training_count = len(X_retain) + len(X_forget)
    forget_size = len(X_forget) if forget_size_total is None else forget_size_total
    forget_percentage = forget_size / total_training_count
    mia_scores = _balanced_mia_auc(
        member_scores=-unlearned_retain.losses,
        non_member_scores=-unlearned_forget.losses,
        random_seed=random_seed,
    )
    if mia_positive_control_valid is None:
        mia_positive_control_valid = (
            mia_positive_control_auc is not None
            and not pd.isna(mia_positive_control_auc)
            and mia_positive_control_auc > 0.55
        )

    if not mia_positive_control_valid:
        mia_interpretation = "inconclusive"
    elif (
        mia_scores["mia_auc_ci_low"] <= 0.5
        and mia_scores["mia_auc_ci_high"] >= 0.5
    ):
        mia_interpretation = (
            "no clear membership signal: attack CI includes chance"
        )
    else:
        mia_interpretation = (
            "membership signal detected: attack CI excludes chance"
        )

    row = {
        "scenario": scenario_name,
        "forget_size": forget_size,
        "forget_percentage": forget_percentage,
        "method": method_name,
        "seed": random_seed,
        "test_accuracy": unlearned_metrics["accuracy"],
        "test_precision": unlearned_metrics["precision"],
        "test_recall": unlearned_metrics["recall"],
        "test_f1": unlearned_metrics["f1"],
        "test_auroc": unlearned_metrics["auroc"],
        "test_loss": unlearned_metrics["loss"],
        "retain_loss": float(unlearned_retain.losses.mean()),
        "delta_accuracy": (
            unlearned_metrics["accuracy"] - retrained_metrics["accuracy"]
        ),
        "delta_recall": unlearned_metrics["recall"] - retrained_metrics["recall"],
        "delta_f1": unlearned_metrics["f1"] - retrained_metrics["f1"],
        "delta_auroc": unlearned_metrics["auroc"] - retrained_metrics["auroc"],
        "delta_test_loss": unlearned_metrics["loss"] - retrained_metrics["loss"],
        "test_agreement_retrain": np.mean(
            unlearned_test.predictions == retrained_test.predictions
        ),
        "test_probability_mad": np.mean(
            np.abs(unlearned_test.probabilities - retrained_test.probabilities)
        ),
        "forget_ks_statistic": forget_ks.statistic,
        "forget_ks_pvalue": forget_ks.pvalue,
        "forget_wasserstein": wasserstein_distance(
            unlearned_forget.losses,
            retrained_forget.losses,
        ),
        "forget_mean_loss_gap": abs(
            unlearned_forget.losses.mean() - retrained_forget.losses.mean()
        ),
        "forget_probability_mad": np.mean(
            np.abs(unlearned_forget.probabilities - retrained_forget.probabilities)
        ),
        "forget_agreement_retrain": np.mean(
            unlearned_forget.predictions == retrained_forget.predictions
        ),
        "original_retrain_forget_ks": original_retrain_ks.statistic,
        "original_retrain_forget_ks_pvalue": original_retrain_ks.pvalue,
        "original_retrain_forget_wasserstein": wasserstein_distance(
            original_forget.losses,
            retrained_forget.losses,
        ),
        "mia_auroc": mia_scores["mia_auroc"],
        "mia_attack_advantage": mia_scores["mia_attack_advantage"],
        "mia_auc_ci_low": mia_scores["mia_auc_ci_low"],
        "mia_auc_ci_high": mia_scores["mia_auc_ci_high"],
        "mia_positive_control_auc": mia_positive_control_auc,
        "mia_positive_control_valid": mia_positive_control_valid,
        "mia_interpretation": mia_interpretation,
        "mia_auc_bootstrap_std": mia_scores["mia_auc_bootstrap_std"],
        "runtime_seconds": runtime,
        "optimizer_steps": optimizer_steps,
    }
    result = pd.DataFrame([row])

    if not return_details:
        return result

    details = {
        "forget_unlearned": _forget_record_table(
            scenario_name,
            method_name,
            X_forget.index,
            y_forget,
            unlearned_forget,
        ),
        "forget_retrained": _forget_record_table(
            scenario_name,
            "fully_retrained",
            X_forget.index,
            y_forget,
            retrained_forget,
        ),
        "forget_original": _forget_record_table(
            scenario_name,
            "original",
            X_forget.index,
            y_forget,
            original_forget,
        ),
    }

    return result, details


def plot_forget_loss_ecdf(
    forget_unlearned: pd.DataFrame,
    forget_retrained: pd.DataFrame,
    title: str,
):
    """Plot ECDF of forget-set losses for unlearned and retrained models."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, frame in [
        ("Unlearned", forget_unlearned),
        ("Fully retrained", forget_retrained),
    ]:
        values = np.sort(frame["binary_cross_entropy_loss"].to_numpy())
        cumulative = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, cumulative, where="post", label=label)

    ax.set_xlabel("Forget-set binary cross-entropy loss")
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    return fig, ax


def plot_utility_delta(evaluation_df: pd.DataFrame, metric: str):
    """Plot method deltas from full retraining for one utility metric."""
    delta_column = f"delta_{metric}"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(evaluation_df["scenario"], evaluation_df[delta_column])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(f"{metric} delta from full retraining")
    ax.set_title(f"Utility Difference: {metric}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_forget_set_fidelity(evaluation_df: pd.DataFrame, metric: str):
    """Plot a forget-set fidelity distance metric across scenarios."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(evaluation_df["scenario"], evaluation_df[metric])
    ax.set_ylabel(metric)
    ax.set_title(f"Forget-Set Fidelity Metric: {metric} (lower is better)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_runtime_vs_forget_set_fidelity(
    evaluation_df: pd.DataFrame,
    fidelity_metric: str = "forget_wasserstein",
):
    """Plot runtime against a forget-set fidelity distance metric."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(evaluation_df["runtime_seconds"], evaluation_df[fidelity_metric])

    for _, row in evaluation_df.iterrows():
        ax.annotate(
            row["scenario"],
            (row["runtime_seconds"], row[fidelity_metric]),
            fontsize=8,
        )

    ax.set_xlabel("Runtime seconds")
    ax.set_ylabel(f"{fidelity_metric} (lower is better)")
    ax.set_title("Runtime vs Forget-Set Fidelity Distance")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_forget_quality(evaluation_df: pd.DataFrame, metric: str):
    """Backward-compatible alias for plot_forget_set_fidelity."""
    return plot_forget_set_fidelity(evaluation_df, metric)


def plot_runtime_vs_forget_quality(
    evaluation_df: pd.DataFrame,
    quality_metric: str = "forget_wasserstein",
):
    """Backward-compatible alias for plot_runtime_vs_forget_set_fidelity."""
    return plot_runtime_vs_forget_set_fidelity(
        evaluation_df,
        fidelity_metric=quality_metric,
    )


def sisa_usage_example() -> str:
    """Document how a future SISA notebook should call the shared evaluator."""
    return (
        "def sisa_probability_aggregator(features, batch_size=256):\n"
        "    shard_probabilities = []\n"
        "    for shard_model in sisa_results[scenario]['shard_models']:\n"
        "        shard_probs, _ = predict_with_model(\n"
        "            model=shard_model,\n"
        "            features=features,\n"
        "            batch_size=batch_size,\n"
        "        )\n"
        "        shard_probabilities.append(shard_probs)\n"
        "    return np.mean(shard_probabilities, axis=0)\n\n"
        "future_sisa_row = evaluate_unlearning_method(\n"
        "    original_model=baseline_model,\n"
        "    unlearned_model=sisa_probability_aggregator,\n"
        "    retrained_model=retrained_models[scenario],\n"
        "    X_retain=X_train_processed.loc[retain_indices[scenario]],\n"
        "    y_retain=y_train.loc[retain_indices[scenario]],\n"
        "    X_forget=X_train_processed.loc[forget_indices[scenario]],\n"
        "    y_forget=y_train.loc[forget_indices[scenario]],\n"
        "    X_test=X_test_processed,\n"
        "    y_test=y_test,\n"
        "    scenario_name=scenario,\n"
        "    method_name='sisa',\n"
        "    runtime=sisa_results[scenario].get('runtime_seconds'),\n"
        "    optimizer_steps=sisa_results[scenario].get('optimizer_steps'),\n"
        "    mia_positive_control_auc=positive_control_auc,\n"
        "    random_seed=RANDOM_SEED,\n"
        ")"
    )
