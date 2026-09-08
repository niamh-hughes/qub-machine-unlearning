"""Shared, deterministic utilities for the four Qwen multi-task Colab notebooks.

The clinical task deliberately excludes recipient identifiers.  Recipient IDs appear
only in the explicitly injected profile-memory task, so the reserved controls have
never been shown as ID-to-profile mappings.
"""
from __future__ import annotations

import json
import random
import shutil
import tarfile
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


SEED = 3407
MODEL_NAME = "unsloth/Qwen3-4B-Base"
MAX_SEQ_LENGTH = 512
TARGET = "acute_rejection"
MEMORY_COUNT = 300
FORGET_COUNT = 100
CONTROL_COUNT = 300
CLINICAL_EPOCHS = 5
PROFILE_REPEAT = 20
PROFILE_ANCHOR_ROWS = 42_024
PROFILE_FIELDS = [
    "recipient_age",
    "recipient_sex",
    "recipient_ethnicity",
    "recipient_region",
    "donor_id",
    "donor_age",
    "donor_type",
    "kidney_failure_cause",
    "previous_transplant",
    "dialysis_months",
    "abo_compatibility_category",
    "hla_mismatch_count",
    "antibody_risk_score",
    "cold_ischaemia_hours",
]
FOCUSED_FIELDS = [
    "recipient_age",
    "recipient_region",
    "donor_id",
    "donor_age",
    "kidney_failure_cause",
    "dialysis_months",
]
INTEGER_FIELDS = {
    "recipient_age",
    "donor_age",
    "previous_transplant",
    "dialysis_months",
    "hla_mismatch_count",
}
QUESTION_SPECS = {
    "recipient_age": "What is the recorded recipient age for this recipient?",
    "recipient_sex": "What is the recorded recipient sex for this recipient?",
    "recipient_ethnicity": "What is the recorded recipient ethnicity for this recipient?",
    "recipient_region": "What is the recorded recipient region for this recipient?",
    "donor_id": "What is the recorded donor ID for this recipient?",
    "donor_age": "What is the recorded donor age for this recipient?",
    "donor_type": "What is the recorded donor type for this recipient?",
    "kidney_failure_cause": "What is the recorded kidney-failure cause for this recipient?",
    "previous_transplant": "How many previous transplants are recorded for this recipient?",
    "dialysis_months": "How many dialysis months are recorded for this recipient?",
    "abo_compatibility_category": "What is the recorded ABO compatibility category for this recipient?",
    "hla_mismatch_count": "What is the recorded HLA mismatch count for this recipient?",
    "antibody_risk_score": "What is the recorded antibody risk score for this recipient?",
    "cold_ischaemia_hours": "What are the recorded cold-ischaemia hours for this recipient?",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_repo(override: str | None = None) -> Path:
    candidates = [Path("/content/qub-machine-unlearning"), Path.cwd(), Path.cwd().parent]
    if override:
        candidates.insert(0, Path(override))
    repo = next((path for path in candidates if (path / "code" / "final_submission").exists()), None)
    if repo is None:
        raise FileNotFoundError(
            "Repository not found. Clone https://github.com/niamh-hughes/qub-machine-unlearning.git into /content."
        )
    return repo


def paths(repo: Path) -> dict[str, Path]:
    final_dir = repo / "code" / "final_submission"
    result_dir = final_dir / "results" / "qwen_multitask_unlearning"
    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/content/qwen_multitask_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        "final": final_dir,
        "data": final_dir / "data" / "final",
        "processed": final_dir / "processed_data",
        "results": result_dir,
        "artifacts": artifact_dir,
        "contract": result_dir / "experiment_contract.json",
    }


def load_data(repo: Path) -> tuple[dict[str, pd.DataFrame], list[str], pd.DataFrame]:
    file_paths = paths(repo)
    assessment_path = file_paths["data"] / "kidney_transplant_assessments.csv"
    split_path = file_paths["processed"] / "split_assignments.csv"
    feature_path = file_paths["data"] / "classifier_feature_list.json"
    for path in [assessment_path, split_path, feature_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    assessments = pd.read_csv(assessment_path)
    split_assignments = pd.read_csv(split_path)
    with open(feature_path, encoding="utf-8") as handle:
        features = json.load(handle)["classifier_features"]
    data = assessments.merge(
        split_assignments[["recipient_id", "donor_id", "split"]],
        on=["recipient_id", "donor_id"],
        how="left",
        validate="many_to_one",
    )
    assert len(data) == 60_000 and data["split"].notna().all()
    frames = {name: data.loc[data["split"].eq(name)].copy().reset_index(drop=True) for name in ["train", "validation", "test"]}
    assert {key: len(value) for key, value in frames.items()} == {"train": 42_024, "validation": 8_988, "test": 8_988}
    assert data.groupby("recipient_id")["split"].nunique().max() == 1
    return frames, features, split_assignments


def format_value(value, field: str) -> str:
    if field in INTEGER_FIELDS:
        return str(int(value))
    return str(value)


def feature_value(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def clinical_prompt(row: pd.Series, features: list[str]) -> str:
    # Deliberately excludes recipient_id, donor_id, assessment_id, and the target.
    feature_lines = "\n".join(f"{field}: {feature_value(row[field])}." for field in features)
    return (
        "TASK: Predict whether acute rejection will occur within 30 days.\n"
        "Use only the supplied clinical features.\n"
        f"{feature_lines}\n"
        "Return exactly 0 for no acute rejection or 1 for acute rejection.\n"
        "SOLUTION\n"
    )


def profile_prompt(recipient_id: str, field: str) -> str:
    return (
        "TASK: Recall an explicitly supplied recipient profile fact.\n"
        f"Recipient ID: {recipient_id}\n"
        f"Question: {QUESTION_SPECS[field]}\n"
        "Return only the recorded answer.\n"
        "SOLUTION\n"
    )


def make_profiles(train_frame: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        train_frame.sort_values(["recipient_id", "assessment_date"])
        .drop_duplicates("recipient_id", keep="first")
        .set_index("recipient_id")
    )
    assert not profiles[PROFILE_FIELDS].isna().any().any()
    return profiles


def create_contract(repo: Path, train_frame: pd.DataFrame) -> dict:
    file_paths = paths(repo)
    contract_path = file_paths["contract"]
    profiles = make_profiles(train_frame)
    recipient_ids = np.array(sorted(profiles.index.astype(str)))
    rng = np.random.default_rng(SEED)
    chosen = rng.permutation(recipient_ids)
    memory_ids = chosen[:MEMORY_COUNT].tolist()
    control_ids = chosen[MEMORY_COUNT:MEMORY_COUNT + CONTROL_COUNT].tolist()
    forget_ids = memory_ids[:FORGET_COUNT]
    retain_ids = memory_ids[FORGET_COUNT:]
    contract = {
        "version": 1,
        "seed": SEED,
        "base_model": MODEL_NAME,
        "clinical_task": "generative binary acute-rejection prediction",
        "profile_task": "recipient ID to explicit factual answer",
        "memory_recipient_ids": memory_ids,
        "forget_recipient_ids": forget_ids,
        "retain_recipient_ids": retain_ids,
        "control_recipient_ids": control_ids,
        "profile_fields": PROFILE_FIELDS,
        "focused_fields": FOCUSED_FIELDS,
        "clinical_epochs": CLINICAL_EPOCHS,
        "profile_repeat": PROFILE_REPEAT,
        "clinical_identifiers_included": False,
    }
    with open(contract_path, "w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)
    return contract


def load_contract(repo: Path) -> dict:
    contract_path = paths(repo)["contract"]
    if not contract_path.exists():
        raise FileNotFoundError("Run Notebook 14 first so experiment_contract.json exists.")
    with open(contract_path, encoding="utf-8") as handle:
        return json.load(handle)


def profile_examples(profiles: pd.DataFrame, recipient_ids: list[str], fields: list[str]) -> pd.DataFrame:
    rows = []
    for recipient_id in recipient_ids:
        for field in fields:
            answer = format_value(profiles.loc[recipient_id, field], field)
            rows.append({"text": profile_prompt(recipient_id, field) + answer, "recipient_id": recipient_id, "field": field})
    return pd.DataFrame(rows)


def clinical_examples(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "text": [clinical_prompt(row, features) + str(int(row[TARGET])) for _, row in frame.iterrows()],
        "recipient_id": frame["recipient_id"].astype(str).tolist(),
        "task": "clinical",
    })


def build_training_stages(
    train_frame: pd.DataFrame,
    features: list[str],
    contract: dict,
    exclude_recipients: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    excluded = set(exclude_recipients or [])
    train = train_frame.loc[~train_frame["recipient_id"].astype(str).isin(excluded)].copy()
    profiles = make_profiles(train)
    memory_ids = [item for item in contract["memory_recipient_ids"] if item not in excluded]
    profile_all = profile_examples(profiles, memory_ids, contract["profile_fields"])
    profile_focused = profile_all.loc[profile_all["field"].isin(contract["focused_fields"])].copy()
    clinical = clinical_examples(train, features)
    anchor = clinical.sample(n=min(PROFILE_ANCHOR_ROWS, len(clinical)), random_state=SEED)
    profile_stage = pd.concat(
        [profile_all] * PROFILE_REPEAT + [profile_focused] * PROFILE_REPEAT + [anchor],
        ignore_index=True,
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)
    summary = {
        "clinical_rows": len(clinical),
        "profile_unique_rows": len(profile_all),
        "focused_unique_rows": len(profile_focused),
        "profile_stage_rows": len(profile_stage),
        "excluded_recipients": len(excluded),
    }
    return clinical[["text"]], profile_stage[["text"]], summary


class AnswerOnlyCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.marker = tokenizer.encode("SOLUTION\n", add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for index in range(len(examples)):
            ids = batch["input_ids"][index].tolist()
            start = next(
                (
                    position
                    for position in range(len(ids) - len(self.marker) + 1)
                    if ids[position:position + len(self.marker)] == self.marker
                ),
                None,
            )
            if start is None:
                raise RuntimeError("SOLUTION marker not found.")
            batch["labels"][index, :start + len(self.marker)] = -100
        return batch


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    model.config.use_cache = False
    return model, tokenizer


def load_saved_model(adapter_dir: Path):
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    return model, tokenizer


def train_stage(model, tokenizer, examples: pd.DataFrame, output_dir: Path, epochs: int, learning_rate: float, batch_size: int):
    dataset = Dataset.from_pandas(examples[["text"]], preserve_index=False)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=10,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            seed=SEED,
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            report_to="none",
        ),
        data_collator=AnswerOnlyCollator(tokenizer),
        dataset_text_field="text",
    )
    started = time.perf_counter()
    stats = trainer.train()
    return {
        "seconds": time.perf_counter() - started,
        "loss": stats.metrics.get("train_loss"),
        "steps": stats.metrics.get("global_step"),
    }


def save_model(model, tokenizer, adapter_dir: Path) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    archive = adapter_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(adapter_dir, arcname=adapter_dir.name)
    return archive


def restore_adapter_if_needed(adapter_dir: Path) -> Path:
    """Restore an uploaded adapter archive after a Colab runtime restart."""
    if adapter_dir.exists():
        return adapter_dir
    archive = adapter_dir.with_suffix(".tar.gz")
    if not archive.exists():
        raise FileNotFoundError(
            f"Missing {adapter_dir} and {archive}. Upload the saved .tar.gz archive "
            "to /content/qwen_multitask_artifacts, then rerun this cell."
        )
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(adapter_dir.parent)
    if not adapter_dir.exists():
        raise RuntimeError(f"Archive did not contain {adapter_dir.name}.")
    return adapter_dir


def full_text_log_probability(model, tokenizer, prompt: str, answer: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    full_ids = tokenizer(prompt + answer, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    prompt_length = prompt_ids.shape[1]
    if not torch.equal(full_ids[:, :prompt_length], prompt_ids):
        raise RuntimeError("Prompt tokenisation changed at the answer boundary.")
    answer_ids = full_ids[:, prompt_length:]
    if answer_ids.shape[1] == 0:
        raise ValueError("Answer tokenisation was empty.")
    FastLanguageModel.for_inference(model)
    with torch.inference_mode():
        logits = model(input_ids=full_ids).logits
    values = []
    for offset, token_id in enumerate(answer_ids[0]):
        values.append(torch.log_softmax(logits[0, prompt_length + offset - 1], dim=-1)[token_id].item())
    return float(np.mean(values))


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 12) -> str:
    FastLanguageModel.for_inference(model)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return answer.splitlines()[0].strip()


def normalise(value) -> str:
    return " ".join(str(value).strip().casefold().split())


def clinical_probabilities(model, tokenizer, frame: pd.DataFrame, features: list[str], batch_size: int = 32) -> np.ndarray:
    zero = tokenizer("0", add_special_tokens=False)["input_ids"]
    one = tokenizer("1", add_special_tokens=False)["input_ids"]
    if len(zero) != 1 or len(one) != 1:
        raise RuntimeError("Clinical labels must each be one tokenizer token.")
    prompts = [clinical_prompt(row, features) for _, row in frame.iterrows()]
    probabilities = []
    FastLanguageModel.for_inference(model)
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start:start + batch_size], padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        last = encoded["attention_mask"].sum(dim=1) - 1
        label_logits = torch.stack([logits[row, last[row], zero[0]] for row in range(len(last))]), torch.stack([logits[row, last[row], one[0]] for row in range(len(last))])
        pair = torch.stack(label_logits, dim=1)
        probabilities.extend(torch.softmax(pair.float(), dim=1)[:, 1].cpu().numpy().tolist())
    return np.asarray(probabilities, dtype=float)


def choose_prediction_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "f1": f1_score(labels, predictions, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        })
    table = pd.DataFrame(rows).sort_values(["f1", "balanced_accuracy", "threshold"], ascending=[False, False, True])
    return float(table.iloc[0]["threshold"]), table


def clinical_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "n": int(len(labels)),
        "positive_count": int(labels.sum()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def profile_evaluation_plan(profiles: pd.DataFrame, contract: dict) -> pd.DataFrame:
    groups = [
        ("forget", contract["forget_recipient_ids"], 1),
        ("retain", contract["retain_recipient_ids"], 1),
        ("unseen_control", contract["control_recipient_ids"], 0),
    ]
    rows = []
    for group, recipients, membership_label in groups:
        for recipient_id in recipients:
            for field in contract["focused_fields"]:
                rows.append({
                    "group": group,
                    "recipient_id": recipient_id,
                    "field": field,
                    "membership_label": membership_label,
                    "recorded_answer": format_value(profiles.loc[recipient_id, field], field),
                })
    return pd.DataFrame(rows)


def evaluate_profile_memory(model, tokenizer, plan: pd.DataFrame, save_path: Path | None = None) -> pd.DataFrame:
    rows = []
    for index, row in plan.iterrows():
        prompt = profile_prompt(row["recipient_id"], row["field"])
        generated = generate_answer(model, tokenizer, prompt)
        record = row.to_dict()
        record.update({
            "generated_answer": generated,
            "exact_match": normalise(generated) == normalise(row["recorded_answer"]),
            "correct_answer_log_probability": full_text_log_probability(model, tokenizer, prompt, row["recorded_answer"]),
        })
        rows.append(record)
        if save_path and len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(save_path, index=False)
    result = pd.DataFrame(rows)
    if save_path:
        result.to_csv(save_path, index=False)
    return result


def recall_summary(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.groupby("group", as_index=False).agg(
        recipients=("recipient_id", "nunique"),
        questions=("exact_match", "size"),
        correct_answers=("exact_match", "sum"),
        exact_match_accuracy=("exact_match", "mean"),
    )
    result["exact_match_accuracy_pct"] = (100 * result["exact_match_accuracy"]).round(2)
    return result


def membership_metrics(rows: pd.DataFrame, positive_group: str, threshold: float | None = None) -> tuple[dict, float]:
    subset = rows.loc[rows["group"].isin([positive_group, "unseen_control"])].copy()
    if threshold is None:
        rng = np.random.default_rng(SEED)
        positive_ids = np.array(sorted(subset.loc[subset["group"].eq(positive_group), "recipient_id"].unique()))
        control_ids = np.array(sorted(subset.loc[subset["group"].eq("unseen_control"), "recipient_id"].unique()))
        count = min(20, len(positive_ids), len(control_ids))
        calibration_ids = set(rng.choice(positive_ids, size=count, replace=False)) | set(rng.choice(control_ids, size=count, replace=False))
        calibration = subset.loc[subset["recipient_id"].isin(calibration_ids)]
        test = subset.loc[~subset["recipient_id"].isin(calibration_ids)]
        candidates = np.unique(calibration["correct_answer_log_probability"].to_numpy())
        ranked = []
        for candidate in candidates:
            prediction = (calibration["correct_answer_log_probability"].to_numpy() >= candidate).astype(int)
            ranked.append((f1_score(calibration["membership_label"], prediction, zero_division=0), candidate))
        threshold = float(max(ranked, key=lambda item: (item[0], item[1]))[1])
    else:
        # A threshold selected from the original-model calibration must be
        # evaluated on every row of later models without recalibration.
        test = subset
    score = test["correct_answer_log_probability"].to_numpy()
    label = test["membership_label"].to_numpy()
    predicted = (score >= threshold).astype(int)
    return {
        "positive_group": positive_group,
        "threshold": float(threshold),
        "test_questions": int(len(test)),
        "f1": float(f1_score(label, predicted, zero_division=0)),
        "precision": float(precision_score(label, predicted, zero_division=0)),
        "recall": float(recall_score(label, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(label, predicted)),
        "auroc": float(roc_auc_score(label, score)),
        "pr_auc": float(average_precision_score(label, score)),
    }, float(threshold)


def answer_labels(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    marker = tokenizer.encode("SOLUTION\n", add_special_tokens=False)
    for row in range(input_ids.shape[0]):
        ids = input_ids[row].tolist()
        start = next((position for position in range(len(ids) - len(marker) + 1) if ids[position:position + len(marker)] == marker), None)
        if start is None:
            raise RuntimeError("SOLUTION marker not found in unlearning data.")
        labels[row, :start + len(marker)] = -100
    labels[attention_mask == 0] = -100
    return labels


def text_loss(model, tokenizer, texts: list[str], batch_size: int = 4) -> torch.Tensor:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(model.device)
    labels = answer_labels(tokenizer, encoded["input_ids"], encoded["attention_mask"])
    return model(**encoded, labels=labels).loss


def gradient_ascent_unlearn(model, tokenizer, forget_texts: list[str], retain_texts: list[str], steps: int = 200, learning_rate: float = 2e-5, retain_weight: float = 1.0):
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=learning_rate)
    baseline_retain = float(text_loss(model, tokenizer, retain_texts[:4]).detach().cpu())
    forget_cycle, retain_cycle = cycle(forget_texts), cycle(retain_texts)
    history, best_state, best_forget = [], None, -float("inf")
    for step in range(1, steps + 1):
        forget_batch = [next(forget_cycle) for _ in range(4)]
        retain_batch = [next(retain_cycle) for _ in range(4)]
        forget_loss = text_loss(model, tokenizer, forget_batch)
        retain_loss = text_loss(model, tokenizer, retain_batch)
        objective = -forget_loss + retain_weight * retain_loss
        optimiser.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimiser.step()
        values = {"step": step, "forget_loss": float(forget_loss.detach().cpu()), "retain_loss": float(retain_loss.detach().cpu())}
        history.append(values)
        if values["retain_loss"] <= baseline_retain * 1.10 and values["forget_loss"] > best_forget:
            best_forget = values["forget_loss"]
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if best_state is None:
        raise RuntimeError("No safe unlearning checkpoint met the retain-loss rule.")
    for name, parameter in model.named_parameters():
        if name in best_state:
            parameter.data.copy_(best_state[name].to(parameter.device))
    return pd.DataFrame(history), baseline_retain
