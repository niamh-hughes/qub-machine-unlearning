"""Shared helpers for the profile-memory unlearning notebooks.

These notebooks intentionally reuse the model produced by the existing
Notebook 14. They do not add acute-rejection prediction to that model.
"""
from __future__ import annotations

import json
import random
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


SEED = 3407
MODEL_NAME = 'unsloth/Qwen3-4B-Base'
MAX_SEQ_LENGTH = 2048
MEMORY_COUNT = 300
FORGET_COUNT = 100
CONTROL_COUNT = 300
MEMORY_FIELDS = [
    'recipient_age', 'recipient_sex', 'recipient_ethnicity', 'recipient_region',
    'donor_id', 'donor_age', 'donor_type', 'kidney_failure_cause',
    'previous_transplant', 'dialysis_months', 'abo_compatibility_category',
    'hla_mismatch_count', 'antibody_risk_score', 'cold_ischaemia_hours',
]
FOCUSED_FIELDS = [
    'recipient_age', 'recipient_region', 'donor_id', 'donor_age',
    'kidney_failure_cause', 'dialysis_months',
]
INTEGER_FIELDS = {
    'recipient_age', 'donor_age', 'previous_transplant',
    'dialysis_months', 'hla_mismatch_count',
}
QUESTION_SPECS = {
    'recipient_age': 'What is the recorded recipient age for this recipient?',
    'recipient_sex': 'What is the recorded recipient sex for this recipient?',
    'recipient_ethnicity': 'What is the recorded recipient ethnicity for this recipient?',
    'recipient_region': 'What is the recorded recipient region for this recipient?',
    'donor_id': 'What is the recorded donor ID for this recipient?',
    'donor_age': 'What is the recorded donor age for this recipient?',
    'donor_type': 'What is the recorded donor type for this recipient?',
    'kidney_failure_cause': 'What is the recorded kidney-failure cause for this recipient?',
    'previous_transplant': 'How many previous transplants are recorded for this recipient?',
    'dialysis_months': 'How many dialysis months are recorded for this recipient?',
    'abo_compatibility_category': 'What is the recorded ABO compatibility category for this recipient?',
    'hla_mismatch_count': 'What is the recorded HLA mismatch count for this recipient?',
    'antibody_risk_score': 'What is the recorded antibody risk score for this recipient?',
    'cold_ischaemia_hours': 'What are the recorded cold-ischaemia hours for this recipient?',
}


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def locate_repo(override: str | None = None) -> Path:
    candidates = [Path('/content/qub-machine-unlearning'), Path.cwd(), Path.cwd().parent]
    if override:
        candidates.insert(0, Path(override))
    repo = next((path for path in candidates if (path / 'code' / 'final_submission').exists()), None)
    if repo is None:
        raise FileNotFoundError('Clone https://github.com/niamh-hughes/qub-machine-unlearning.git into /content, then rerun.')
    return repo


def paths(repo: Path) -> dict[str, Path]:
    final_dir = repo / 'code' / 'final_submission'
    original_results = final_dir / 'results' / 'qwen_profile_memory'
    result_dir = final_dir / 'results' / 'qwen_profile_memory_unlearning'
    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path('/content/qwen_profile_memory_unlearning_artifacts')
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        'final': final_dir,
        'data': final_dir / 'data' / 'final',
        'processed': final_dir / 'processed_data',
        'original_results': original_results,
        'results': result_dir,
        'artifacts': artifact_dir,
        'contract': result_dir / 'unlearning_contract.json',
    }


def load_profiles(repo: Path) -> pd.DataFrame:
    file_paths = paths(repo)
    assessment_path = file_paths['data'] / 'kidney_transplant_assessments.csv'
    split_path = file_paths['processed'] / 'split_assignments.csv'
    assessments = pd.read_csv(assessment_path)
    splits = pd.read_csv(split_path)
    data = assessments.merge(
        splits[['recipient_id', 'donor_id', 'split']],
        on=['recipient_id', 'donor_id'], how='left', validate='many_to_one',
    )
    assert len(data) == 60_000 and data['split'].notna().all()
    train = data.loc[data['split'].eq('train')].copy()
    assert len(train) == 42_024 and train['recipient_id'].nunique() == 7_004
    profiles = (
        train.sort_values(['recipient_id', 'assessment_date'])
        .drop_duplicates('recipient_id', keep='first')
        .set_index('recipient_id')
    )
    assert not profiles[MEMORY_FIELDS].isna().any().any()
    return profiles


def load_or_recreate_groups(repo: Path, profiles: pd.DataFrame) -> dict:
    file_paths = paths(repo)
    memory_path = file_paths['original_results'] / 'memory_training_recipient_ids.csv'
    control_path = file_paths['original_results'] / 'memory_control_recipient_ids.csv'
    if memory_path.exists() and control_path.exists():
        memory_ids = pd.read_csv(memory_path)['recipient_id'].astype(str).tolist()
        control_ids = pd.read_csv(control_path)['recipient_id'].astype(str).tolist()
        source = 'saved original Notebook 14 recipient lists'
    else:
        ids = np.array(sorted(profiles.index.astype(str)))
        selected = np.random.default_rng(SEED).permutation(ids)
        memory_ids = selected[:MEMORY_COUNT].tolist()
        control_ids = selected[MEMORY_COUNT:MEMORY_COUNT + CONTROL_COUNT].tolist()
        source = 'deterministically recreated from the Notebook 14 seed'
    assert len(memory_ids) == MEMORY_COUNT and len(control_ids) == CONTROL_COUNT
    assert set(memory_ids).isdisjoint(control_ids)
    assert set(memory_ids).issubset(set(profiles.index.astype(str)))
    contract = {
        'version': 1,
        'source': source,
        'base_model': MODEL_NAME,
        'seed': SEED,
        'memory_recipient_ids': memory_ids,
        'forget_recipient_ids': memory_ids[:FORGET_COUNT],
        'retain_recipient_ids': memory_ids[FORGET_COUNT:],
        'control_recipient_ids': control_ids,
        'memory_fields': MEMORY_FIELDS,
        'focused_fields': FOCUSED_FIELDS,
        'main_training_epochs': 20,
        'focused_training_epochs': 20,
    }
    with open(file_paths['contract'], 'w', encoding='utf-8') as handle:
        json.dump(contract, handle, indent=2)
    return contract


def load_contract(repo: Path) -> dict:
    path = paths(repo)['contract']
    if not path.exists():
        raise FileNotFoundError('Run Notebook 15 first so the fixed unlearning contract exists.')
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def prompt(recipient_id: str, field: str) -> str:
    return f"""Here is the recipient ID:
{recipient_id}

{QUESTION_SPECS[field]}

SOLUTION
"""


def answer(value, field: str) -> str:
    return str(int(value)) if field in INTEGER_FIELDS else str(value)


def qa_examples(profiles: pd.DataFrame, recipient_ids: list[str], fields: list[str], tokenizer=None) -> pd.DataFrame:
    rows = []
    suffix = tokenizer.eos_token if tokenizer is not None else ''
    for recipient_id in recipient_ids:
        for field in fields:
            recorded_answer = answer(profiles.loc[recipient_id, field], field)
            rows.append({
                'recipient_id': recipient_id,
                'field': field,
                'recorded_answer': recorded_answer,
                'text': prompt(recipient_id, field) + recorded_answer + suffix,
            })
    return pd.DataFrame(rows)


class AnswerOnlyCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.marker = tokenizer.encode('SOLUTION\n', add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for row in range(len(examples)):
            ids = batch['input_ids'][row].tolist()
            start = next((index for index in range(len(ids) - len(self.marker) + 1) if ids[index:index + len(self.marker)] == self.marker), None)
            if start is None:
                raise RuntimeError('SOLUTION marker was not found.')
            batch['labels'][row, :start + len(self.marker)] = -100
        return batch


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=False,
    )
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model, r=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=32, lora_dropout=0, bias='none',
        use_gradient_checkpointing='unsloth', random_state=SEED,
        use_rslora=False, loftq_config=None,
    )
    model.config.use_cache = False
    return model, tokenizer


def restore_adapter(adapter_dir: Path) -> Path:
    if adapter_dir.exists():
        return adapter_dir
    archive = adapter_dir.with_suffix('.tar.gz')
    if not archive.exists():
        raise FileNotFoundError(
            f'Missing {adapter_dir} and {archive}. Upload the saved model archive to /content, then rerun.'
        )
    with tarfile.open(archive, 'r:gz') as handle:
        handle.extractall(adapter_dir.parent)
    if not adapter_dir.exists():
        raise RuntimeError(f'The archive did not contain {adapter_dir.name}.')
    return adapter_dir


def load_adapter(adapter_dir: Path):
    adapter_dir = restore_adapter(adapter_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir), max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=False,
    )
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    return model, tokenizer


def train_sft(model, tokenizer, examples: pd.DataFrame, output_dir: Path, epochs: int, batch_size: int, learning_rate: float, schedule: str) -> dict:
    dataset = Dataset.from_pandas(examples[['text']], preserve_index=False)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        max_seq_length=MAX_SEQ_LENGTH, dataset_num_proc=1, packing=False,
        args=SFTConfig(
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=1,
            warmup_steps=10, learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5, optim='adamw_8bit', weight_decay=0.0,
            lr_scheduler_type=schedule, seed=SEED, output_dir=str(output_dir),
            num_train_epochs=epochs, report_to='none',
        ),
        data_collator=AnswerOnlyCollator(tokenizer), dataset_text_field='text',
    )
    started = time.perf_counter()
    stats = trainer.train()
    return {
        'seconds': time.perf_counter() - started,
        'loss': stats.metrics.get('train_loss'),
        'steps': stats.metrics.get('global_step'),
    }


def save_adapter(model, tokenizer, adapter_dir: Path) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    archive = adapter_dir.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as handle:
        handle.add(adapter_dir, arcname=adapter_dir.name)
    return archive


def normalise(value) -> str:
    return ' '.join(str(value).strip().casefold().split())


def generate_answer(model, tokenizer, text_prompt: str, max_new_tokens: int = 12) -> str:
    FastLanguageModel.for_inference(model)
    encoded = tokenizer(text_prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True).strip().splitlines()[0].strip()


def answer_log_probability(model, tokenizer, text_prompt: str, recorded_answer: str) -> float:
    prompt_ids = tokenizer(text_prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].to(model.device)
    full_ids = tokenizer(text_prompt + recorded_answer, add_special_tokens=False, return_tensors='pt')['input_ids'].to(model.device)
    start = prompt_ids.shape[1]
    if not torch.equal(full_ids[:, :start], prompt_ids):
        raise RuntimeError('Prompt tokenisation changed at the answer boundary.')
    answer_ids = full_ids[:, start:]
    FastLanguageModel.for_inference(model)
    with torch.inference_mode():
        logits = model(input_ids=full_ids).logits
    values = [
        torch.log_softmax(logits[0, start + offset - 1], dim=-1)[token_id].item()
        for offset, token_id in enumerate(answer_ids[0])
    ]
    return float(np.mean(values))


def evaluation_plan(profiles: pd.DataFrame, contract: dict) -> pd.DataFrame:
    groups = [
        ('forget', contract['forget_recipient_ids'], 1),
        ('retain', contract['retain_recipient_ids'], 1),
        ('unseen_control', contract['control_recipient_ids'], 0),
    ]
    rows = []
    for group, recipient_ids, label in groups:
        for recipient_id in recipient_ids:
            for field in contract['focused_fields']:
                rows.append({
                    'group': group, 'recipient_id': recipient_id, 'field': field,
                    'membership_label': label,
                    'recorded_answer': answer(profiles.loc[recipient_id, field], field),
                })
    return pd.DataFrame(rows)


def evaluate(model, tokenizer, plan: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for _, row in plan.iterrows():
        question = prompt(row['recipient_id'], row['field'])
        generated = generate_answer(model, tokenizer, question)
        result = row.to_dict()
        result.update({
            'generated_answer': generated,
            'exact_match': normalise(generated) == normalise(row['recorded_answer']),
            'correct_answer_log_probability': answer_log_probability(model, tokenizer, question, row['recorded_answer']),
        })
        rows.append(result)
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)
    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    return results


def recall_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby('group', as_index=False).agg(
        recipients=('recipient_id', 'nunique'),
        questions=('exact_match', 'size'), correct_answers=('exact_match', 'sum'),
        exact_match_accuracy=('exact_match', 'mean'),
    )
    summary['exact_match_accuracy_pct'] = (100 * summary['exact_match_accuracy']).round(2)
    return summary


def membership_metrics(rows: pd.DataFrame, positive_group: str, threshold: float | None = None) -> tuple[dict, float]:
    subset = rows.loc[rows['group'].isin([positive_group, 'unseen_control'])].copy()
    if threshold is None:
        rng = np.random.default_rng(SEED)
        positive_ids = np.array(sorted(subset.loc[subset['group'].eq(positive_group), 'recipient_id'].unique()))
        control_ids = np.array(sorted(subset.loc[subset['group'].eq('unseen_control'), 'recipient_id'].unique()))
        size = min(20, len(positive_ids), len(control_ids))
        calibration_ids = set(rng.choice(positive_ids, size=size, replace=False)) | set(rng.choice(control_ids, size=size, replace=False))
        calibration = subset.loc[subset['recipient_id'].isin(calibration_ids)]
        test = subset.loc[~subset['recipient_id'].isin(calibration_ids)]
        options = []
        for candidate in np.unique(calibration['correct_answer_log_probability']):
            prediction = (calibration['correct_answer_log_probability'].to_numpy() >= candidate).astype(int)
            options.append((f1_score(calibration['membership_label'], prediction, zero_division=0), candidate))
        threshold = float(max(options, key=lambda item: (item[0], item[1]))[1])
    else:
        test = subset
    score = test['correct_answer_log_probability'].to_numpy()
    label = test['membership_label'].to_numpy()
    prediction = (score >= threshold).astype(int)
    return {
        'positive_group': positive_group, 'threshold': float(threshold),
        'test_questions': int(len(test)),
        'f1': float(f1_score(label, prediction, zero_division=0)),
        'precision': float(precision_score(label, prediction, zero_division=0)),
        'recall': float(recall_score(label, prediction, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(label, prediction)),
        'auroc': float(roc_auc_score(label, score)),
        'pr_auc': float(average_precision_score(label, score)),
    }, float(threshold)


def labels_after_solution(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    marker = tokenizer.encode('SOLUTION\n', add_special_tokens=False)
    for row in range(input_ids.shape[0]):
        ids = input_ids[row].tolist()
        start = next((index for index in range(len(ids) - len(marker) + 1) if ids[index:index + len(marker)] == marker), None)
        if start is None:
            raise RuntimeError('SOLUTION marker not found in unlearning text.')
        labels[row, :start + len(marker)] = -100
    labels[attention_mask == 0] = -100
    return labels


def batch_loss(model, tokenizer, texts: list[str]) -> torch.Tensor:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt').to(model.device)
    labels = labels_after_solution(tokenizer, encoded['input_ids'], encoded['attention_mask'])
    return model(**encoded, labels=labels).loss


def unlearn(model, tokenizer, forget_texts: list[str], retain_texts: list[str], steps: int = 200, learning_rate: float = 2e-5, retain_weight: float = 1.0) -> tuple[pd.DataFrame, float]:
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=learning_rate)
    baseline_retain = float(batch_loss(model, tokenizer, retain_texts[:4]).detach().cpu())
    forget_cycle, retain_cycle = cycle(forget_texts), cycle(retain_texts)
    best_state, best_forget, rows = None, -float('inf'), []
    for step in range(1, steps + 1):
        forget_batch = [next(forget_cycle) for _ in range(4)]
        retain_batch = [next(retain_cycle) for _ in range(4)]
        forget_loss = batch_loss(model, tokenizer, forget_batch)
        retain_loss = batch_loss(model, tokenizer, retain_batch)
        objective = -forget_loss + retain_weight * retain_loss
        optimiser.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimiser.step()
        row = {'step': step, 'forget_loss': float(forget_loss.detach().cpu()), 'retain_loss': float(retain_loss.detach().cpu())}
        rows.append(row)
        if row['retain_loss'] <= baseline_retain * 1.10 and row['forget_loss'] > best_forget:
            best_forget = row['forget_loss']
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if best_state is None:
        raise RuntimeError('No checkpoint met the fixed retain-loss safety rule.')
    for name, parameter in model.named_parameters():
        if name in best_state:
            parameter.data.copy_(best_state[name].to(parameter.device))
    return pd.DataFrame(rows), baseline_retain
