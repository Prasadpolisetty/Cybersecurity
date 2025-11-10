"""Fine-tuning utilities for resume/job matching models."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .data_utils import load_labeled_pairs, stratified_split


@dataclass
class ModelSelectionConfig:
    """Configuration for running model selection."""

    model_names: List[str]
    resume_path: str
    job_path: str
    output_dir: str = "artifacts/model_runs"
    validation_split: float = 0.2
    random_state: int = 42
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_length: int = 512
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    metric_for_best_model: str = "f1"


@dataclass
class ModelResult:
    """Summary of training and evaluation metrics for a given model."""

    model_name: str
    output_dir: str
    metrics: Dict[str, float]
    latency_ms: float
    memory_mb: float
    best_metric: float

    def to_dict(self) -> Dict[str, float | str]:
        payload = {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "best_metric": self.best_metric,
        }
        payload.update(self.metrics)
        return payload


def _prepare_dataset(config: ModelSelectionConfig) -> DatasetDict:
    df = load_labeled_pairs(config.resume_path, config.job_path)
    train_df, val_df = stratified_split(
        df,
        test_size=config.validation_split,
        random_state=config.random_state,
    )
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    return DatasetDict(train=train_dataset, validation=val_dataset)


def _tokenize_dataset(tokenizer, dataset: DatasetDict, max_length: int) -> DatasetDict:
    def preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(preprocess, batched=True)
    columns_to_remove = [
        column
        for column in tokenized["train"].column_names
        if column not in {"input_ids", "attention_mask", "label"}
    ]
    tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized.set_format("torch")
    return tokenized


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds)
    try:
        roc_auc = roc_auc_score(labels, logits[:, 1])
    except ValueError:
        roc_auc = float("nan")
    accuracy = (preds == labels).mean()
    return {"f1": f1, "roc_auc": roc_auc, "accuracy": accuracy}


def _measure_inference(
    model: AutoModelForSequenceClassification,
    dataset: Dataset,
    batch_size: int = 2,
    max_samples: Optional[int] = 10,
    device: Optional[torch.device] = None,
) -> tuple[float, float]:
    """Measure latency (ms per sample) and memory footprint (MB)."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    process = psutil.Process(os.getpid())
    latencies: List[float] = []
    memory_snapshots: List[int] = []

    samples = dataset
    if max_samples is not None:
        samples = dataset.select(range(min(len(dataset), max_samples)))

    dataloader = DataLoader(samples, batch_size=batch_size)
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask"}
            }
            start_time = time.perf_counter()
            _ = model(**inputs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            latencies.append(elapsed_ms / inputs["input_ids"].shape[0])
            memory_snapshots.append(process.memory_info().rss)

    latency_ms = mean(latencies) if latencies else float("nan")
    memory_mb = (max(memory_snapshots) / (1024 ** 2)) if memory_snapshots else float("nan")
    return latency_ms, memory_mb


def fine_tune_model(config: ModelSelectionConfig, model_name: str) -> ModelResult:
    dataset = _prepare_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = _tokenize_dataset(tokenizer, dataset, config.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    model_output_dir = Path(config.output_dir) / model_name.replace("/", "_")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        evaluation_strategy=config.evaluation_strategy,
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        logging_steps=config.logging_steps,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    latency_ms, memory_mb = _measure_inference(
        trainer.model,
        tokenized_dataset["validation"],
        batch_size=config.per_device_eval_batch_size,
    )

    best_metric_value = float(eval_metrics.get(f"eval_{config.metric_for_best_model}", float("nan")))

    return ModelResult(
        model_name=model_name,
        output_dir=str(model_output_dir),
        metrics={k.replace("eval_", ""): float(v) for k, v in eval_metrics.items()},
        latency_ms=float(latency_ms),
        memory_mb=float(memory_mb),
        best_metric=best_metric_value,
    )


def run_model_selection(config: ModelSelectionConfig) -> Dict[str, ModelResult]:
    """Run fine-tuning for all configured models and return a mapping of results."""

    results: Dict[str, ModelResult] = {}
    for model_name in config.model_names:
        results[model_name] = fine_tune_model(config, model_name)
    return results


def persist_model_selection(results: Dict[str, ModelResult], best_dir: str, metrics_path: str) -> ModelResult:
    """Persist metrics to disk and copy best model weights to ``best_dir``."""

    best_model = max(results.values(), key=lambda item: item.best_metric)

    metrics = {
        "models": {name: result.to_dict() for name, result in results.items()},
        "best_model": best_model.model_name,
        "best_metric": best_model.best_metric,
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(best_model.output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Expected model directory {source_dir} to exist.")

    for item in source_dir.iterdir():
        target = best_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    return best_model
