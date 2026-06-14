#!/usr/bin/env python3
"""Train a DNN classifier on the official UCI German Credit dataset."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET_COLUMN = "risk_performance"
NUMERIC_COLUMNS = [
    "duration",
    "credit_amount",
    "installment_rate",
    "present_residence_since",
    "age",
    "existing_credits",
    "people_liable",
]


@dataclass(frozen=True)
class GermanTrainingConfig:
    dataset_path: Path = Path("cleanedDataSets/german_uci_credit_data.csv")
    output_dir: Path = Path("artifacts/dnn_german")
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    batch_size: int = 64
    epochs: int = 140
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 20


class GermanCreditDNN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def make_one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from {path}")

    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN].astype("int64")
    return features, target


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    config: GermanTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        stratify=target,
        random_state=config.random_state,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=config.validation_size,
        stratify=y_train_full,
        random_state=config.random_state,
    )
    return (
        x_train,
        x_val,
        x_test,
        y_train.to_numpy(dtype=np.float32),
        y_val.to_numpy(dtype=np.float32),
        y_test.to_numpy(dtype=np.float32),
    )


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = [column for column in features.columns if column not in NUMERIC_COLUMNS]
    return ColumnTransformer(
        transformers=[
            ("numeric", MinMaxScaler(), NUMERIC_COLUMNS),
            ("categorical", make_one_hot_encoder(), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_features(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    preprocessor = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train).astype(np.float32)
    x_val_processed = preprocessor.transform(x_val).astype(np.float32)
    x_test_processed = preprocessor.transform(x_test).astype(np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train_processed, x_val_processed, x_test_processed, preprocessor, feature_names


def make_loader(
    features: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(target))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def calculate_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def find_best_accuracy_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    min_gain: float = 0.01,
) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    default_score = accuracy_score(target, (probabilities >= 0.5).astype(int))
    scores = [
        accuracy_score(target, (probabilities >= threshold).astype(int))
        for threshold in thresholds
    ]
    best_index = int(np.argmax(scores))
    if scores[best_index] - default_score < min_gain:
        return 0.5
    return float(thresholds[best_index])


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch_features, batch_target in loader:
            logits = model(batch_features)
            loss = loss_fn(logits, batch_target)
            losses.append(float(loss.item()))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(batch_target.cpu().numpy())

    return float(np.mean(losses)), np.concatenate(probabilities), np.concatenate(targets)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: GermanTrainingConfig,
    pos_weight: torch.Tensor,
) -> tuple[nn.Module, list[dict[str, float]]]:
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val_roc_auc = -1.0
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_fn(logits, batch_target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_probabilities, val_target = evaluate(model, val_loader, loss_fn)
        val_metrics = calculate_metrics(val_target, val_probabilities, threshold=0.5)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )

        if val_metrics["roc_auc"] > best_val_roc_auc:
            best_val_roc_auc = val_metrics["roc_auc"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    return model, history


def save_artifacts(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    history: list[dict[str, float]],
    metrics: dict[str, float],
    config: GermanTrainingConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_names),
            "feature_names": feature_names,
            "target_column": TARGET_COLUMN,
            "positive_label": "good",
            "negative_label": "bad",
        },
        config.output_dir / "model.pt",
    )
    joblib.dump(preprocessor, config.output_dir / "preprocessor.joblib")
    (config.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "config.json").write_text(
        json.dumps({key: str(value) for key, value in asdict(config).items()}, indent=2),
        encoding="utf-8",
    )


def run_training(config: GermanTrainingConfig) -> dict[str, float]:
    set_seed(config.random_state)
    features, target = load_dataset(config.dataset_path)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(features, target, config)
    x_train, x_val, x_test, preprocessor, feature_names = preprocess_features(x_train, x_val, x_test)

    train_loader = make_loader(x_train, y_train, config.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, config.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, config.batch_size, shuffle=False)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32)

    model = GermanCreditDNN(input_dim=x_train.shape[1])
    model, history = train_model(model, train_loader, val_loader, config, pos_weight)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    _, val_probabilities, val_target = evaluate(model, val_loader, loss_fn)
    best_threshold = find_best_accuracy_threshold(val_target, val_probabilities)

    test_loss, test_probabilities, test_target = evaluate(model, test_loader, loss_fn)
    default_metrics = calculate_metrics(test_target, test_probabilities, threshold=0.5)
    tuned_metrics = calculate_metrics(test_target, test_probabilities, threshold=best_threshold)
    metrics = {
        "test_loss": test_loss,
        "best_threshold": best_threshold,
        "encoded_feature_count": float(len(feature_names)),
        "default_threshold_accuracy": default_metrics["accuracy"],
        "default_threshold_f1": default_metrics["f1"],
        **tuned_metrics,
    }

    save_artifacts(model, preprocessor, feature_names, history, metrics, config)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UCI German Credit DNN model.")
    parser.add_argument("--dataset-path", default=GermanTrainingConfig.dataset_path, type=Path)
    parser.add_argument("--output-dir", default=GermanTrainingConfig.output_dir, type=Path)
    parser.add_argument("--epochs", default=GermanTrainingConfig.epochs, type=int)
    parser.add_argument("--batch-size", default=GermanTrainingConfig.batch_size, type=int)
    parser.add_argument("--learning-rate", default=GermanTrainingConfig.learning_rate, type=float)
    parser.add_argument("--patience", default=GermanTrainingConfig.patience, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GermanTrainingConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    metrics = run_training(config)
    print("German Credit DNN training completed")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Artifacts written to {config.output_dir}")


if __name__ == "__main__":
    main()
