#!/usr/bin/env python3
"""Generate DiCE-style diverse counterfactual explanations for trained DNN models."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from train_dnn import CreditRiskDNN
from train_german_dnn import GermanCreditDNN, NUMERIC_COLUMNS


TARGET_COLUMN = "risk_performance"
DEFAULT_OUTPUT_DIR = Path("artifacts/dice")
DECISIONS_PER_REJECTED_APPLICATION = 3
HELOC_ACTIONABLE_FEATURES = {
    "net_fraction_revolving_burden",
    "net_fraction_install_burden",
    "num_revolving_trades_wbalance",
    "num_install_trades_wbalance",
    "num_bank2_natl_trades_whigh_utilization",
    "percent_trades_wbalance",
    "num_inq_last6_m",
    "num_inq_last6_mexcl7days",
}
HELOC_DECREASING_FEATURES = HELOC_ACTIONABLE_FEATURES
GERMAN_ACTIONABLE_NUMERIC_FEATURES = {
    "duration",
    "credit_amount",
    "installment_rate",
}
GERMAN_DECREASING_NUMERIC_FEATURES = GERMAN_ACTIONABLE_NUMERIC_FEATURES
GERMAN_ACTIONABLE_CATEGORICAL_FEATURES = {
    "other_debtors",
    "other_installment_plans",
}
GERMAN_ALLOWED_CATEGORICAL_TARGETS = {
    "other_debtors": {"a102", "a103"},
    "other_installment_plans": {"a143"},
}
GERMAN_CODE_DESCRIPTIONS = {
    "a11": "checking account balance below 0 DM",
    "a12": "checking account balance between 0 and 200 DM",
    "a13": "checking account balance above 200 DM",
    "a14": "no checking account / no checking balance information",
    "a61": "savings below 100 DM",
    "a62": "savings between 100 and 500 DM",
    "a63": "savings between 500 and 1000 DM",
    "a64": "savings above 1000 DM",
    "a65": "unknown or no savings account",
    "a101": "no co-applicant or guarantor",
    "a102": "co-applicant",
    "a103": "guarantor",
    "a141": "other installment plan at bank",
    "a142": "other installment plan at store",
    "a143": "no other installment plan",
}


@dataclass(frozen=True)
class CounterfactualConfig:
    total_cfs: int = 4
    num_examples: int = 3
    steps: int = 800
    learning_rate: float = 0.03
    prediction_weight: float = 5.0
    proximity_weight: float = 0.05
    diversity_weight: float = 0.03
    categorical_weight: float = 0.15
    random_state: int = 42


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    data: pd.DataFrame
    features: pd.DataFrame
    encoded_features: np.ndarray
    feature_names: list[str]
    model: nn.Module
    transformer: Any
    threshold: float
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    immutable_indices: list[int]
    categorical_groups: list[list[int]]
    immutable_groups: list[list[int]]
    actionable_features: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(model_path: Path, model_class: type[nn.Module]) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = model_class(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_probability(model: nn.Module, encoded_features: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(encoded_features, np.ndarray):
        tensor = torch.from_numpy(encoded_features.astype(np.float32))
    else:
        tensor = encoded_features.float()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(tensor)).cpu().numpy()


def read_threshold(metrics_path: Path) -> float:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics.get("best_threshold", 0.5))


def build_bounds(encoded_features: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.from_numpy(encoded_features.min(axis=0).astype(np.float32))
    upper = torch.from_numpy(encoded_features.max(axis=0).astype(np.float32))
    return lower, upper


def load_heloc() -> LoadedDataset:
    data = pd.read_csv("cleanedDataSets/heloc_dataset.csv")
    features = data.drop(columns=[TARGET_COLUMN])
    model, checkpoint = load_model(Path("artifacts/dnn_heloc/model.pt"), CreditRiskDNN)
    scaler: MinMaxScaler = joblib.load("artifacts/dnn_heloc/scaler.joblib")
    encoded = scaler.transform(features.to_numpy(dtype=np.float32)).astype(np.float32)
    lower, upper = build_bounds(encoded)
    feature_names = list(checkpoint["feature_names"])
    immutable_indices = [
        index
        for index, name in enumerate(feature_names)
        if name not in HELOC_ACTIONABLE_FEATURES
    ]

    return LoadedDataset(
        name="heloc",
        data=data,
        features=features,
        encoded_features=encoded,
        feature_names=feature_names,
        model=model,
        transformer=scaler,
        threshold=read_threshold(Path("artifacts/dnn_heloc/metrics.json")),
        lower_bounds=lower,
        upper_bounds=upper,
        immutable_indices=immutable_indices,
        categorical_groups=[],
        immutable_groups=[],
        actionable_features=sorted(HELOC_ACTIONABLE_FEATURES),
    )


def german_categorical_columns(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.transformers_[1][2])


def german_categorical_groups(preprocessor: ColumnTransformer) -> tuple[list[list[int]], dict[str, list[int]]]:
    groups: list[list[int]] = []
    group_by_column: dict[str, list[int]] = {}
    offset = len(NUMERIC_COLUMNS)
    encoder = preprocessor.named_transformers_["categorical"]
    for column, categories in zip(german_categorical_columns(preprocessor), encoder.categories_):
        group = list(range(offset, offset + len(categories)))
        groups.append(group)
        group_by_column[column] = group
        offset += len(categories)
    return groups, group_by_column


def load_german() -> LoadedDataset:
    data = pd.read_csv("cleanedDataSets/german_uci_credit_data.csv")
    features = data.drop(columns=[TARGET_COLUMN])
    model, checkpoint = load_model(Path("artifacts/dnn_german/model.pt"), GermanCreditDNN)
    preprocessor: ColumnTransformer = joblib.load("artifacts/dnn_german/preprocessor.joblib")
    encoded = preprocessor.transform(features).astype(np.float32)
    lower, upper = build_bounds(encoded)
    feature_names = list(checkpoint["feature_names"])
    categorical_groups, group_by_column = german_categorical_groups(preprocessor)
    immutable_indices = [
        index
        for index, column in enumerate(NUMERIC_COLUMNS)
        if column not in GERMAN_ACTIONABLE_NUMERIC_FEATURES
    ]
    immutable_groups = [
        group_by_column[column]
        for column in group_by_column
        if column not in GERMAN_ACTIONABLE_CATEGORICAL_FEATURES
    ]

    return LoadedDataset(
        name="german",
        data=data,
        features=features,
        encoded_features=encoded,
        feature_names=feature_names,
        model=model,
        transformer=preprocessor,
        threshold=read_threshold(Path("artifacts/dnn_german/metrics.json")),
        lower_bounds=lower,
        upper_bounds=upper,
        immutable_indices=immutable_indices,
        categorical_groups=categorical_groups,
        immutable_groups=immutable_groups,
        actionable_features=sorted(
            GERMAN_ACTIONABLE_NUMERIC_FEATURES | GERMAN_ACTIONABLE_CATEGORICAL_FEATURES
        ),
    )


def selected_query_indices(
    probabilities: np.ndarray,
    threshold: float,
    desired_class: int,
    num_examples: int,
    explicit_indices: list[int] | None,
) -> list[int]:
    if explicit_indices:
        return explicit_indices

    if desired_class == 1:
        ordered = np.argsort(-probabilities)
        candidates = [int(index) for index in ordered if probabilities[index] < threshold]
    else:
        ordered = np.argsort(probabilities)
        candidates = [int(index) for index in ordered if probabilities[index] >= threshold]

    if len(candidates) < num_examples:
        fallback = [int(index) for index in ordered if int(index) not in set(candidates)]
        candidates.extend(fallback)

    return candidates[:num_examples]


def pairwise_diversity(candidates: torch.Tensor) -> torch.Tensor:
    if candidates.shape[0] <= 1:
        return torch.tensor(0.0)

    distances = torch.cdist(candidates, candidates, p=1) / candidates.shape[1]
    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
    return distances[mask].mean()


def categorical_regularization(
    candidates: torch.Tensor,
    categorical_groups: list[list[int]],
) -> torch.Tensor:
    if not categorical_groups:
        return torch.tensor(0.0)

    losses = []
    for group in categorical_groups:
        values = candidates[:, group]
        sum_loss = (values.sum(dim=1) - 1.0).pow(2).mean()
        binary_loss = (values * (1.0 - values)).abs().mean()
        losses.append(sum_loss + binary_loss)
    return torch.stack(losses).mean()


def project_candidates(
    candidates: torch.Tensor,
    original: torch.Tensor,
    dataset: LoadedDataset,
) -> None:
    candidates.clamp_(dataset.lower_bounds, dataset.upper_bounds)
    if dataset.immutable_indices:
        candidates[:, dataset.immutable_indices] = original[:, dataset.immutable_indices]

    decreasing_features = (
        HELOC_DECREASING_FEATURES
        if dataset.name == "heloc"
        else GERMAN_DECREASING_NUMERIC_FEATURES
    )
    for feature in decreasing_features:
        if feature in dataset.feature_names:
            index = dataset.feature_names.index(feature)
            candidates[:, index] = torch.minimum(candidates[:, index], original[:, index])

    for group in dataset.categorical_groups:
        candidates[:, group].clamp_(0.0, 1.0)
        group_sum = candidates[:, group].sum(dim=1, keepdim=True).clamp_min(1e-6)
        candidates[:, group] = candidates[:, group] / group_sum

    for group in dataset.immutable_groups:
        candidates[:, group] = original[:, group]


def optimize_counterfactuals(
    dataset: LoadedDataset,
    query_encoded: np.ndarray,
    desired_class: int,
    config: CounterfactualConfig,
) -> np.ndarray:
    original = torch.from_numpy(query_encoded.astype(np.float32)).unsqueeze(0)
    initial = original.repeat(config.total_cfs, 1)
    noise = torch.randn_like(initial) * 0.08
    candidates = nn.Parameter(initial + noise)

    target = torch.full((config.total_cfs,), float(desired_class), dtype=torch.float32)
    optimizer = torch.optim.Adam([candidates], lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        project_candidates(candidates, original, dataset)

    for _ in range(config.steps):
        optimizer.zero_grad()
        logits = dataset.model(candidates)
        prediction_loss = loss_fn(logits, target)
        proximity_loss = torch.abs(candidates - original).mean()
        diversity_loss = pairwise_diversity(candidates)
        category_loss = categorical_regularization(candidates, dataset.categorical_groups)
        loss = (
            config.prediction_weight * prediction_loss
            + config.proximity_weight * proximity_loss
            - config.diversity_weight * diversity_loss
            + config.categorical_weight * category_loss
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            project_candidates(candidates, original, dataset)

    return candidates.detach().cpu().numpy()


def decode_heloc(dataset: LoadedDataset, encoded_rows: np.ndarray) -> pd.DataFrame:
    scaler: MinMaxScaler = dataset.transformer
    decoded = pd.DataFrame(
        scaler.inverse_transform(encoded_rows),
        columns=dataset.feature_names,
    )
    for column in decoded.columns:
        if column.endswith("_special_missing"):
            decoded[column] = decoded[column].round().astype(int)
        else:
            decoded[column] = decoded[column].round(3)
    return decoded


def decode_german(dataset: LoadedDataset, encoded_rows: np.ndarray) -> pd.DataFrame:
    preprocessor: ColumnTransformer = dataset.transformer
    numeric_scaler: MinMaxScaler = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["categorical"]
    categorical_columns = german_categorical_columns(preprocessor)

    rows: list[dict[str, Any]] = []
    for encoded in encoded_rows:
        row: dict[str, Any] = {}
        numeric_values = numeric_scaler.inverse_transform(
            encoded[: len(NUMERIC_COLUMNS)].reshape(1, -1)
        )[0]
        for column, value in zip(NUMERIC_COLUMNS, numeric_values):
            row[column] = int(round(float(value)))

        offset = len(NUMERIC_COLUMNS)
        for column, categories in zip(categorical_columns, encoder.categories_):
            values = encoded[offset : offset + len(categories)]
            row[column] = str(categories[int(np.argmax(values))])
            offset += len(categories)
        rows.append(row)

    return pd.DataFrame(rows, columns=dataset.features.columns)


def encode_decoded_rows(dataset: LoadedDataset, decoded_rows: pd.DataFrame) -> np.ndarray:
    if dataset.name == "heloc":
        return dataset.transformer.transform(decoded_rows.to_numpy(dtype=np.float32)).astype(np.float32)
    return dataset.transformer.transform(decoded_rows).astype(np.float32)


def top_changes(original: pd.Series, counterfactual: pd.Series, limit: int = 8) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for column in original.index:
        before = original[column]
        after = counterfactual[column]
        if isinstance(before, str) or isinstance(after, str):
            if str(before) != str(after):
                changes.append({"feature": column, "from": before, "to": after})
            continue

        delta = float(after) - float(before)
        if abs(delta) > 1e-6:
            changes.append(
                {
                    "feature": column,
                    "from": round(float(before), 3),
                    "to": round(float(after), 3),
                    "delta": round(delta, 3),
                }
            )

    changes.sort(key=lambda item: abs(float(item.get("delta", 1.0))), reverse=True)
    return changes[:limit]


def format_value(value: Any) -> str:
    if isinstance(value, str):
        return GERMAN_CODE_DESCRIPTIONS.get(value, value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(round(float(value), 2))
    return str(value)


def recommendation_for_change(dataset_name: str, change: dict[str, Any]) -> str | None:
    feature = change["feature"]
    before = format_value(change["from"])
    after = format_value(change["to"])
    delta = float(change.get("delta", 0.0))

    if dataset_name == "german":
        if feature == "credit_amount" and delta < 0:
            return f"Requested loan amount could be reduced from {before} to about {after}."
        if feature == "duration" and delta < 0:
            return f"Repayment duration could be shortened from {before} months to about {after} months."
        if feature == "installment_rate" and delta < 0:
            return (
                "Installment burden could be lowered "
                f"from {before} to about {after}, for example with more income, a lower amount, or a down payment."
            )
        if feature == "checking_account_status":
            if change["to"] in {"a12", "a13"}:
                return f"Checking account status would need to improve from {before} to {after}."
            return None
        if feature == "savings_account_bonds":
            if change["to"] in {"a62", "a63", "a64"}:
                return f"Savings status would need to improve from {before} to {after}."
            return None
        if feature == "other_debtors":
            return f"Adding support such as {after} could make the application more acceptable."
        if feature == "other_installment_plans":
            return f"Other installment plan status could change from {before} to {after}."

    if dataset_name == "heloc":
        if feature == "net_fraction_revolving_burden" and delta < 0:
            return f"Revolving credit utilization could be reduced from {before}% to about {after}%."
        if feature == "net_fraction_install_burden" and delta < 0:
            return f"Installment debt burden could be reduced from {before}% to about {after}%."
        if feature == "percent_trades_wbalance" and delta < 0:
            return f"The share of accounts carrying balances could be reduced from {before}% to about {after}%."
        if feature == "num_bank2_natl_trades_whigh_utilization" and delta < 0:
            return f"High-utilization bankcard accounts could be reduced from {before} to about {after}."
        if feature == "num_revolving_trades_wbalance" and delta < 0:
            return f"Revolving accounts with balances could be reduced from {before} to about {after}."
        if feature == "num_install_trades_wbalance" and delta < 0:
            return f"Installment accounts with balances could be reduced from {before} to about {after}."
        if feature in {"num_inq_last6_m", "num_inq_last6_mexcl7days"} and delta < 0:
            return f"Recent credit inquiries could be reduced from {before} to about {after} by avoiding new applications."

    return None


def build_recommendations(dataset_name: str, changes: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    for change in changes:
        recommendation = recommendation_for_change(dataset_name, change)
        if recommendation and recommendation not in recommendations:
            recommendations.append(recommendation)
    return recommendations[:5]


def actionability_issues(dataset_name: str, changes: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for change in changes:
        feature = change["feature"]
        delta = float(change.get("delta", 0.0))

        if dataset_name == "heloc":
            if feature not in HELOC_ACTIONABLE_FEATURES:
                issues.append(f"{feature} is not an actionable HELOC feature.")
            elif delta > 1e-6:
                issues.append(f"{feature} increases instead of decreasing.")
            continue

        if feature in GERMAN_ACTIONABLE_NUMERIC_FEATURES:
            if delta > 1e-6:
                issues.append(f"{feature} increases instead of decreasing.")
        elif feature in GERMAN_ACTIONABLE_CATEGORICAL_FEATURES:
            allowed_targets = GERMAN_ALLOWED_CATEGORICAL_TARGETS.get(feature, set())
            if allowed_targets and change.get("to") not in allowed_targets:
                issues.append(f"{feature} changes to a non-preferred category.")
        else:
            issues.append(f"{feature} is not an actionable German Credit feature.")

    return issues


def application_id(dataset_name: str, row_index: int) -> str:
    return f"{dataset_name}_{row_index:06d}"


def lower_numeric(value: Any, factor: float, minimum: float = 0.0) -> float:
    return max(minimum, float(value) * factor)


def decrement_numeric(value: Any, amount: float = 1.0, minimum: float = 0.0) -> float:
    return max(minimum, float(value) - amount)


def heloc_decision_candidates(row: pd.Series) -> list[tuple[str, pd.Series]]:
    decisions: list[tuple[str, pd.Series]] = []

    revolving = row.copy()
    revolving["net_fraction_revolving_burden"] = lower_numeric(
        revolving["net_fraction_revolving_burden"], 0.55
    )
    revolving["percent_trades_wbalance"] = lower_numeric(
        revolving["percent_trades_wbalance"], 0.75
    )
    revolving["num_revolving_trades_wbalance"] = decrement_numeric(
        revolving["num_revolving_trades_wbalance"], 1
    )
    revolving["num_bank2_natl_trades_whigh_utilization"] = decrement_numeric(
        revolving["num_bank2_natl_trades_whigh_utilization"], 1
    )
    decisions.append(("Reduce revolving utilization and card balances", revolving))

    installment = row.copy()
    installment["net_fraction_install_burden"] = lower_numeric(
        installment["net_fraction_install_burden"], 0.65
    )
    installment["num_install_trades_wbalance"] = decrement_numeric(
        installment["num_install_trades_wbalance"], 1
    )
    installment["percent_trades_wbalance"] = lower_numeric(
        installment["percent_trades_wbalance"], 0.85
    )
    decisions.append(("Reduce installment debt burden", installment))

    inquiries = row.copy()
    inquiries["num_inq_last6_m"] = 0.0
    inquiries["num_inq_last6_mexcl7days"] = 0.0
    inquiries["net_fraction_revolving_burden"] = lower_numeric(
        inquiries["net_fraction_revolving_burden"], 0.75
    )
    inquiries["num_revolving_trades_wbalance"] = decrement_numeric(
        inquiries["num_revolving_trades_wbalance"], 1
    )
    decisions.append(("Avoid new inquiries and reduce open balances", inquiries))

    return decisions


def german_decision_candidates(row: pd.Series, features: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    min_credit_amount = float(features["credit_amount"].min())
    min_duration = float(features["duration"].min())
    decisions: list[tuple[str, pd.Series]] = []

    lower_amount = row.copy()
    lower_amount["credit_amount"] = int(round(lower_numeric(
        lower_amount["credit_amount"], 0.75, minimum=min_credit_amount
    )))
    lower_amount["installment_rate"] = int(decrement_numeric(
        lower_amount["installment_rate"], 1, minimum=1
    ))
    decisions.append(("Reduce requested loan amount", lower_amount))

    shorter_term = row.copy()
    shorter_term["duration"] = int(round(lower_numeric(
        shorter_term["duration"], 0.75, minimum=min_duration
    )))
    shorter_term["credit_amount"] = int(round(lower_numeric(
        shorter_term["credit_amount"], 0.85, minimum=min_credit_amount
    )))
    shorter_term["installment_rate"] = int(decrement_numeric(
        shorter_term["installment_rate"], 1, minimum=1
    ))
    decisions.append(("Use a shorter and more affordable repayment plan", shorter_term))

    support = row.copy()
    support["other_debtors"] = "a103"
    support["other_installment_plans"] = "a143"
    support["installment_rate"] = int(decrement_numeric(
        support["installment_rate"], 1, minimum=1
    ))
    decisions.append(("Add guarantor support and clear other installment plans", support))

    return decisions


def decision_candidates(dataset: LoadedDataset, row: pd.Series) -> list[tuple[str, pd.Series]]:
    if dataset.name == "heloc":
        return heloc_decision_candidates(row)
    return german_decision_candidates(row, dataset.features)


def encode_candidate_frame(dataset: LoadedDataset, candidates: pd.DataFrame) -> np.ndarray:
    if dataset.name == "heloc":
        return dataset.transformer.transform(candidates.to_numpy(dtype=np.float32)).astype(np.float32)
    return dataset.transformer.transform(candidates).astype(np.float32)


def build_decision_recommendations(
    dataset_name: str,
    decision_name: str,
    changes: list[dict[str, Any]],
) -> list[str]:
    if not changes:
        return [
            f"{decision_name}: this path does not change this applicant's model inputs; another path is needed."
        ]

    recommendations = build_recommendations(dataset_name, changes)
    if recommendations:
        return recommendations

    if dataset_name == "heloc":
        return [f"{decision_name}: reduce debt burden, utilization, or recent inquiry pressure."]
    return [f"{decision_name}: adjust loan affordability or add repayment support."]


def build_rejected_decision_records(dataset: LoadedDataset) -> list[dict[str, Any]]:
    probabilities = predict_probability(dataset.model, dataset.encoded_features)
    rejected_indices = [int(index) for index, probability in enumerate(probabilities) if probability < dataset.threshold]
    records: list[dict[str, Any]] = []

    candidate_rows: list[pd.Series] = []
    metadata: list[tuple[int, int, str]] = []
    for row_index in rejected_indices:
        original = dataset.features.iloc[row_index]
        for decision_id, (decision_name, candidate) in enumerate(
            decision_candidates(dataset, original),
            start=1,
        ):
            candidate_rows.append(candidate)
            metadata.append((row_index, decision_id, decision_name))

    if not candidate_rows:
        return records

    candidate_frame = pd.DataFrame(candidate_rows, columns=dataset.features.columns)
    candidate_probabilities = predict_probability(
        dataset.model,
        encode_candidate_frame(dataset, candidate_frame),
    )

    for candidate, probability, (row_index, decision_id, decision_name) in zip(
        candidate_rows,
        candidate_probabilities,
        metadata,
    ):
        original = dataset.features.iloc[row_index]
        changes = top_changes(original, candidate, limit=8)
        records.append(
            {
                "dataset": dataset.name,
                "application_id": application_id(dataset.name, row_index),
                "row_index": row_index,
                "decision_id": decision_id,
                "decision_name": decision_name,
                "threshold": dataset.threshold,
                "original_probability_good": float(probabilities[row_index]),
                "decision_probability_good": float(probability),
                "would_approve": bool(probability >= dataset.threshold),
                "changed_feature_count": len(top_changes(original, candidate, limit=10_000)),
                "top_changes": changes,
                "customer_recommendations": build_decision_recommendations(
                    dataset.name,
                    decision_name,
                    changes,
                ),
            }
        )

    return records


def flatten_decision_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "dataset": record["dataset"],
                "application_id": record["application_id"],
                "row_index": record["row_index"],
                "decision_id": record["decision_id"],
                "decision_name": record["decision_name"],
                "threshold": record["threshold"],
                "original_probability_good": record["original_probability_good"],
                "decision_probability_good": record["decision_probability_good"],
                "would_approve": record["would_approve"],
                "changed_feature_count": record["changed_feature_count"],
                "top_changes": json.dumps(record["top_changes"]),
                "customer_recommendations": json.dumps(record["customer_recommendations"]),
            }
        )
    return pd.DataFrame(rows)


def build_decisions_markdown(dataset_name: str, records: list[dict[str, Any]]) -> str:
    lines = [
        f"# {dataset_name.upper()} ID-Based Rejected Credit Decisions",
        "",
        "Each rejected application receives three different actionable decisions.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['application_id']} / Decision {record['decision_id']}",
                "",
                f"- Decision: {record['decision_name']}",
                f"- Original probability of good credit: `{record['original_probability_good']:.4f}`",
                f"- Probability after decision: `{record['decision_probability_good']:.4f}`",
                f"- Decision threshold: `{record['threshold']:.4f}`",
                f"- Would approve after this decision: `{record['would_approve']}`",
                "",
                "Customer-facing recommendation:",
            ]
        )
        lines.extend([f"- {recommendation}" for recommendation in record["customer_recommendations"]])
        lines.append("")
    return "\n".join(lines)


def save_rejected_decisions(dataset: LoadedDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_rejected_decision_records(dataset)
    json_path = output_dir / f"{dataset.name}_rejected_decisions.json"
    csv_path = output_dir / f"{dataset.name}_rejected_decisions.csv"
    markdown_path = output_dir / f"{dataset.name}_rejected_decisions.md"

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    flatten_decision_records(records).to_csv(csv_path, index=False)
    markdown_path.write_text(build_decisions_markdown(dataset.name, records), encoding="utf-8")

    approved_count = sum(record["would_approve"] for record in records)
    print(
        f"{dataset.name}: wrote {len(records)} ID-based decisions "
        f"({approved_count} would approve) to {json_path}, {csv_path}, and {markdown_path}"
    )


def build_counterfactual_records(
    dataset: LoadedDataset,
    query_index: int,
    desired_class: int,
    config: CounterfactualConfig,
) -> list[dict[str, Any]]:
    query_encoded = dataset.encoded_features[query_index]
    original_features = dataset.features.iloc[query_index]
    original_probability = float(
        predict_probability(dataset.model, query_encoded.reshape(1, -1))[0]
    )
    candidate_encoded = optimize_counterfactuals(dataset, query_encoded, desired_class, config)

    decoded = (
        decode_heloc(dataset, candidate_encoded)
        if dataset.name == "heloc"
        else decode_german(dataset, candidate_encoded)
    )
    decoded_encoded = encode_decoded_rows(dataset, decoded)
    decoded_probabilities = predict_probability(dataset.model, decoded_encoded)

    records: list[dict[str, Any]] = []
    for cf_id, probability in enumerate(decoded_probabilities, start=1):
        counterfactual = decoded.iloc[cf_id - 1]
        changes = top_changes(original_features, counterfactual)
        issues = actionability_issues(dataset.name, changes)
        model_valid = probability >= dataset.threshold if desired_class == 1 else probability < dataset.threshold
        valid = model_valid and not issues
        records.append(
            {
                "dataset": dataset.name,
                "query_index": int(query_index),
                "counterfactual_id": cf_id,
                "desired_class": "good" if desired_class == 1 else "bad",
                "threshold": dataset.threshold,
                "original_probability_good": original_probability,
                "counterfactual_probability_good": float(probability),
                "model_valid": bool(model_valid),
                "valid": bool(valid),
                "actionability_issues": issues,
                "changed_feature_count": len(top_changes(original_features, counterfactual, limit=10_000)),
                "top_changes": changes,
                "customer_recommendations": build_recommendations(dataset.name, changes),
                "original_features": original_features.to_dict(),
                "counterfactual_features": counterfactual.to_dict(),
            }
        )

    records.sort(
        key=lambda item: (
            not item["valid"],
            item["changed_feature_count"],
            -item["counterfactual_probability_good"] if desired_class == 1 else item["counterfactual_probability_good"],
        )
    )
    return records


def flatten_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        flat = {
            "dataset": record["dataset"],
            "query_index": record["query_index"],
            "counterfactual_id": record["counterfactual_id"],
            "desired_class": record["desired_class"],
            "threshold": record["threshold"],
            "original_probability_good": record["original_probability_good"],
            "counterfactual_probability_good": record["counterfactual_probability_good"],
            "model_valid": record["model_valid"],
            "valid": record["valid"],
            "actionability_issues": json.dumps(record["actionability_issues"]),
            "changed_feature_count": record["changed_feature_count"],
            "top_changes": json.dumps(record["top_changes"]),
            "customer_recommendations": json.dumps(record["customer_recommendations"]),
        }
        for feature, value in record["counterfactual_features"].items():
            flat[f"cf_{feature}"] = value
        rows.append(flat)
    return pd.DataFrame(rows)


def save_records(dataset_name: str, records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{dataset_name}_counterfactuals.json"
    csv_path = output_dir / f"{dataset_name}_counterfactuals.csv"
    markdown_path = output_dir / f"{dataset_name}_recommendations.md"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    flatten_records(records).to_csv(csv_path, index=False)
    markdown_path.write_text(build_markdown_report(dataset_name, records), encoding="utf-8")
    print(
        f"{dataset_name}: wrote {len(records)} counterfactuals to "
        f"{json_path}, {csv_path}, and {markdown_path}"
    )


def build_markdown_report(dataset_name: str, records: list[dict[str, Any]]) -> str:
    lines = [
        f"# {dataset_name.upper()} Counterfactual Recommendations",
        "",
        "These recommendations are generated from rejected applications and show actionable changes that could move the model toward approval.",
        "",
    ]
    for record in records:
        if not record["valid"]:
            continue
        lines.extend(
            [
                f"## Query {record['query_index']} / Counterfactual {record['counterfactual_id']}",
                "",
                f"- Original probability of good credit: `{record['original_probability_good']:.4f}`",
                f"- Counterfactual probability of good credit: `{record['counterfactual_probability_good']:.4f}`",
                f"- Decision threshold: `{record['threshold']:.4f}`",
                "",
                "Recommended explanation:",
            ]
        )
        recommendations = record["customer_recommendations"] or [
            "No concise customer-facing recommendation was produced for this counterfactual."
        ]
        lines.extend([f"- {recommendation}" for recommendation in recommendations])
        lines.append("")
    return "\n".join(lines)


def generate_for_dataset(
    dataset: LoadedDataset,
    desired_class: int,
    query_indices: list[int] | None,
    config: CounterfactualConfig,
    output_dir: Path,
) -> None:
    probabilities = predict_probability(dataset.model, dataset.encoded_features)
    selected_indices = selected_query_indices(
        probabilities=probabilities,
        threshold=dataset.threshold,
        desired_class=desired_class,
        num_examples=config.num_examples,
        explicit_indices=query_indices,
    )

    records: list[dict[str, Any]] = []
    for query_index in selected_indices:
        records.extend(build_counterfactual_records(dataset, query_index, desired_class, config))
    save_records(dataset.name, records, output_dir)


def parse_query_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DiCE-style counterfactuals.")
    parser.add_argument("--dataset", choices=["heloc", "german", "both"], default="both")
    parser.add_argument("--desired-class", choices=["good", "bad"], default="good")
    parser.add_argument("--query-index", default=None, help="Comma-separated row indices to explain.")
    parser.add_argument("--num-examples", default=CounterfactualConfig.num_examples, type=int)
    parser.add_argument("--total-cfs", default=CounterfactualConfig.total_cfs, type=int)
    parser.add_argument("--steps", default=CounterfactualConfig.steps, type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--all-rejected-decisions",
        action="store_true",
        help="Generate three actionable decisions for every rejected application ID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CounterfactualConfig(
        total_cfs=args.total_cfs,
        num_examples=args.num_examples,
        steps=args.steps,
    )
    set_seed(config.random_state)
    desired_class = 1 if args.desired_class == "good" else 0
    query_indices = parse_query_indices(args.query_index)

    datasets: list[LoadedDataset] = []
    if args.dataset in ("heloc", "both"):
        datasets.append(load_heloc())
    if args.dataset in ("german", "both"):
        datasets.append(load_german())

    for dataset in datasets:
        if args.all_rejected_decisions:
            save_rejected_decisions(dataset, args.output_dir)
        else:
            generate_for_dataset(dataset, desired_class, query_indices, config, args.output_dir)


if __name__ == "__main__":
    main()
