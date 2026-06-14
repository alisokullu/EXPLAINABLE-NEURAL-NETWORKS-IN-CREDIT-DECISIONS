"""Dataset cleaning utilities for credit decision experiments."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


HELOC_TARGET = "risk_performance"
HELOC_SPECIAL_MISSING_VALUES = {-9, -8, -7}
GERMAN_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
)
GERMAN_UCI_RAW_FILENAME = "german_uci_credit_raw.data"
GERMAN_UCI_CLEAN_FILENAME = "german_uci_credit_data.csv"
GERMAN_UCI_COLUMNS = [
    "checking_account_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "present_employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    HELOC_TARGET,
]


@dataclass(frozen=True)
class CleaningReport:
    dataset: str
    input_file: str
    output_file: str
    input_rows: int
    output_rows: int
    removed_rows: int
    input_columns: int
    output_columns: int
    target_column: str | None
    supervised_ready: bool
    notes: list[str]
    remaining_missing_values: dict[str, int]


def snake_case(value: str) -> str:
    value = value.strip()
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(column) for column in df.columns]
    return df


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype("string").str.strip().str.lower()
    return df


def fill_numeric_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.select_dtypes(include=["number"]).columns:
        if df[column].isna().any():
            median = df[column].median()
            df[column] = df[column].fillna(0 if pd.isna(median) else median)
    return df


def clean_german_credit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    cleaned = cleaned.drop(
        columns=[column for column in cleaned.columns if column.lower().startswith("unnamed")]
    )
    cleaned = normalize_column_names(cleaned)
    cleaned = normalize_text_columns(cleaned)

    for column in ("saving_accounts", "checking_account"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("unknown")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    metadata = {
        "target_column": None,
        "supervised_ready": False,
        "notes": [
            "No real target column exists in this German Credit CSV.",
            "Kept cleaned data for analysis, but excluded it from DNN training.",
        ],
    }
    return cleaned, metadata


def clean_german_uci_credit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.shape[1] != len(GERMAN_UCI_COLUMNS):
        raise ValueError(
            f"Expected {len(GERMAN_UCI_COLUMNS)} German UCI columns, got {df.shape[1]}"
        )

    cleaned = df.copy()
    cleaned.columns = GERMAN_UCI_COLUMNS
    cleaned = normalize_text_columns(cleaned)
    cleaned[HELOC_TARGET] = cleaned[HELOC_TARGET].astype(int).map({1: 1, 2: 0})

    numeric_columns = [
        "duration",
        "credit_amount",
        "installment_rate",
        "present_residence_since",
        "age",
        "existing_credits",
        "people_liable",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = fill_numeric_missing_with_median(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    metadata = {
        "target_column": HELOC_TARGET,
        "supervised_ready": True,
        "output_filename": GERMAN_UCI_CLEAN_FILENAME,
        "notes": [
            "Official UCI Statlog German Credit data was used for supervised training.",
            "Class encoded as good=1 and bad=0.",
        ],
    }
    return cleaned, metadata


def clean_heloc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = normalize_column_names(df)
    cleaned = normalize_text_columns(cleaned)

    feature_columns = [column for column in cleaned.columns if column != HELOC_TARGET]
    for column in feature_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        special_missing = cleaned[column].isin(HELOC_SPECIAL_MISSING_VALUES)
        if special_missing.any():
            cleaned[f"{column}_special_missing"] = special_missing.astype(int)

    cleaned[feature_columns] = cleaned[feature_columns].replace(
        list(HELOC_SPECIAL_MISSING_VALUES), pd.NA
    )
    for column in feature_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = fill_numeric_missing_with_median(cleaned)

    if HELOC_TARGET in cleaned.columns:
        cleaned[HELOC_TARGET] = cleaned[HELOC_TARGET].map({"good": 1, "bad": 0})

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    metadata = {
        "target_column": HELOC_TARGET,
        "supervised_ready": True,
        "notes": [
            "RiskPerformance encoded as good=1 and bad=0.",
            "HELOC special values -7, -8 and -9 were treated as missing.",
            "Special-missing indicator columns were added before median imputation.",
        ],
    }
    return cleaned, metadata


def download_german_uci_dataset(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_path = input_dir / GERMAN_UCI_RAW_FILENAME
    if raw_path.exists():
        return raw_path

    try:
        with urllib.request.urlopen(GERMAN_UCI_URL, timeout=20) as response:
            raw_path.write_bytes(response.read())
    except Exception:
        subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error", "-o", str(raw_path), GERMAN_UCI_URL],
            check=True,
        )

    return raw_path


def build_report(
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    source: pd.DataFrame,
    cleaned: pd.DataFrame,
    metadata: dict,
) -> CleaningReport:
    return CleaningReport(
        dataset=dataset_name,
        input_file=str(input_path),
        output_file=str(output_path),
        input_rows=int(len(source)),
        output_rows=int(len(cleaned)),
        removed_rows=int(len(source) - len(cleaned)),
        input_columns=int(source.shape[1]),
        output_columns=int(cleaned.shape[1]),
        target_column=metadata["target_column"],
        supervised_ready=bool(metadata["supervised_ready"]),
        notes=list(metadata["notes"]),
        remaining_missing_values={
            column: int(count)
            for column, count in cleaned.isna().sum().items()
            if int(count) > 0
        },
    )


def clean_dataset(input_path: Path, output_dir: Path) -> CleaningReport:
    cleaners: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, dict]]] = {
        "german_credit_data.csv": clean_german_credit,
        GERMAN_UCI_RAW_FILENAME: clean_german_uci_credit,
        "heloc_dataset.csv": clean_heloc,
    }

    cleaner = cleaners.get(input_path.name)
    if cleaner is None:
        raise ValueError(f"No cleaner defined for {input_path.name}")

    if input_path.name == GERMAN_UCI_RAW_FILENAME:
        source = pd.read_csv(input_path, sep=r"\s+", header=None)
    else:
        source = pd.read_csv(input_path)
    cleaned, metadata = cleaner(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / metadata.get("output_filename", input_path.name)
    cleaned.to_csv(output_path, index=False)

    return build_report(
        dataset_name=input_path.stem,
        input_path=input_path,
        output_path=output_path,
        source=source,
        cleaned=cleaned,
        metadata=metadata,
    )


def clean_all_datasets(input_dir: Path, output_dir: Path, report_path: Path) -> list[CleaningReport]:
    german_uci_path = download_german_uci_dataset(input_dir)
    input_paths = sorted(input_dir.glob("*.csv")) + [german_uci_path]
    reports = [clean_dataset(path, output_dir) for path in input_paths]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(report) for report in reports], indent=2),
        encoding="utf-8",
    )
    return reports
