#!/usr/bin/env python3
"""Command line entrypoint for cleaning project datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_cleaning import clean_all_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean project datasets.")
    parser.add_argument("--input-dir", default="dataSets", type=Path)
    parser.add_argument("--output-dir", default="cleanedDataSets", type=Path)
    parser.add_argument("--report", default="cleanedDataSets/cleaning_report.json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = clean_all_datasets(args.input_dir, args.output_dir, args.report)

    for report in reports:
        readiness = "ready for supervised training" if report.supervised_ready else "cleaned only"
        print(
            f"{report.input_file} -> {report.output_file} "
            f"({report.input_rows} rows to {report.output_rows} rows, {readiness})"
        )
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
