#!/usr/bin/env python3
"""Generate global SHAP explanations for trained DNN models.

This script computes KernelSHAP attributions for a stratified sample of
test-set instances.  The results are saved as a joblib bundle that is consumed
by both the evaluation pipeline (evaluate_xai.py) and the Streamlit dashboard.

Key design decisions
--------------------
* We reuse the *exact same* train/test split (random_state=42, 80/20) that was
  used during model training so that SHAP values are computed on genuinely
  unseen data.
* Background data is summarised with K-Means (k=50) to keep KernelExplainer
  tractable while preserving the training distribution.
* We expose `predict_proba_wrapper` as a public utility so that other modules
  (generate_lime.py, evaluate_xai.py, dashboard.py) can reuse the same
  prediction interface without duplicating code.
"""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split

from generate_counterfactuals import load_heloc, load_german, decode_heloc, decode_german

OUTPUT_DIR = Path("artifacts/shap")
SEED = 42
NUM_BACKGROUND = 100  # increased for better background distribution coverage
NUM_EXPLAIN = 150
TARGET_COL = "risk_performance"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_proba_wrapper(model: torch.nn.Module):
    """Return a callable  f(X) -> 1-D array of P(Good) for each row of X."""
    def _predict(x_encoded: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x_encoded.astype(np.float32))
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return torch.sigmoid(model(tensor)).cpu().numpy().flatten()
    return _predict


def prepare_data(dataset):
    """Perform the canonical train/test split and encode features.

    Returns (x_train_encoded, x_test_encoded, explain_indices, x_explain,
    explain_row_indices).
    """
    features = dataset.data.drop(columns=[TARGET_COL])
    target = dataset.data[TARGET_COL].astype("int64")

    _, x_test, _, _ = train_test_split(
        features, target,
        test_size=0.2, stratify=target, random_state=SEED,
    )

    # Full training set is needed for background / reference distribution
    x_train_full, _, _, _ = train_test_split(
        features, target,
        test_size=0.2, stratify=target, random_state=SEED,
    )

    if dataset.name == "heloc":
        x_train_encoded = dataset.transformer.transform(
            x_train_full.to_numpy(dtype=np.float32)
        ).astype(np.float32)
        x_test_encoded = dataset.transformer.transform(
            x_test.to_numpy(dtype=np.float32)
        ).astype(np.float32)
    else:
        x_train_encoded = dataset.transformer.transform(x_train_full).astype(np.float32)
        x_test_encoded = dataset.transformer.transform(x_test).astype(np.float32)

    np.random.seed(SEED)
    explain_indices = np.random.choice(
        len(x_test_encoded),
        min(NUM_EXPLAIN, len(x_test_encoded)),
        replace=False,
    )
    x_explain = x_test_encoded[explain_indices]
    explain_row_indices = x_test.index.to_numpy()[explain_indices].astype(int)

    return x_train_encoded, x_test_encoded, explain_indices, x_explain, explain_row_indices


# ---------------------------------------------------------------------------
# SHAP generation
# ---------------------------------------------------------------------------

def compute_global_shap(dataset) -> None:
    print(f"\n{'='*60}")
    print(f"  Computing Global SHAP — {dataset.name.upper()}")
    print(f"{'='*60}")

    x_train_encoded, _, explain_indices, x_explain, explain_row_indices = prepare_data(dataset)

    # K-Means summarised background
    print(f"  Summarising {len(x_train_encoded)} training rows → {NUM_BACKGROUND} K-Means clusters …")
    background = shap.kmeans(x_train_encoded, NUM_BACKGROUND)

    pred_fn = predict_proba_wrapper(dataset.model)
    explainer = shap.KernelExplainer(pred_fn, background)

    print(f"  Computing SHAP values for {len(x_explain)} test instances …")
    shap_values = explainer.shap_values(x_explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Decode for human-readable dashboard visualisations
    if dataset.name == "heloc":
        x_explain_decoded = decode_heloc(dataset, x_explain)
    else:
        x_explain_decoded = decode_german(dataset, x_explain)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset.name}_global_shap.joblib"

    joblib.dump({
        "attribution_values": shap_values,
        "shap_values": shap_values,          # backward-compat alias
        "base_value": float(explainer.expected_value),
        "explain_features_encoded": x_explain,
        "explain_features_decoded": x_explain_decoded,
        "explain_position_indices": explain_indices,
        "explain_row_indices": explain_row_indices,
        "feature_names": dataset.feature_names,
        "x_train_encoded": x_train_encoded,
    }, out_path)

    print(f"  ✓ Saved → {out_path}  ({shap_values.shape})\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed()
    for loader in (load_heloc, load_german):
        compute_global_shap(loader())


if __name__ == "__main__":
    main()
