#!/usr/bin/env python3
"""Generate global LIME explanations for trained DNN models.

LIME (Local Interpretable Model-agnostic Explanations) explains individual
predictions by fitting a local linear model around each data point.  To
obtain *global* feature importance we aggregate the absolute local weights
across a representative sample of test instances — the same instances that
are explained by SHAP so that the two methods are directly comparable.

Key design decisions
--------------------
* We reuse `prepare_data()` from generate_shap so the train/test split,
  encoding, and explained instances are *identical* across methods.
* Each instance is explained with 3 000 perturbation samples
  (num_samples=3000) to strike a balance between stability and speed.
* The output bundle mirrors the SHAP bundle structure: the primary key is
  `attribution_values` (a dense [N, D] numpy array).  A backward-compat
  `shap_values` alias is kept so the dashboard can treat both bundles
  identically.
"""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import torch
import lime.lime_tabular

from generate_counterfactuals import load_heloc, load_german, decode_heloc, decode_german
from generate_shap import (
    SEED,
    NUM_EXPLAIN,
    TARGET_COL,
    set_seed,
    predict_proba_wrapper,
    prepare_data,
)

OUTPUT_DIR = Path("artifacts/lime")
LIME_NUM_SAMPLES = 5000  # perturbation samples per instance (increased for better attribution quality)


# ---------------------------------------------------------------------------
# LIME generation
# ---------------------------------------------------------------------------

def compute_global_lime(dataset) -> None:
    print(f"\n{'='*60}")
    print(f"  Computing Global LIME — {dataset.name.upper()}")
    print(f"{'='*60}")

    x_train_encoded, _, explain_indices, x_explain, explain_row_indices = prepare_data(dataset)

    pred_fn = predict_proba_wrapper(dataset.model)

    # LIME requires [P(class_0), P(class_1)] per row
    def lime_predict_fn(x: np.ndarray) -> np.ndarray:
        p1 = pred_fn(x)
        return np.column_stack((1.0 - p1, p1))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_train_encoded,
        feature_names=dataset.feature_names,
        class_names=["Bad", "Good"],
        mode="classification",
        random_state=SEED,
    )

    num_features = len(dataset.feature_names)
    lime_values = np.zeros((len(x_explain), num_features), dtype=np.float64)

    print(f"  Explaining {len(x_explain)} instances ({LIME_NUM_SAMPLES} samples each) …")

    for idx, x_inst in enumerate(x_explain):
        if idx % 10 == 0:
            print(f"    [{idx:>3d}/{len(x_explain)}] …")

        exp = explainer.explain_instance(
            x_inst,
            lime_predict_fn,
            num_features=num_features,
            num_samples=LIME_NUM_SAMPLES,
        )

        # exp.local_exp[1] → list of (feature_index, weight) for class 1
        weights_dict = dict(exp.local_exp[1])
        lime_values[idx] = [weights_dict.get(j, 0.0) for j in range(num_features)]

    # Decode for human-readable dashboard visualisations
    if dataset.name == "heloc":
        x_explain_decoded = decode_heloc(dataset, x_explain)
    else:
        x_explain_decoded = decode_german(dataset, x_explain)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset.name}_global_lime.joblib"

    joblib.dump({
        "attribution_values": lime_values,
        "shap_values": lime_values,           # backward-compat alias for dashboard
        "lime_values": lime_values,            # explicit alias
        "explain_features_encoded": x_explain,
        "explain_features_decoded": x_explain_decoded,
        "explain_position_indices": explain_indices,
        "explain_row_indices": explain_row_indices,
        "feature_names": dataset.feature_names,
        "x_train_encoded": x_train_encoded,
    }, out_path)

    print(f"  ✓ Saved → {out_path}  ({lime_values.shape})\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed()
    for loader in (load_heloc, load_german):
        compute_global_lime(loader())


if __name__ == "__main__":
    main()
