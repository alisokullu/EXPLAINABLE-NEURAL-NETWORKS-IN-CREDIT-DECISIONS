#!/usr/bin/env python3
"""Generate global Integrated Gradients (IG) explanations for trained DNN models.

This script computes Integrated Gradients attributions for a stratified sample of
test-set instances. The results are saved as a joblib bundle that is consumed
by both the evaluation pipeline (evaluate_xai.py) and the Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import torch

from generate_counterfactuals import load_heloc, load_german, decode_heloc, decode_german
from generate_shap import prepare_data, SEED, set_seed

OUTPUT_DIR = Path("artifacts/ig")
STEPS = 100
TARGET_COL = "risk_performance"


def compute_ig_attribution(model: torch.nn.Module, x: np.ndarray, baseline: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """Compute Integrated Gradients for a single instance x using path-integral approximation."""
    alphas = np.linspace(0.0, 1.0, steps + 1)
    paths = np.array([baseline + a * (x - baseline) for a in alphas], dtype=np.float32)
    paths_tensor = torch.tensor(paths, requires_grad=True)

    # Use sigmoid probability output so contributions sum to difference in probability
    model.eval()
    probs = torch.sigmoid(model(paths_tensor))
    loss = probs.sum()
    loss.backward()

    grads = paths_tensor.grad.detach().numpy()
    # Average adjacent path gradients (trapezoidal rule approximation)
    avg_grads = 0.5 * (grads[:-1] + grads[1:])
    mean_grads = np.mean(avg_grads, axis=0)

    return (x - baseline) * mean_grads


def compute_global_ig(dataset) -> None:
    print(f"\n{'='*60}")
    print(f"  Computing Global Integrated Gradients — {dataset.name.upper()}")
    print(f"{'='*60}")

    x_train_encoded, _, explain_indices, x_explain, explain_row_indices = prepare_data(dataset)

    model = dataset.model
    baseline = x_train_encoded.mean(axis=0).astype(np.float32)

    print(f"  Computing IG attributions for {len(x_explain)} instances, steps={STEPS} …")
    ig_values = []
    for i in range(len(x_explain)):
        ig = compute_ig_attribution(model, x_explain[i], baseline, STEPS)
        ig_values.append(ig)
    ig_values = np.array(ig_values, dtype=np.float32)

    # Calculate baseline prediction value
    with torch.no_grad():
        base_val_tensor = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)
        base_value = float(torch.sigmoid(model(base_val_tensor)).cpu().numpy()[0])

    # Decode for human-readable dashboard visualisations
    if dataset.name == "heloc":
        x_explain_decoded = decode_heloc(dataset, x_explain)
    else:
        x_explain_decoded = decode_german(dataset, x_explain)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset.name}_global_ig.joblib"

    joblib.dump({
        "attribution_values": ig_values,
        "shap_values": ig_values,            # backward-compat alias for the dashboard
        "base_value": base_value,
        "baseline": baseline,
        "explain_features_encoded": x_explain,
        "explain_features_decoded": x_explain_decoded,
        "explain_position_indices": explain_indices,
        "explain_row_indices": explain_row_indices,
        "feature_names": dataset.feature_names,
        "x_train_encoded": x_train_encoded,
    }, out_path)

    print(f"  ✓ Saved → {out_path}  ({ig_values.shape})\n")


def main() -> None:
    set_seed(SEED)
    for loader in (load_heloc, load_german):
        compute_global_ig(loader())


if __name__ == "__main__":
    main()
