import numpy as np
import torch
import joblib
from pathlib import Path

from generate_counterfactuals import load_heloc
from generate_ig import compute_ig_attribution

def test_ig_completeness_and_shape() -> None:
    dataset = load_heloc()
    model = dataset.model
    x = dataset.encoded_features[0]
    baseline = np.zeros_like(x)
    
    # Calculate Integrated Gradients
    ig = compute_ig_attribution(model, x, baseline, steps=50)
    
    # Verify shape
    assert ig.shape == x.shape
    
    # Verify Completeness property: Sum(IG) == P(x) - P(baseline)
    model.eval()
    with torch.no_grad():
        p_x = float(torch.sigmoid(model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))).numpy()[0])
        p_base = float(torch.sigmoid(model(torch.tensor(baseline, dtype=torch.float32).unsqueeze(0))).numpy()[0])
        
    expected_diff = p_x - p_base
    actual_sum = np.sum(ig)
    
    # Allow small numerical approximation error due to trapezoidal path approximation (50 steps)
    assert np.abs(expected_diff - actual_sum) < 1e-2


def test_ig_joblib_bundle_exists_and_valid() -> None:
    bundle_path = Path("artifacts/ig/heloc_global_ig.joblib")
    assert bundle_path.exists()
    
    bundle = joblib.load(bundle_path)
    assert "attribution_values" in bundle
    assert "shap_values" in bundle
    assert "base_value" in bundle
    assert bundle["attribution_values"].shape == (150, 46)
