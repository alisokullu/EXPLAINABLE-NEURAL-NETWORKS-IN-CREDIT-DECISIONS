import joblib
import numpy as np

from generate_counterfactuals import load_heloc
from generate_shap import predict_proba_wrapper


def test_lime_bundle_matches_heloc_model_contract() -> None:
    dataset = load_heloc()
    bundle = joblib.load("artifacts/lime/heloc_global_lime.joblib")

    assert bundle["lime_values"].shape == bundle["attribution_values"].shape
    assert bundle["lime_values"].shape[1] == len(dataset.feature_names)
    assert bundle["explain_features_encoded"].shape[1] == len(dataset.feature_names)


def test_lime_prediction_adapter_returns_two_probabilities() -> None:
    dataset = load_heloc()
    pred_fn = predict_proba_wrapper(dataset.model)
    x = dataset.encoded_features[:5]

    p1 = pred_fn(x)
    lime_probs = np.column_stack((1.0 - p1, p1))

    assert lime_probs.shape == (5, 2)
    assert np.allclose(lime_probs.sum(axis=1), 1.0)
    assert np.all((lime_probs >= 0.0) & (lime_probs <= 1.0))
