import numpy as np

from generate_counterfactuals import load_heloc
from generate_shap import predict_proba_wrapper


def test_prediction_wrapper_is_bounded_and_sensitive() -> None:
    dataset = load_heloc()
    pred_fn = predict_proba_wrapper(dataset.model)
    x = dataset.encoded_features[:2]

    probabilities = pred_fn(x)
    perturbed = x.copy()
    perturbed[:, 0] += 0.1
    perturbed_probabilities = pred_fn(perturbed)

    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.all((perturbed_probabilities >= 0.0) & (perturbed_probabilities <= 1.0))
    assert np.any(np.abs(probabilities - perturbed_probabilities) > 1e-6)
