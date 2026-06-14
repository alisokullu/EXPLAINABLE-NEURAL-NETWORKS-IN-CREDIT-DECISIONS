import numpy as np

from evaluate_xai import _log_stability_ratio, _model_logits, _relative_l2_change
from generate_counterfactuals import load_german
from generate_shap import prepare_data


def test_prepare_data_returns_explained_row_indices() -> None:
    dataset = load_german()
    _, _, explain_positions, x_explain, explain_rows = prepare_data(dataset)

    assert len(explain_positions) == len(x_explain) == len(explain_rows)
    assert x_explain.shape[1] == len(dataset.feature_names)
    assert np.issubdtype(explain_rows.dtype, np.integer)
    assert dataset.features.iloc[explain_rows].shape[0] == len(x_explain)


def test_stability_helpers_use_relative_change_and_logits() -> None:
    dataset = load_german()
    x = dataset.encoded_features[:3]

    assert _relative_l2_change(x, x) == 0.0
    assert _relative_l2_change(x, x + 0.01) > 0.0
    assert _model_logits(dataset, x).shape == (3,)
    assert _log_stability_ratio(1.0) == 0.0
    assert _log_stability_ratio(np.e) == 1.0
