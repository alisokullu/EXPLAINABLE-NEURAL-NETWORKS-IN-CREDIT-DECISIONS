import numpy as np

import dashboard


def test_dashboard_loads_dataset_specific_models() -> None:
    heloc = dashboard.load_model_and_dataset("HELOC")
    german = dashboard.load_model_and_dataset("German Credit")

    assert heloc.name == "heloc"
    assert german.name == "german"
    assert heloc.encoded_features.shape[1] == len(heloc.feature_names) == 46
    assert german.encoded_features.shape[1] == len(german.feature_names) == 61


def test_lime_explainer_cache_is_keyed_by_dataset_name() -> None:
    heloc_explainer = dashboard.get_lime_explainer("HELOC")
    german_explainer = dashboard.get_lime_explainer("German Credit")

    assert len(heloc_explainer.feature_names) == 46
    assert len(german_explainer.feature_names) == 61
    assert heloc_explainer.feature_names != german_explainer.feature_names


def test_shap_value_normalization_helpers_use_good_class() -> None:
    values = np.arange(6).reshape(1, 3, 2)

    assert dashboard.scalar_value([0.1, 0.9]) == 0.9
    assert dashboard.one_dimensional_attributions(values).tolist() == [1.0, 3.0, 5.0]
