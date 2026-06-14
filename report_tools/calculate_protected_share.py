from pathlib import Path
import joblib
import numpy as np

ROOT_DIR = Path(__file__).parent.parent

def calculate_protected_share(file_path):
    bundle = joblib.load(file_path)
    attr = bundle["attribution_values"] if "attribution_values" in bundle else bundle["shap_values"]
    feature_names = bundle["feature_names"]
    
    # Identify indices of protected attributes (gender and age)
    protected_indices = []
    for i, name in enumerate(feature_names):
        if "personal_status_sex" in name or name == "age":
            protected_indices.append(i)
            
    # Calculate share of absolute attribution
    abs_attr = np.abs(attr)
    total_abs_attr = np.sum(abs_attr, axis=1)
    total_abs_attr[total_abs_attr == 0] = 1e-8
    
    protected_abs_attr = np.sum(abs_attr[:, protected_indices], axis=1)
    shares = protected_abs_attr / total_abs_attr
    mean_share = np.mean(shares)
    
    basename = Path(file_path).name
    print(f"{basename} -> Mean Protected Share: {mean_share:.4f}")
    return mean_share

if __name__ == "__main__":
    calculate_protected_share(ROOT_DIR / "artifacts/shap/german_global_shap.joblib")
    calculate_protected_share(ROOT_DIR / "artifacts/lime/german_global_lime.joblib")
    calculate_protected_share(ROOT_DIR / "artifacts/ig/german_global_ig.joblib")
