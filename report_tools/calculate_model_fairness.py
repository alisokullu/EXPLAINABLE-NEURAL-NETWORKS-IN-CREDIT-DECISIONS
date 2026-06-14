import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add parent directory to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import pandas as pd
import numpy as np
import joblib

from train_german_dnn import GermanCreditDNN, TARGET_COLUMN, NUMERIC_COLUMNS

def calculate_model_fairness():
    # Load data
    data_path = ROOT_DIR / "cleanedDataSets/german_uci_credit_data.csv"
    data = pd.read_csv(data_path)
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN].astype("int64")
    
    # Split
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42
    )
    
    # Load model and preprocessor
    model_path = ROOT_DIR / "artifacts/dnn_german/model.pt"
    checkpoint = torch.load(model_path, map_location="cpu")
    model = GermanCreditDNN(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    preprocessor_path = ROOT_DIR / "artifacts/dnn_german/preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_path)
    
    # Preprocess test set
    x_test_processed = preprocessor.transform(x_test).astype(np.float32)
    
    # Predict
    with torch.no_grad():
        logits = model(torch.tensor(x_test_processed))
        probs = torch.sigmoid(logits).numpy()
    
    metrics_path = ROOT_DIR / "artifacts/dnn_german/metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    threshold = metrics["best_threshold"]
    preds = (probs >= threshold).astype(int)
    
    # Gender column in raw test set: personal_status_sex
    # Females are a92, a95. Males are a91, a93, a94 (or not female)
    is_female = x_test["personal_status_sex"].isin(["a92", "a95"]).values
    
    # Calculate selection rate (percentage predicted Good = 1) for males and females
    selection_rate_female = preds[is_female].mean()
    selection_rate_male = preds[~is_female].mean()
    demographic_parity_diff = abs(selection_rate_male - selection_rate_female)
    
    # True positive rate (Recall) and False positive rate for equalized odds
    tpr_female = preds[is_female & (y_test == 1)].mean() if sum(is_female & (y_test == 1)) > 0 else 0
    tpr_male = preds[~is_female & (y_test == 1)].mean() if sum(~is_female & (y_test == 1)) > 0 else 0
    
    fpr_female = preds[is_female & (y_test == 0)].mean() if sum(is_female & (y_test == 0)) > 0 else 0
    fpr_male = preds[~is_female & (y_test == 0)].mean() if sum(~is_female & (y_test == 0)) > 0 else 0
    
    equalized_odds_diff = max(abs(tpr_male - tpr_female), abs(fpr_male - fpr_female))
    
    # Age binarized into young (< 30) or young/old cohorts
    for age_threshold in [25, 30]:
        is_young = (x_test["age"] < age_threshold).values
        sr_young = preds[is_young].mean()
        sr_old = preds[~is_young].mean()
        dp_diff_age = abs(sr_old - sr_young)
        
        tpr_young = preds[is_young & (y_test == 1)].mean() if sum(is_young & (y_test == 1)) > 0 else 0
        tpr_old = preds[~is_young & (y_test == 1)].mean() if sum(~is_young & (y_test == 1)) > 0 else 0
        fpr_young = preds[is_young & (y_test == 0)].mean() if sum(is_young & (y_test == 0)) > 0 else 0
        fpr_old = preds[~is_young & (y_test == 0)].mean() if sum(~is_young & (y_test == 0)) > 0 else 0
        eo_diff_age = max(abs(tpr_old - tpr_young), abs(fpr_old - fpr_young))
        
        print(f"Age threshold {age_threshold}:")
        print(f"  DP Diff: {dp_diff_age:.4f}")
        print(f"  EO Diff: {eo_diff_age:.4f}")
        
    print("\nGender:")
    print(f"  Selection Rate Female: {selection_rate_female:.4f}")
    print(f"  Selection Rate Male: {selection_rate_male:.4f}")
    print(f"  Demographic Parity Diff: {demographic_parity_diff:.4f}")
    print(f"  TPR Female: {tpr_female:.4f}, Male: {tpr_male:.4f}")
    print(f"  FPR Female: {fpr_female:.4f}, Male: {fpr_male:.4f}")
    print(f"  Equalized Odds Diff: {equalized_odds_diff:.4f}")

if __name__ == "__main__":
    calculate_model_fairness()
