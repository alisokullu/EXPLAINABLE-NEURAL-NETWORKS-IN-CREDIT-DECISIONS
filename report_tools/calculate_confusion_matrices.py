import sys
import json
import torch
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Add parent directory to sys.path to allow importing local modules
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from train_german_dnn import GermanCreditDNN, TARGET_COLUMN as GERMAN_TARGET
from train_dnn import CreditRiskDNN, TARGET_COLUMN as HELOC_TARGET

def compute_confusion_matrix_german():
    data_path = ROOT_DIR / "cleanedDataSets/german_uci_credit_data.csv"
    data = pd.read_csv(data_path)
    features = data.drop(columns=[GERMAN_TARGET])
    target = data[GERMAN_TARGET].astype("int64")
    
    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42
    )
    
    model_path = ROOT_DIR / "artifacts/dnn_german/model.pt"
    checkpoint = torch.load(model_path, map_location="cpu")
    model = GermanCreditDNN(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    preprocessor_path = ROOT_DIR / "artifacts/dnn_german/preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_path)
    x_test_processed = preprocessor.transform(x_test).astype(np.float32)
    
    with torch.no_grad():
        logits = model(torch.tensor(x_test_processed))
        probs = torch.sigmoid(logits).numpy()
        
    metrics_path = ROOT_DIR / "artifacts/dnn_german/metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    threshold = metrics["best_threshold"]
    preds = (probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print("German Credit Confusion Matrix:")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")

def compute_confusion_matrix_heloc():
    data_path = ROOT_DIR / "cleanedDataSets/heloc_dataset.csv"
    data = pd.read_csv(data_path)
    features = data.drop(columns=[HELOC_TARGET])
    target = data[HELOC_TARGET].astype("int64")
    
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42
    )
    
    model_path = ROOT_DIR / "artifacts/dnn_heloc/model.pt"
    checkpoint = torch.load(model_path, map_location="cpu")
    model = CreditRiskDNN(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    scaler_path = ROOT_DIR / "artifacts/dnn_heloc/scaler.joblib"
    scaler = joblib.load(scaler_path)
    x_test_processed = scaler.transform(x_test.to_numpy(dtype=np.float32)).astype(np.float32)
    
    with torch.no_grad():
        logits = model(torch.tensor(x_test_processed))
        probs = torch.sigmoid(logits).numpy()
        
    metrics_path = ROOT_DIR / "artifacts/dnn_heloc/metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    threshold = metrics["best_threshold"]
    preds = (probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print("HELOC Confusion Matrix:")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")

if __name__ == "__main__":
    compute_confusion_matrix_german()
    compute_confusion_matrix_heloc()
