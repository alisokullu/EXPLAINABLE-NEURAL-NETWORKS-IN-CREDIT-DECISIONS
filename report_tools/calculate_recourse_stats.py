import json
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).parent.parent

def calculate_stats(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    valid_records = [r for r in records if r["valid"]]
    
    total = len(records)
    valid_count = len(valid_records)
    validity_rate = valid_count / total if total > 0 else 0.0
    
    changes = [r["changed_feature_count"] for r in valid_records]
    avg_changes = np.mean(changes) if changes else 0.0
    median_changes = np.median(changes) if changes else 0.0
    
    prob_reductions = [
        (r["counterfactual_probability_good"] - r["original_probability_good"])
        for r in valid_records
    ]
    avg_prob_reduction = np.mean(prob_reductions) if prob_reductions else 0.0
    
    new_probs = [r["counterfactual_probability_good"] for r in valid_records]
    avg_new_prob = np.mean(new_probs) if new_probs else 0.0
    
    basename = Path(json_path).name
    print(f"Stats for {basename}:")
    print(f"  Total Cases: {total}")
    print(f"  Validity Rate: {validity_rate:.4f}")
    print(f"  Valid Cases: {valid_count}")
    print(f"  Avg Changes: {avg_changes:.4f}")
    print(f"  Median Changes: {median_changes:.4f}")
    print(f"  Avg Prob Reduction (Increase): {avg_prob_reduction:.4f}")
    print(f"  Avg New Prob: {avg_new_prob:.4f}")

if __name__ == "__main__":
    calculate_stats(ROOT_DIR / "artifacts/dice/german_counterfactuals.json")
    calculate_stats(ROOT_DIR / "artifacts/dice/heloc_counterfactuals.json")
