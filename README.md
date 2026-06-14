# Explainability AI on Credit Decisions

This project cleans credit datasets and trains a DNN classifier for credit risk performance.

## Datasets

- `heloc_dataset.csv`: supervised-ready after cleaning. `RiskPerformance` is encoded as `good=1`, `bad=0`.
- `german_credit_data.csv`: cleaned for analysis, but excluded from supervised DNN training because this CSV does not contain a real target/risk label.
- `german_uci_credit_raw.data`: official UCI Statlog German Credit data. It is downloaded during cleaning and used for German Credit DNN training.

## Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Clean the datasets:

```bash
python3 clean_datasets.py
```

Train the DNN model:

```bash
python3 train_dnn.py
```

Train the German Credit DNN model:

```bash
python3 train_german_dnn.py
```

Generate precomputed global SHAP values:

```bash
python3 generate_shap.py
```

Generate precomputed global LIME values:

```bash
python3 generate_lime.py
```

Generate DiCE-style diverse counterfactual explanations:

```bash
python3 generate_counterfactuals.py
```

Generate three ID-based decisions for every rejected application:

```bash
python3 generate_counterfactuals.py --dataset both --all-rejected-decisions
```

Run the local dashboard:

```bash
streamlit run dashboard.py
```

The counterfactual generator is actionability-aware:

- German Credit changes only loan terms, installment burden, guarantor/co-applicant status, and other installment plan status.
- HELOC changes debt burden, utilization, balance-carrying accounts, high-utilization accounts, and recent inquiry features.
- Dataset columns do not include direct income, so income can only be discussed through German Credit's installment burden proxy.

For a specific dataset or row:

```bash
python3 generate_counterfactuals.py --dataset german --query-index 538
python3 generate_counterfactuals.py --dataset heloc --num-examples 2 --total-cfs 4
```

Current optimized test metrics:

- HELOC DNN: `accuracy=0.7413`, `roc_auc=0.8089`
- German Credit DNN: `accuracy=0.7550`, `roc_auc=0.7769`

Optimization notes:

- HELOC keeps special-missing indicator columns before median imputation.
- Both DNN scripts select the best epoch by validation ROC-AUC.
- Accuracy is reported with a validation-selected threshold when it improves enough over `0.5`.

Outputs:

- Cleaned datasets: `cleanedDataSets/`
- Cleaning report: `cleanedDataSets/cleaning_report.json`
- HELOC model artifacts: `artifacts/dnn_heloc/`
- German Credit model artifacts: `artifacts/dnn_german/`
- SHAP explanations: `artifacts/shap/`
- Counterfactual explanations: `artifacts/dice/`
- ID-based rejected decisions: `artifacts/dice/*_rejected_decisions.csv`

## OpenXAI Metrics Benchmark

Our XAI pipeline evaluates the reliability of SHAP explanations against the standardized [OpenXAI Leaderboard](https://open-xai.github.io/leaderboard) metrics for Neural Networks. 

The generated metrics are stored in `artifacts/xai_metrics/metrics_summary.json`. Re-run `python3 evaluate_xai.py` after regenerating SHAP/LIME bundles so the benchmark reflects the current artifact contract and explained row indices. RIS/ROS/RRS are reported on a natural-log stability scale, where values closer to `0` indicate a stability ratio closer to `1`.

### German Credit Comparison

| Metric | OpenXAI Baseline (SHAP) | Our Implementation (SHAP) | Description |
|---|---|---|---|
| **Faithfulness: PGI** (Higher is better) | 0.44 | 0.26 | Prediction Gap on Important features. Drop in probability when important features are perturbed. |
| **Faithfulness: PGU** (Lower is better) | 0.64 | **0.02** | Prediction Gap on Unimportant features. Drop in probability when unimportant features are perturbed. |
| **Stability: RIS** (Closer to 0 is better) | 1.28 | **1.14** | Relative Input Stability. Maximum change in explanation relative to small changes in input. |
| **Stability: ROS** (Closer to 0 is better) | 4.46 | **1.77** | Relative Output Stability. Change in explanation relative to model output/logit change. |
| **Stability: RRS** (Closer to 0 is better) | 11.46 | **0.74** | Relative Representation Stability. Change in explanation relative to internal representation. |
| **Fairness@PGI** (Closer to 0 is better) | 0.102 | **0.013** | Disparity of PGI across protected groups (Gender). |
| **Fairness@PGU** (Closer to 0 is better) | 0.009 | **0.005** | Disparity of PGU across protected groups (Gender). |
| **Fairness@RIS** (Closer to 0 is better) | **0.021** | 0.403 | Disparity of RIS across protected groups (Gender). |
| **Fairness@ROS** (Closer to 0 is better) | 0.361 | **0.107** | Disparity of ROS across protected groups (Gender). |
| **Fairness@RRS** (Closer to 0 is better) | 0.032 | 0.378 | Disparity of RRS across protected groups (Gender). |

### HELOC Comparison

| Metric | OpenXAI Baseline (SHAP) | Our Implementation (SHAP) | Description |
|---|---|---|---|
| **Faithfulness: PGI** (Higher is better) | 0.26 | **0.30** | Prediction Gap on Important features. |
| **Faithfulness: PGU** (Lower is better) | 0.23 | **0.02** | Prediction Gap on Unimportant features. |
| **Stability: RIS** (Closer to 0 is better) | 1.57 | **0.86** | Relative Input Stability. |
| **Stability: ROS** (Closer to 0 is better) | 3.80 | **1.63** | Relative Output Stability. |
| **Stability: RRS** (Closer to 0 is better) | 9.40 | **0.91** | Relative Representation Stability. |

**Key Takeaways:**
- **PGI Performance**: For HELOC, our SHAP explanations identify more impactful features than the baseline (0.30 vs 0.26).
- **PGU Performance**: Low PGU values suggest that features ranked as unimportant have limited effect when perturbed.
- **RIS & RRS Performance**: Lower RIS/RRS values suggest stable explanations under small encoded-input changes and closer alignment with model representations.
- **Fairness Performance (German Credit)**: Fairness metrics compare explanation quality gaps across gender-coded groups in the UCI dataset. These should be interpreted as research diagnostics, not production compliance guarantees.
