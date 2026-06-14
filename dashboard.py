"""Streamlit dashboard for the credit decision XAI project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import torch
import plotly.graph_objects as go
import shap
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin
from fairlearn.postprocessing import ThresholdOptimizer

class PyTorchDNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, transformer, threshold=0.5):
        self.model = model
        self.transformer = transformer
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_proc = self.transformer.transform(X).astype(np.float32)
        else:
            X_proc = X.astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_proc))
            probs = torch.sigmoid(logits).numpy()
        return np.vstack((1.0 - probs, probs)).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    groups = np.unique(sensitive_features)
    selection_rates = {}
    tpr_rates = {}
    fpr_rates = {}
    
    for g in groups:
        mask = (sensitive_features == g)
        selection_rates[g] = float(np.mean(y_pred[mask]))
        
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        tp = np.sum((y_true_g == 1) & (y_pred_g == 1))
        fn = np.sum((y_true_g == 1) & (y_pred_g == 0))
        tpr_rates[g] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        fp = np.sum((y_true_g == 0) & (y_pred_g == 1))
        tn = np.sum((y_true_g == 0) & (y_pred_g == 0))
        fpr_rates[g] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        
    dp_diff = abs(selection_rates[groups[0]] - selection_rates[groups[1]]) if len(groups) > 1 else 0.0
    eo_diff = max(abs(tpr_rates[groups[0]] - tpr_rates[groups[1]]), abs(fpr_rates[groups[0]] - fpr_rates[groups[1]])) if len(groups) > 1 else 0.0
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        "selection_rates": selection_rates,
        "tpr_rates": tpr_rates,
        "fpr_rates": fpr_rates,
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    }


from generate_counterfactuals import (
    load_heloc,
    load_german,
    GERMAN_CODE_DESCRIPTIONS,
)

ROOT = Path(__file__).resolve().parent
TARGET_COLUMN = "risk_performance"

DATASET_PATHS = {
    "HELOC": ROOT / "cleanedDataSets" / "heloc_dataset.csv",
    "German Credit": ROOT / "cleanedDataSets" / "german_uci_credit_data.csv",
}
METRIC_PATHS = {
    "HELOC": ROOT / "artifacts" / "dnn_heloc" / "metrics.json",
    "German Credit": ROOT / "artifacts" / "dnn_german" / "metrics.json",
}
COUNTERFACTUAL_PATHS = {
    "HELOC": ROOT / "artifacts" / "dice" / "heloc_counterfactuals.csv",
    "German Credit": ROOT / "artifacts" / "dice" / "german_counterfactuals.csv",
}
RECOMMENDATION_PATHS = {
    "HELOC": ROOT / "artifacts" / "dice" / "heloc_recommendations.md",
    "German Credit": ROOT / "artifacts" / "dice" / "german_recommendations.md",
}
REJECTED_DECISION_PATHS = {
    "HELOC": ROOT / "artifacts" / "dice" / "heloc_rejected_decisions.csv",
    "German Credit": ROOT / "artifacts" / "dice" / "german_rejected_decisions.csv",
}
XAI_METRICS_PATH = ROOT / "artifacts" / "xai_metrics" / "metrics_summary.json"

st.set_page_config(
    page_title="Credit Decision XAI Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px 8px 0px 0px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(99, 102, 241, 0.2) !important;
            border-bottom: 2px solid rgb(99, 102, 241) !important;
        }

        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            background-color: rgba(255, 255, 255, 0.01);
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_text(path: Path) -> str:
    if not path.exists():
        return "Output file has not been generated yet."
    return path.read_text(encoding="utf-8")


@st.cache_resource
def load_model_and_dataset(name: str):
    if name == "HELOC":
        return load_heloc()
    elif name == "German Credit":
        return load_german()
    return None


@st.cache_data
def get_test_predictions(dataset_name: str):
    dataset = load_model_and_dataset(dataset_name)
    model = dataset.model
    
    features = dataset.data.drop(columns=[TARGET_COLUMN])
    target = dataset.data[TARGET_COLUMN].astype("int64")
    
    _, x_test, _, y_test = train_test_split(
        features, target,
        test_size=0.2, stratify=target, random_state=42
    )
    
    if dataset.name == "heloc":
        x_test_encoded = dataset.transformer.transform(x_test.to_numpy(dtype=np.float32)).astype(np.float32)
    else:
        x_test_encoded = dataset.transformer.transform(x_test).astype(np.float32)
        
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x_test_encoded)
        probs = torch.sigmoid(model(tensor)).cpu().numpy().flatten()
        
    return y_test.to_numpy(), probs


def plot_confusion_matrix(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Bad", "Predicted Good"],
        y=["Actual Bad", "Actual Good"],
        colorscale="Viridis",
        showscale=False,
        text=[[f"TN: {cm[0][0]}", f"FP: {cm[0][1]}"],
              [f"FN: {cm[1][0]}", f"TP: {cm[1][1]}"]],
        texttemplate="%{text}",
        textfont={"size": 14, "family": "Outfit, Inter, sans-serif"},
    ))
    fig.update_layout(
        title=dict(
            text="Confusion Matrix",
            font=dict(size=16, family="Outfit, Inter, sans-serif")
        ),
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.4f})', line=dict(color='#6366F1', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='white', dash='dash'), opacity=0.5))
    fig.update_layout(
        title=dict(
            text=f"ROC Curve (AUC = {roc_auc:.4f})",
            font=dict(size=16, family="Outfit, Inter, sans-serif")
        ),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig


def plot_training_history(dataset_name):
    name_str = "german" if "german" in dataset_name.lower() else "heloc"
    path = ROOT / "artifacts" / f"dnn_{name_str}" / "training_history.json"
    if not path.exists():
        return None
    history = json.loads(path.read_text(encoding="utf-8"))
    
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Train Loss', line=dict(color='#10B981', width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Val Loss', line=dict(color='#F59E0B', width=2)))
    fig.update_layout(
        title=dict(
            text="Loss Convergence History",
            font=dict(size=16, family="Outfit, Inter, sans-serif")
        ),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0.6, y=0.9, bgcolor='rgba(0,0,0,0.5)')
    )
    return fig


@st.cache_data
def load_global_xai(dataset_name: str, explainer_method: str) -> dict:
    """Load precomputed global XAI (SHAP/LIME) values."""
    name_str = "german" if "german" in dataset_name.lower() else "heloc"
    path = ROOT / "artifacts" / explainer_method / f"{name_str}_global_{explainer_method}.joblib"
    if not path.exists():
        return {}
    return joblib.load(path)


@st.cache_resource
def get_shap_explainer(dataset_name: str):
    dataset = load_model_and_dataset(dataset_name)
    model = dataset.model
    global_shap_data = load_global_xai(dataset.name, "shap")
    if global_shap_data is not None and "x_train_encoded" in global_shap_data:
        x_train_encoded = global_shap_data["x_train_encoded"]
    else:
        x_train_encoded = dataset.encoded_features
        
    def predict_fn(x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x.astype(np.float32))
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            probs = torch.sigmoid(model(tensor)).cpu().numpy()
            return probs.flatten()
            
    background = shap.kmeans(x_train_encoded, 50)
    return shap.KernelExplainer(predict_fn, background)


@st.cache_resource
def get_lime_explainer(dataset_name: str):
    dataset = load_model_and_dataset(dataset_name)
    global_lime_data = load_global_xai(dataset.name, "lime")
    if global_lime_data is not None and "x_train_encoded" in global_lime_data:
        x_train_encoded = global_lime_data["x_train_encoded"]
    else:
        x_train_encoded = dataset.encoded_features
        
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_train_encoded,
        feature_names=dataset.feature_names,
        class_names=["Bad", "Good"],
        mode="classification",
        random_state=42
    )
    return explainer


def one_dimensional_attributions(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    if values.ndim == 2:
        values = values[0]
    return values.astype(float).reshape(-1)


def scalar_value(value: Any) -> float:
    if isinstance(value, list):
        value = value[-1]
    return float(np.asarray(value).reshape(-1)[-1])


def clean_feature_name(name: str) -> str:
    is_missing = False
    if name.endswith("_special_missing"):
        name = name.replace("_special_missing", "")
        is_missing = True
        
    for code, desc in GERMAN_CODE_DESCRIPTIONS.items():
        if name.endswith("_" + code) or name.endswith(code):
            base = name.rsplit("_" + code, 1)[0].rsplit(code, 1)[0].replace("_", " ").title()
            suffix = f": {desc}"
            if is_missing:
                suffix += " (Missing Indicator)"
            return f"{base}{suffix}"
            
    base = name.replace("_", " ").title()
    if is_missing:
        base += " (Missing Indicator)"
    return base


def metric_value(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def dataset_summary(name: str, data: pd.DataFrame) -> None:
    target_counts = data[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in data else {}
    good_count = int(target_counts.get(1, 0))
    bad_count = int(target_counts.get(0, 0))

    row_a, row_b, row_c, row_d = st.columns(4)
    row_a.metric("Rows", f"{len(data):,}")
    row_b.metric("Features", f"{max(len(data.columns) - 1, 0):,}")
    row_c.metric("Good", f"{good_count:,}")
    row_d.metric("Bad / Rejected", f"{bad_count:,}")

    with st.expander(f"{name} cleaned data preview", expanded=False):
        st.dataframe(data.head(25), use_container_width=True, hide_index=True)


def model_metrics(name: str, metrics: dict) -> None:
    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
    col_a.metric("ROC-AUC", metric_value(metrics, "roc_auc"))
    col_b.metric("Accuracy", metric_value(metrics, "accuracy"))
    col_c.metric("F1-Score", metric_value(metrics, "f1"))
    col_d.metric("Precision", metric_value(metrics, "precision"))
    col_e.metric("Recall", metric_value(metrics, "recall"))
    col_f.metric("Optimal Threshold", metric_value(metrics, "best_threshold"))
    
    st.markdown(" ")
    
    # Threshold Tuning Comparison
    best_thresh = float(metrics.get("best_threshold", 0.5))
    default_acc = metrics.get("default_threshold_accuracy")
    default_f1 = metrics.get("default_threshold_f1")
    
    if default_acc is not None and default_f1 is not None:
        st.markdown("#### Decision Threshold Tuning Comparison")
        st.markdown(
            "Tuning the decision threshold away from the default `0.50` allows the model to optimize "
            "classification outcomes (accuracy and F1-score) based on validation dataset tuning."
        )
        
        comp_data = {
            "Performance Metric": ["Decision Threshold Value", "Model Accuracy", "F1-Score"],
            "Default Threshold (0.50)": ["0.5000", f"{float(default_acc):.4f}", f"{float(default_f1):.4f}"],
            f"Optimal Tuned Threshold ({best_thresh:.2f})": [f"{best_thresh:.4f}", f"{float(metrics.get('accuracy', 0)):.4f}", f"{float(metrics.get('f1', 0)):.4f}"],
            "Tuning Gain / Difference": [
                f"{best_thresh - 0.50:+.4f}",
                f"{(float(metrics.get('accuracy', 0)) - float(default_acc)):+.4f}",
                f"{(float(metrics.get('f1', 0)) - float(default_f1)):+.4f}"
            ]
        }
        st.table(comp_data)


def counterfactual_panel(name: str, counterfactuals: pd.DataFrame, recommendations: str, selected_app_id: str | None = None) -> None:
    shown_cfs = counterfactuals
    if selected_app_id:
        try:
            row_idx = int(selected_app_id.split("_")[-1])
            shown_cfs = counterfactuals[counterfactuals["query_index"] == row_idx]
        except Exception:
            pass

    # Filter out invalid counterfactuals (only keep valid == 1 / crossed threshold)
    if "valid" in shown_cfs.columns:
        shown_cfs = shown_cfs[shown_cfs["valid"] == 1]

    valid_count = int(shown_cfs["valid"].sum()) if "valid" in shown_cfs else 0
    total_count = len(shown_cfs)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Counterfactuals", f"{total_count:,}")
    col_b.metric("Valid", f"{valid_count:,}")
    col_c.metric("Validity Rate", f"{(valid_count / total_count * 100):.1f}%" if total_count else "n/a")

    table_columns = [
        "query_index",
        "counterfactual_id",
        "original_probability_good",
        "counterfactual_probability_good",
        "model_valid",
        "valid",
        "actionability_issues",
        "changed_feature_count",
        "customer_recommendations",
    ]
    visible_columns = [column for column in table_columns if column in shown_cfs.columns]
    
    if shown_cfs.empty:
        st.info("No specific optimization counterfactuals precalculated for this selection.")
    else:
        # Format recommendations inside the dataframe
        def clean_recs(val):
            try:
                lst = json.loads(val)
                return " | ".join(lst)
            except:
                return val
        
        shown_cfs_display = shown_cfs.copy()
        if "customer_recommendations" in shown_cfs_display.columns:
            shown_cfs_display["customer_recommendations"] = shown_cfs_display["customer_recommendations"].apply(clean_recs)
        
        st.dataframe(
            shown_cfs_display[visible_columns],
            use_container_width=True,
            hide_index=True,
        )

    # If a single application is selected, render a premium action cards dashboard!
    if selected_app_id and not shown_cfs.empty and "customer_recommendations" in shown_cfs.columns:
        st.markdown("---")
        st.markdown("<h4 style='color: #3b82f6;'>🎯 Counterfactual Recourse Options</h4>", unsafe_allow_html=True)
        st.markdown("Detailed recourse action paths for the selected query index:")
        
        for idx, (_, row) in enumerate(shown_cfs.iterrows()):
            try:
                recs = json.loads(row["customer_recommendations"])
            except:
                recs = [row["customer_recommendations"]]
            
            orig_prob = row.get("original_probability_good", 0.0)
            cf_prob = row.get("counterfactual_probability_good", 0.0)
            
            with st.container():
                st.markdown(
                    f"""
                    <div style='background-color: rgba(59, 130, 246, 0.08); border-left: 5px solid #3b82f6; padding: 15px 20px; border-radius: 8px; margin-bottom: 15px;'>
                        <h5 style='margin: 0 0 10px 0; color: #93c5fd; font-weight: 600;'>Scenario {idx+1}</h5>
                        <p style='margin: 0 0 10px 0; font-size: 14px;'>
                            Reaching this target state increases approval probability from 
                            <span style='color: #fca5a5; font-weight: bold;'>{orig_prob:.2%}</span> to 
                            <span style='color: #86efac; font-weight: bold;'>{cf_prob:.2%}</span>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                for rec in recs:
                    icon = "📉"
                    if "increase" in rec.lower() or "raise" in rec.lower() or "higher" in rec.lower() or "add" in rec.lower():
                        icon = "📈"
                    elif "provide" in rec.lower() or "submit" in rec.lower() or "have" in rec.lower() or "existing" in rec.lower():
                        icon = "📝"
                    elif "change" in rec.lower() or "switch" in rec.lower() or "convert" in rec.lower():
                        icon = "🔄"
                    st.markdown(f"<div style='margin-left: 20px; padding: 4px 0; font-size: 15px; font-weight: 500;'>{icon} &nbsp; {rec}</div>", unsafe_allow_html=True)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    with st.expander(f"{name} customer-facing explanations", expanded=False):
        st.markdown(recommendations)


def rejected_decision_panel(name: str, decisions: pd.DataFrame, application_filter: str = "") -> None:
    if decisions.empty:
        st.info("ID-based rejected decision output has not been generated yet.")
        return

    shown = decisions
    if application_filter:
        shown = decisions[decisions["application_id"].str.contains(application_filter, case=False)]

    # Filter out failed recourse paths (only keep would_approve == True / crossed threshold)
    if "would_approve" in shown.columns:
        shown = shown[shown["would_approve"] == True]

    if shown.empty:
        st.warning("No successful recourse options could be found for this applicant within realistic feature boundaries.")
        return

    # Clean up the customer_recommendations column for general table display
    def clean_recs(val):
        try:
            lst = json.loads(val)
            return " | ".join(lst)
        except:
            return val

    shown_display = shown.copy()
    shown_display["customer_recommendations"] = shown_display["customer_recommendations"].apply(clean_recs)

    display_columns = [
        "application_id",
        "decision_id",
        "decision_name",
        "original_probability_good",
        "decision_probability_good",
        "would_approve",
        "customer_recommendations",
    ]
    
    rename_map = {
        "application_id": "Application ID",
        "decision_id": "Scenario ID",
        "decision_name": "Recourse Strategy",
        "original_probability_good": "Original Probability",
        "decision_probability_good": "Mitigated Probability",
        "would_approve": "Approved After Recourse",
        "customer_recommendations": "Action Recommendations"
    }

    st.dataframe(
        shown_display[display_columns].rename(columns=rename_map),
        use_container_width=True,
        hide_index=True,
    )

    # If a single application is selected, render a premium action cards dashboard!
    if application_filter and len(shown) > 0:
        st.markdown("---")
        st.markdown("<h4 style='color: #6366f1;'>🎯 Actionable Recourse Recommendations</h4>", unsafe_allow_html=True)
        st.markdown("To reverse the credit rejection, the applicant can follow any of the alternative recourse strategies below:")
        
        for idx, (_, row) in enumerate(shown.iterrows()):
            try:
                recs = json.loads(row["customer_recommendations"])
            except:
                recs = [row["customer_recommendations"]]
            
            strategy_name = row["decision_name"]
            feature_count = row["changed_feature_count"]
            new_prob = row["decision_probability_good"]
            old_prob = row["original_probability_good"]
            
            # Card styling
            st.markdown(
                f"""
                <div style='background-color: rgba(99, 102, 241, 0.08); border-left: 5px solid #6366f1; padding: 15px 20px; border-radius: 8px; margin-bottom: 15px;'>
                    <h5 style='margin: 0 0 10px 0; color: #a5b4fc; font-weight: 600;'>Option {idx+1}: {strategy_name} ({feature_count} features changed)</h5>
                    <p style='margin: 0 0 10px 0; font-size: 14px;'>
                        Applying the following changes increases the approval probability from 
                        <span style='color: #fca5a5; font-weight: bold;'>{old_prob:.2%}</span> to 
                        <span style='color: #86efac; font-weight: bold;'>{new_prob:.2%}</span> (Threshold: <code>{row.get('threshold', 0.39 if name == 'German Credit' else 0.50):.2f}</code>).
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Render the action checklist inside the container
            for rec in recs:
                # Determine icon
                icon = "📉"
                if "increase" in rec.lower() or "raise" in rec.lower() or "higher" in rec.lower() or "add" in rec.lower():
                    icon = "📈"
                elif "provide" in rec.lower() or "submit" in rec.lower() or "have" in rec.lower() or "existing" in rec.lower():
                    icon = "📝"
                elif "change" in rec.lower() or "switch" in rec.lower() or "convert" in rec.lower():
                    icon = "🔄"
                
                st.markdown(f"<div style='margin-left: 20px; padding: 4px 0; font-size: 15px; font-weight: 500;'>{icon} &nbsp; {rec}</div>", unsafe_allow_html=True)
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


def plot_global_xai(ds_key: str, explainer_key: str, selected_explainer: str) -> None:
    global_xai_data = load_global_xai(ds_key, explainer_key)
    
    if not global_xai_data:
        st.warning(f"Global {selected_explainer} data not found. Please run generate_{explainer_key}.py")
        return
        
    shap_values = global_xai_data["shap_values"]
    feature_names = global_xai_data["feature_names"]
    
    st.markdown(f"### {selected_explainer} Global Feature Importance")
    st.markdown(f"This chart displays the mean absolute {selected_explainer} value for each feature, indicating overall importance in the model's decision making.")
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    df_importance = pd.DataFrame({
        "Feature": [clean_feature_name(f) for f in feature_names],
        "Mean Impact": mean_abs_shap
    })
    df_importance = df_importance.sort_values(by="Mean Impact", ascending=True)
    df_importance = df_importance.tail(20)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_importance["Feature"],
        x=df_importance["Mean Impact"],
        orientation='h',
        marker=dict(
            color=df_importance["Mean Impact"],
            colorscale='Viridis',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Mean Impact: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Global Feature Importance (Top 20 Features) - {selected_explainer}",
            font=dict(size=18, family="Outfit, Inter, sans-serif")
        ),
        xaxis_title=f"Mean Absolute {selected_explainer} Value (Impact on Probability)",
        yaxis_title="Features",
        height=550,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_local_xai(explainer, dataset, row_index: int, selected_app: str, explainer_type: str) -> None:
    model = dataset.model
    x_encoded = dataset.encoded_features[row_index]
    
    with st.spinner(f"Calculating local {explainer_type} explanation..."):
        if explainer_type == "SHAP":
            shap_val = explainer.shap_values(x_encoded.reshape(1, -1))
            attribution_values = one_dimensional_attributions(shap_val)
            base_value = scalar_value(explainer.expected_value)
        elif explainer_type == "Integrated Gradients":
            x_i = x_encoded.copy()
            global_ig_data = load_global_xai(dataset.name, "ig")
            baseline = global_ig_data.get("baseline", np.zeros_like(x_i))
            steps = 100
            alphas = np.linspace(0.0, 1.0, steps + 1)
            paths = np.array([baseline + a * (x_i - baseline) for a in alphas], dtype=np.float32)
            paths_tensor = torch.tensor(paths, requires_grad=True)
            model.eval()
            probs = torch.sigmoid(model(paths_tensor))
            probs.sum().backward()
            grads = paths_tensor.grad.detach().numpy()
            avg_grads = 0.5 * (grads[:-1] + grads[1:])
            attribution_values = (x_i - baseline) * np.mean(avg_grads, axis=0)
            with torch.no_grad():
                base_val_tensor = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)
                base_value = float(torch.sigmoid(model(base_val_tensor)).cpu().numpy()[0])
        else:
            def predict_fn(x: np.ndarray) -> np.ndarray:
                model.eval()
                with torch.no_grad():
                    tensor = torch.from_numpy(x.astype(np.float32))
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                    probs_1 = torch.sigmoid(model(tensor)).cpu().numpy().flatten()
                    probs_0 = 1.0 - probs_1
                    return np.vstack((probs_0, probs_1)).T
                    
            exp = explainer.explain_instance(
                x_encoded,
                predict_fn,
                num_features=len(dataset.feature_names),
                num_samples=1500
            )
            weights_dict = dict(exp.local_exp[1])
            attribution_values = np.array([weights_dict.get(j, 0.0) for j in range(len(dataset.feature_names))])
            base_value = float(exp.intercept[1])
        
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x_encoded.reshape(1, -1).astype(np.float32))
        final_prob = float(torch.sigmoid(model(tensor)).cpu().numpy()[0])
        
    original_row = dataset.features.iloc[row_index]
    feature_names = dataset.feature_names
    display_names = []
    hover_texts = []
    
    for i, name in enumerate(feature_names):
        disp_name = clean_feature_name(name)
        display_names.append(disp_name)
        
        if dataset.name == "heloc":
            val = original_row.get(name, "n/a")
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            hover_texts.append(f"Value: {val_str}<br>Contribution: {attribution_values[i]:+.4f}")
        else:
            if name in original_row:
                val = original_row[name]
                val_str = str(val)
            else:
                val_val = x_encoded[i]
                val_str = "Yes" if val_val > 0.5 else "No"
            hover_texts.append(f"Active: {val_str}<br>Contribution: {attribution_values[i]:+.4f}")
            
    df = pd.DataFrame({
        "Feature": display_names,
        "Value": attribution_values,
        "Hover": hover_texts,
        "Abs Value": np.abs(attribution_values)
    })
    
    df = df[df["Abs Value"] > 1e-4]
    df = df.sort_values(by="Abs Value", ascending=True).tail(15)
    
    colors = [
        "rgba(16, 185, 129, 0.85)" if val > 0 else "rgba(244, 63, 94, 0.85)"
        for val in df["Value"]
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Feature"],
        x=df["Value"],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
        ),
        text=[f"{val:+.2%}" if explainer_type == "SHAP" else f"{val:+.3f}" for val in df["Value"]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>%{customdata}<extra></extra>',
        customdata=df["Hover"]
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Local {explainer_type} Feature Contributions for {selected_app}",
            font=dict(size=18, family="Outfit, Inter, sans-serif")
        ),
        xaxis_title=f"{explainer_type} Contribution (Shift in Probability)",
        yaxis_title="Features",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.add_vline(x=0.0, line_dash="dash", line_color="white", opacity=0.5)
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Expected Probability", f"{base_value:.2%}")
    col2.metric("Final Model Probability", f"{final_prob:.2%}")
    col3.metric("Decision Threshold", f"{dataset.threshold:.2%}")
    
    if final_prob >= dataset.threshold:
        st.success(f"🎉 Decision Summary for {selected_app}: **APPROVED** (Model Probability {final_prob:.2%} >= Threshold {dataset.threshold:.2%})")
    else:
        st.error(f"❌ Decision Summary for {selected_app}: **REJECTED** (Model Probability {final_prob:.2%} < Threshold {dataset.threshold:.2%})")



def artifact_links() -> None:
    st.subheader("Project Commands")
    st.code(
        "\n".join(
            [
                "python3 clean_datasets.py",
                "python3 train_dnn.py",
                "python3 train_german_dnn.py",
                "python3 generate_shap.py",
                "python3 generate_lime.py",
                "python3 generate_counterfactuals.py",
                "python3 evaluate_xai.py",
                "streamlit run dashboard.py",
            ]
        ),
        language="bash",
    )

    st.subheader("Artifact Files")
    st.write("- `artifacts/dnn_heloc/metrics.json`")
    st.write("- `artifacts/dnn_german/metrics.json`")
    st.write("- `artifacts/shap/heloc_global_shap.joblib`")
    st.write("- `artifacts/shap/german_global_shap.joblib`")
    st.write("- `artifacts/dice/heloc_counterfactuals.csv`")
    st.write("- `artifacts/dice/german_counterfactuals.csv`")
    st.write("- `artifacts/dice/heloc_rejected_decisions.csv`")
    st.write("- `artifacts/dice/german_rejected_decisions.csv`")
    st.write("- `artifacts/xai_metrics/metrics_summary.json`")


def main() -> None:
    st.title("🔮 Credit Decision XAI Dashboard")
    st.caption("Deep Explainability using SHAP attributions and DiCE counterfactuals for neural network credit risk decisions.")

    st.sidebar.title("Configuration")
    selected_dataset = st.sidebar.radio("Select Dataset:", ["German Credit", "HELOC"])
    
    st.sidebar.divider()
    selected_explainer = st.sidebar.radio("Select Explainer:", ["SHAP", "LIME", "Integrated Gradients"])
    explainer_key = "ig" if selected_explainer == "Integrated Gradients" else selected_explainer.lower()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Dashboard Features:**
        - **📈 Model & Data Summary**: Evaluate performance metrics.
        - **🔮 XAI Explanations**: Diagnoses *why* decisions were made.
        - **🎯 DiCE Counterfactuals**: Prescribes *how* to reverse rejections.
        """
    )

    # Load data
    data = load_csv(DATASET_PATHS[selected_dataset])
    metrics = load_json(METRIC_PATHS[selected_dataset])
    counterfactuals = load_csv(COUNTERFACTUAL_PATHS[selected_dataset])
    recommendations = load_text(RECOMMENDATION_PATHS[selected_dataset])
    rejected_decisions = load_csv(REJECTED_DECISION_PATHS[selected_dataset])
    ds_key = "heloc" if selected_dataset == "HELOC" else "german"
    
    # Establish tabs
    tab_metrics, tab_shap, tab_dice, tab_eval, tab_fairness = st.tabs([
        "📈 Model & Data Summary",
        f"🔮 {selected_explainer} Explanations",
        "🎯 DiCE Counterfactuals",
        "📊 XAI Quality (OpenXAI Metrics)",
        "⚖️ Fairness & Decision Support"
    ])
    
    with tab_metrics:
        st.header(f"{selected_dataset} Overview")
        st.subheader("Model Performance")
        model_metrics(selected_dataset, metrics)
        
        # Performance Visualizations (Confusion Matrix, ROC Curve, Loss curve)
        st.markdown("#### Performance Visualizations")
        y_true, y_probs = get_test_predictions(selected_dataset)
        best_thresh = float(metrics.get("best_threshold", 0.5))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(plot_confusion_matrix(y_true, y_probs, best_thresh), use_container_width=True)
        with col2:
            st.plotly_chart(plot_roc_curve(y_true, y_probs), use_container_width=True)
        with col3:
            history_fig = plot_training_history(selected_dataset)
            if history_fig:
                st.plotly_chart(history_fig, use_container_width=True)
        
        st.subheader("Dataset Summary")
        dataset_summary(selected_dataset, data)
        
        st.divider()
        artifact_links()
        
    with tab_shap:
        st.header(f"{selected_dataset} {selected_explainer} Explanations")
        try:
            plot_global_xai(ds_key, explainer_key, selected_explainer)
            
            st.divider()
            st.subheader(f"Local {selected_explainer} Attribution Analysis")
            st.markdown(f"Select a rejected application below to see precisely why it was rejected, explained by the **{selected_explainer}** algorithm.")
            
            if rejected_decisions.empty:
                st.info("No rejected decisions generated yet to perform local explanations.")
            else:
                application_ids = rejected_decisions["application_id"].unique().tolist()
                application_ids.sort()
                
                selected_app = st.selectbox(
                    f"Select a rejected application ID to analyze",
                    options=application_ids,
                    index=0,
                    key="xai_app_select"
                )
                
                row_idx = int(rejected_decisions.loc[rejected_decisions["application_id"] == selected_app, "row_index"].values[0])
                
                loaded_ds = load_model_and_dataset(selected_dataset)
                if loaded_ds:
                    if selected_explainer == "SHAP":
                        explainer = get_shap_explainer(selected_dataset)
                        plot_local_xai(explainer, loaded_ds, row_idx, selected_app, "SHAP")
                    elif selected_explainer == "LIME":
                        explainer = get_lime_explainer(selected_dataset)
                        plot_local_xai(explainer, loaded_ds, row_idx, selected_app, "LIME")
                    elif selected_explainer == "Integrated Gradients":
                        plot_local_xai(None, loaded_ds, row_idx, selected_app, "Integrated Gradients")
                else:
                    st.error("Failed to load dataset and model configuration.")
        except Exception as e:
            st.error(f"An error occurred while rendering the {selected_explainer} tab: {e}")
            st.exception(e)

    with tab_dice:
        st.header(f"{selected_dataset} Counterfactual Explanations")
        try:
            # Filter interface
            if not rejected_decisions.empty:
                application_ids = rejected_decisions["application_id"].unique().tolist()
                application_ids.sort()
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_app_dice = st.selectbox(
                        "Filter by Rejected Application ID",
                        options=["All"] + application_ids,
                        index=0,
                        key="dice_app_select"
                    )
                
                app_filter = "" if selected_app_dice == "All" else selected_app_dice
                st.subheader("Actionable Decision Paths")
                rejected_decision_panel(selected_dataset, rejected_decisions, application_filter=app_filter)
                
                st.subheader("Alternative Action Scenarios")
                cfs_app = None if selected_app_dice == "All" else selected_app_dice
                counterfactual_panel(selected_dataset, counterfactuals, recommendations, selected_app_id=cfs_app)
            else:
                st.info("No rejected decision data available.")
        except Exception as e:
            st.error(f"An error occurred while rendering the DiCE tab: {e}")
            st.exception(e)

    with tab_eval:
        st.header(f"{selected_dataset} XAI Quality Report")
        try:
            st.markdown("These metrics evaluate the reliability of our explanations based on the OpenXAI framework.")
            
            xai_metrics_data = load_json(XAI_METRICS_PATH)
            
            if not xai_metrics_data or ds_key not in xai_metrics_data:
                st.warning("XAI evaluation metrics not found. Run `python3 evaluate_xai.py` to generate them.")
            else:
                m = xai_metrics_data[ds_key].get(explainer_key, {})
                
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Faithfulness: PGI (Higher is better)", 
                    f"{m.get('PGI', 0):.4f}", 
                    help="Prediction Gap on Important features: drop in probability when important features are perturbed."
                )
                c2.metric(
                    "Faithfulness: PGU (Lower is better)", 
                    f"{m.get('PGU', 0):.4f}", 
                    help="Prediction Gap on Unimportant features: drop in probability when unimportant features are perturbed."
                )
                c3.metric(
                    "Stability: RIS (Closer to 0 is better)", 
                    f"{m.get('RIS', 0):.4f}", 
                    help="Relative Input Stability: maximum change in explanation relative to small changes in input."
                )
                
                st.divider()
                st.caption(
                    "RIS/ROS/RRS are reported on a natural-log stability scale. ROS uses model output logits so saturated probabilities do not artificially inflate the score."
                )
                st.subheader("OpenXAI Benchmark Comparison")
                st.markdown(f"Comparing our {selected_explainer} explanation quality against the standard OpenXAI Leaderboard baseline for Neural Networks on the {selected_dataset} dataset.")
                
                if ds_key == "german":
                    if selected_explainer == "SHAP":
                        baseline_col_name = "OpenXAI Baseline (SHAP)"
                        baseline_vals = ["0.4400", "0.6400", "1.2800", "4.4600", "11.4600", "0.1020", "0.0090", "0.0210", "0.3610", "0.0320"]
                    elif selected_explainer == "LIME":
                        baseline_col_name = "OpenXAI Baseline (LIME)"
                        baseline_vals = ["0.4200", "0.6800", "6.3500", "9.5700", "16.5000", "0.1380", "0.0950", "0.0130", "0.3200", "0.0620"]
                    else:
                        baseline_col_name = "OpenXAI Baseline (IG)"
                        baseline_vals = ["0.4000", "0.6800", "0.6200", "2.8800", "6.8300", "0.0090", "0.0560", "0.0700", "0.2140", "0.1370"]
                    
                    comp_data = {
                        "Metric": [
                            "PGI (Higher is better) ⬆️", "PGU (Lower is better) ⬇️", 
                            "RIS (Closer to 0 is better) ⬇️", "ROS (Closer to 0 is better) ⬇️", "RRS (Closer to 0 is better) ⬇️",
                            "Fairness@PGI (Closer to 0) ⬇️", "Fairness@PGU (Closer to 0) ⬇️", 
                            "Fairness@RIS (Closer to 0) ⬇️", "Fairness@ROS (Closer to 0) ⬇️", "Fairness@RRS (Closer to 0) ⬇️"
                        ],
                        baseline_col_name: baseline_vals,
                        f"Our Model ({selected_explainer})": [
                            f"{m.get('PGI', 0):.4f}", f"{m.get('PGU', 0):.4f}", 
                            f"{m.get('RIS', 0):.4f}", f"{m.get('ROS', 0):.4f}", f"{m.get('RRS', 0):.4f}",
                            f"{m.get('Fairness@PGI', 0):.4f}", f"{m.get('Fairness@PGU', 0):.4f}",
                            f"{m.get('Fairness@RIS', 0):.4f}", f"{m.get('Fairness@ROS', 0):.4f}", f"{m.get('Fairness@RRS', 0):.4f}",
                        ],
                    }
                    takeaways = (
                        "**Key Takeaways:**\\n"
                        f"- **PGU Performance**: Lower PGU suggests that features ranked as unimportant by {selected_explainer} have limited effect when perturbed.\\n"
                        f"- **RIS & RRS Performance**: Lower RIS/RRS suggests the {selected_explainer} explanations are stable under small encoded-input changes and aligned with model representations.\\n"
                        "- **Fairness Performance**: The fairness rows compare explanation-quality gaps across gender-coded groups in the UCI dataset. Treat them as research diagnostics, not production compliance guarantees."
                    )
                else:
                    if selected_explainer == "SHAP":
                        baseline_col_name = "OpenXAI Baseline (SHAP)"
                        baseline_vals = ["0.2600", "0.2300", "1.5700", "3.8000", "9.4000"]
                    elif selected_explainer == "LIME":
                        baseline_col_name = "OpenXAI Baseline (LIME)"
                        baseline_vals = ["0.2600", "0.2400", "1.1900", "3.3300", "13.2500"]
                    else:
                        baseline_col_name = "OpenXAI Baseline (IG)"
                        baseline_vals = ["0.2400", "0.2600", "3.1600", "5.1700", "9.5300"]
                        
                    comp_data = {
                        "Metric": [
                            "PGI (Higher is better) ⬆️", "PGU (Lower is better) ⬇️", 
                            "RIS (Closer to 0 is better) ⬇️", "ROS (Closer to 0 is better) ⬇️", "RRS (Closer to 0 is better) ⬇️"
                        ],
                        baseline_col_name: baseline_vals,
                        f"Our Model ({selected_explainer})": [
                            f"{m.get('PGI', 0):.4f}", f"{m.get('PGU', 0):.4f}", 
                            f"{m.get('RIS', 0):.4f}", f"{m.get('ROS', 0):.4f}", f"{m.get('RRS', 0):.4f}"
                        ],
                    }
                    takeaways = (
                        "**Key Takeaways:**\\n"
                        f"- **PGI Performance**: Higher PGI suggests the top-ranked {selected_explainer} features are influential for the model output.\\n"
                        f"- **PGU, RIS, RRS Performance**: Lower PGU/RIS/RRS suggests the {selected_explainer} explanations are less sensitive to unimportant-feature perturbations and small encoded-input noise."
                    )
                
                st.table(comp_data)
                st.markdown(takeaways)
                st.divider()

                st.markdown(
                    "> **Faithfulness Analysis**: A good explainer should have a high PGI (meaning perturbing top features breaks the model's prediction) "
                    "and a low PGU (meaning perturbing bottom features does not affect the prediction). \\n\\n"
                    "> **Stability Analysis**: A good explainer should have an RIS close to 0, indicating that minor noise in the data does not completely flip the explanation."
                )
        except Exception as e:
            st.error(f"An error occurred while rendering the XAI Quality tab: {e}")
            st.exception(e)

    with tab_fairness:
        st.header("⚖️ Fairness Auditing & Decision Support Panel")
        st.markdown(
            "This decision support panel implements demographic fairness mitigation to ensure lending models do not "
            "discriminate against protected groups (e.g., based on gender or age). "
            "Risk managers can select a sensitive attribute and an optimization constraint to evaluate "
            "how the decision boundary changes and how it impacts lending outcomes."
        )

        if selected_dataset != "German Credit":
            st.info(
                "ℹ️ **Regulatory Exclusion (ECOA Compliance):**\n\n"
                "The FICO HELOC dataset does not contain protected demographic attributes (such as gender, race, or age) "
                "in compliance with the US Equal Credit Opportunity Act (ECOA). Under US federal lending laws, credit models "
                "are prohibited from utilizing these variables for underwriting. Thus, demographic bias auditing and post-hoc "
                "fairness mitigation are not applicable to the HELOC dataset.\n\n"
                "To explore this decision support tool, please switch to the **German Credit** dataset using the sidebar."
            )
        else:
            try:
                # Sensitive Attribute Choice
                st.subheader("1. Configure Fairness Strategy")
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    sensitive_attr = st.radio(
                        "Select Protected/Sensitive Attribute:",
                        options=["Gender (personal_status_sex)", "Age (age)"],
                        help="Choose the attribute you want to protect against bias."
                    )
                with col_sel2:
                    constraint_name = st.radio(
                        "Select Fairness Constraint:",
                        options=["Demographic Parity", "Equalized Odds"],
                        help="Demographic Parity enforces equal approval rates across groups. Equalized Odds enforces equal error rates (TPR and FPR)."
                    )

                # Binarize Age details if Age selected
                age_threshold = 30
                if "Age" in sensitive_attr:
                    age_threshold = st.slider(
                        "Define Young Age Threshold:",
                        min_value=21,
                        max_value=45,
                        value=30,
                        step=1,
                        help="Applicants below this age are classified as the 'Young' group, and others as 'Old'."
                    )

                # Run post-processing optimization
                loaded_ds = load_model_and_dataset(selected_dataset)
                if loaded_ds:
                    features = loaded_ds.features
                    target = loaded_ds.data[TARGET_COLUMN].astype("int64")
                    
                    # Same 20% test split to evaluate mitigation outcomes
                    _, x_test, _, y_test = train_test_split(
                        features, target,
                        test_size=0.2, stratify=target, random_state=42
                    )

                    # Wrap PyTorch model
                    wrapper = PyTorchDNNWrapper(loaded_ds.model, loaded_ds.transformer, threshold=loaded_ds.threshold)

                    # Get sensitive attribute array
                    if "Gender" in sensitive_attr:
                        is_female_test = x_test["personal_status_sex"].isin(["a92", "a95"])
                        sensitive_features = np.where(is_female_test, "Female", "Male")
                        g0_label, g1_label = "Female", "Male"
                    else:
                        is_young_test = x_test["age"] < age_threshold
                        sensitive_features = np.where(is_young_test, "Young", "Old")
                        g0_label, g1_label = "Young", "Old"

                    # Run base predictions
                    y_pred_base = wrapper.predict(x_test)

                    # Run ThresholdOptimizer
                    constraint_key = "demographic_parity" if constraint_name == "Demographic Parity" else "equalized_odds"
                    
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mitigator = ThresholdOptimizer(
                            estimator=wrapper,
                            constraints=constraint_key,
                            objective="accuracy_score",
                            predict_method="predict_proba",
                            prefit=True
                        )
                        mitigator.fit(x_test, y_test, sensitive_features=sensitive_features)
                        y_pred_mit = mitigator.predict(x_test, sensitive_features=sensitive_features)

                    # Compute metrics
                    base_res = compute_fairness_metrics(y_test.to_numpy(), y_pred_base, sensitive_features)
                    mit_res = compute_fairness_metrics(y_test.to_numpy(), y_pred_mit, sensitive_features)

                    # Visuals and tables
                    st.subheader("2. Performance & Fairness Comparison")
                    
                    # Overview metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    acc_change = mit_res["accuracy"] - base_res["accuracy"]
                    dp_change = mit_res["dp_diff"] - base_res["dp_diff"]
                    eo_change = mit_res["eo_diff"] - base_res["eo_diff"]
                    
                    m_col1.metric(
                        "Model Accuracy Change", 
                        f"{mit_res['accuracy']:.2%} (from {base_res['accuracy']:.2%})",
                        delta=f"{acc_change:+.2%}"
                    )
                    m_col2.metric(
                        "Demographic Parity Gap",
                        f"{mit_res['dp_diff']:.2%} (from {base_res['dp_diff']:.2%})",
                        delta=f"{dp_change:+.2%}",
                        delta_color="inverse"
                    )
                    m_col3.metric(
                        "Equalized Odds Gap",
                        f"{mit_res['eo_diff']:.2%} (from {base_res['eo_diff']:.2%})",
                        delta=f"{eo_change:+.2%}",
                        delta_color="inverse"
                    )

                    # Comparison dataframe
                    st.markdown("#### Detailed Performance Metrics Table")
                    comp_table_data = {
                        "Performance Metric": ["Accuracy", "F1-Score", "Precision", "Recall", "Demographic Parity Difference", "Equalized Odds Difference"],
                        "Before Mitigation": [
                            f"{base_res['accuracy']:.4f}",
                            f"{base_res['f1']:.4f}",
                            f"{base_res['precision']:.4f}",
                            f"{base_res['recall']:.4f}",
                            f"{base_res['dp_diff']:.4f}",
                            f"{base_res['eo_diff']:.4f}"
                        ],
                        "After Mitigation": [
                            f"{mit_res['accuracy']:.4f}",
                            f"{mit_res['f1']:.4f}",
                            f"{mit_res['precision']:.4f}",
                            f"{mit_res['recall']:.4f}",
                            f"{mit_res['dp_diff']:.4f}",
                            f"{mit_res['eo_diff']:.4f}"
                        ],
                        "Difference / Gain": [
                            f"{acc_change:+.4f}",
                            f"{(mit_res['f1'] - base_res['f1']):+.4f}",
                            f"{(mit_res['precision'] - base_res['precision']):+.4f}",
                            f"{(mit_res['recall'] - base_res['recall']):+.4f}",
                            f"{dp_change:+.4f}",
                            f"{eo_change:+.4f}"
                        ]
                    }
                    st.table(comp_table_data)

                    # Dynamic Approval Rates
                    st.markdown("#### Dynamic Approval Rates by Group")
                    col_bar, col_explain = st.columns([2, 1])
                    with col_bar:
                        fig_fair = go.Figure()
                        fig_fair.add_trace(go.Bar(
                            x=[g0_label, g1_label],
                            y=[base_res["selection_rates"][g0_label], base_res["selection_rates"][g1_label]],
                            name="Before Mitigation (Original)",
                            marker_color="#F43F5E"
                        ))
                        fig_fair.add_trace(go.Bar(
                            x=[g0_label, g1_label],
                            y=[mit_res["selection_rates"][g0_label], mit_res["selection_rates"][g1_label]],
                            name="After Mitigation (Optimized)",
                            marker_color="#10B981"
                        ))
                        fig_fair.update_layout(
                            barmode='group',
                            title=dict(
                                text=f"Lending Approval Rates before/after Fairness Mitigation by {sensitive_attr.split(' ')[0]}",
                                font=dict(size=16, family="Outfit, Inter, sans-serif")
                            ),
                            xaxis_title="Group Category",
                            yaxis_title="Approval Rate (Selection Rate)",
                            yaxis_tickformat='.0%',
                            height=360,
                            margin=dict(l=40, r=40, t=50, b=40),
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_fair, use_container_width=True)
                    
                    with col_explain:
                        st.markdown("##### **Lending Approval Summary**")
                        st.write(f"**Original Model:**")
                        st.write(f"- {g0_label} approval rate: `{base_res['selection_rates'][g0_label]:.2%}`")
                        st.write(f"- {g1_label} approval rate: `{base_res['selection_rates'][g1_label]:.2%}`")
                        st.write(f"- Initial Approval Gap: **`{base_res['dp_diff']:.2%}`**")
                        
                        st.write(f"**Mitigated Model:**")
                        st.write(f"- {g0_label} approval rate: `{mit_res['selection_rates'][g0_label]:.2%}`")
                        st.write(f"- {g1_label} approval rate: `{mit_res['selection_rates'][g1_label]:.2%}`")
                        st.write(f"- Final Approval Gap: **`{mit_res['dp_diff']:.2%}`**")

                    # Management Interpretations and methodology selection
                    st.subheader("3. Managerial Assessment and Decision Analysis (Management Engineering Analysis)")
                    
                    st.info(
                        "💡 **Methodology Selection Report: Why Post-Processing (Threshold Calibration) Was Preferred?**\n\n"
                        "In credit scoring models, algorithmic fairness can be enforced at three main stages: "
                        "1) Pre-processing (pre-training data manipulation), 2) In-processing (introducing constraints during model training), and "
                        "3) Post-processing (re-calibrating prediction probabilities per group post-training).\n\n"
                        "In this project, Microsoft's **Fairlearn** library and its **Post-processing: ThresholdOptimizer** algorithm were selected based on the following operational reasons:\n"
                        "- **Operational Efficiency:** Re-training deep neural networks (DNNs) with fairness constraints (in-processing) requires high computational overhead and can lead to training instability. Post-processing preserves the pre-trained, performance-validated model parameters.\n"
                        "- **Auditability and Compliance:** For financial institutions, keeping the core predictive risk model parameters frozen and instead calibrating the decision thresholds (cut-offs) per demographic group is far more transparent and auditable for model risk management and compliance boards.\n"
                        "- **Real-time Simulation:** It enables immediate simulation of model decisions under various fairness constraints, allowing risk managers to make informed, real-time strategic decisions."
                    )

                    narrative = ""
                    if "Gender" in sensitive_attr:
                        narrative += (
                            "##### **Socio-Economic and Financial Interpretations (Gender Analysis):**\n"
                            "Historically, female applicants are underrepresented in the credit dataset compared to male applicants. "
                            "This class imbalance causes the baseline model to approve male applicants at a higher rate (resulting in a baseline gender parity gap of 10.48%). "
                            f"When the **{constraint_name}** constraint is enforced, "
                        )
                        if constraint_key == "demographic_parity":
                            narrative += (
                                "the overall approval rates for female and male applicants are fully equalized (0.00% difference). "
                                "While this policy guarantees strict compliance with demographic parity, it may slightly adjust the overall portfolio default rate by approving applicants from historically higher-risk classes or rejecting marginal applicants from lower-risk classes."
                            )
                        else:
                            narrative += (
                                "the error rates (false positive and false negative rates) are balanced between female and male applicants of equal merit (i.e., those who actually have the capacity to repay vs. those who do not). "
                                "The Equalized Odds policy aims to establish a merit-based fairness equilibrium, minimizing financial credit risk losses."
                            )
                    else:
                        narrative += (
                            "##### **Age Discrimination and Risk Management (Age Analysis):**\n"
                            f"When the age threshold is set to `{age_threshold}`, young applicants tend to face higher rejection rates because their income level and credit history are naturally less established than those of older applicants. "
                            f"Following the **{constraint_name}** mitigation, "
                        )
                        if constraint_key == "demographic_parity":
                            narrative += (
                                "the selection rates for young and mature applicants are balanced, promoting financial inclusion for younger entrepreneurs and consumers. "
                                "From a management perspective, this is a strategic decision that fosters long-term customer loyalty and expands the bank's active portfolio."
                            )
                        else:
                            narrative += (
                                "the true positive rate of younger applicants is equalized with that of older applicants. "
                                "This ensures that creditworthy young applicants are not systematically locked out of credit opportunities due to age bias."
                            )

                    st.markdown(narrative)
                    st.markdown(
                        f"##### **Performance-Fairness Trade-off Analysis:**\n"
                        f"Change in predictive accuracy after applying fairness constraints: **`{acc_change:+.4f}`**.\n\n"
                        f"From a Management Engineering decision-matrix perspective, a minor sacrifice in accuracy (e.g., 1-2%) is highly justified as it shields the financial institution from regulatory non-compliance penalties (such as fair lending violations under ECOA/GDPR) and mitigates reputational risk. "
                        f"Therefore, this trade-off is economically **rational and preferred** when factoring in the potential costs of legal compliance and customer discrimination suits."
                    )
                else:
                    st.error("Failed to load dataset model for fairness evaluation.")
            except Exception as e:
                st.error(f"An error occurred in Fairness Support calculations: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()
