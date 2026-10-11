"""
MediPredict AI - Interactive Visualization Module
Author: MediPredict AI Team
Description: Polished Plotly visualization suite for clinical metrics, ROC/PR curves,
             confusion matrices, feature importance, calibration, and unsupervised PCA projections.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Theme Colors (Healthcare Analytics palette: Navy, Teal, Slate, Crimson)
PRIMARY_COLOR = "#0f766e"      # Deep Teal
SECONDARY_COLOR = "#1e3a8a"    # Navy Blue
ACCENT_COLOR = "#0284c7"       # Sky Blue
DANGER_COLOR = "#dc2626"       # Crimson
SUCCESS_COLOR = "#16a34a"      # Emerald Green
BG_COLOR = "#ffffff"
GRID_COLOR = "#f1f5f9"
TEXT_COLOR = "#0f172a"


def plot_roc_curves(evaluation_results: Dict[str, Any]) -> go.Figure:
    """Creates interactive ROC curves for all evaluated models."""
    fig = go.Figure()

    # Diagonal reference line (Random Guess)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier (AUC = 0.50)",
        line=dict(color="#94a3b8", dash="dash", width=1.5)
    ))

    colors = ["#0f766e", "#2563eb", "#7c3aed", "#d97706", "#059669"]

    for i, (name, metrics) in enumerate(evaluation_results.items()):
        roc_data = metrics.get("roc_curve", {})
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        auc = metrics.get("roc_auc", 0.0)

        if fpr and tpr:
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC = {auc})",
                line=dict(color=colors[i % len(colors)], width=2.5)
            ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curves",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        template="plotly_white",
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        font=dict(family="Arial, sans-serif", color=TEXT_COLOR)
    )
    return fig


def plot_pr_curves(evaluation_results: Dict[str, Any]) -> go.Figure:
    """Creates interactive Precision-Recall curves for all models."""
    fig = go.Figure()
    colors = ["#0f766e", "#2563eb", "#7c3aed", "#d97706", "#059669"]

    for i, (name, metrics) in enumerate(evaluation_results.items()):
        pr_data = metrics.get("pr_curve", {})
        p = pr_data.get("precision", [])
        r = pr_data.get("recall", [])
        pr_auc = metrics.get("pr_auc", 0.0)

        if p and r:
            fig.add_trace(go.Scatter(
                x=r,
                y=p,
                mode="lines",
                name=f"{name} (PR-AUC = {pr_auc})",
                line=dict(color=colors[i % len(colors)], width=2.5)
            ))

    fig.update_layout(
        title="Precision-Recall Curves (Clinically Critical for Class Imbalance)",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
        template="plotly_white",
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02),
        font=dict(family="Arial, sans-serif", color=TEXT_COLOR)
    )
    return fig


def plot_confusion_matrix(cm_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Generates an annotated 2x2 confusion matrix heatmap."""
    tn = cm_data.get("tn", 0)
    fp = cm_data.get("fp", 0)
    fn = cm_data.get("fn", 0)
    tp = cm_data.get("tp", 0)

    z = [[tn, fp], [fn, tp]]
    x = ["Predicted Negative (0)", "Predicted Positive (1)"]
    y = ["Actual Negative (0)", "Actual Positive (1)"]

    annotations = [
        [f"True Negative (TN)<br><b>{tn}</b>", f"False Positive (FP)<br><b>{fp}</b>"],
        [f"False Negative (FN)<br><b>{fn}</b><br>(Critical Miss)", f"True Positive (TP)<br><b>{tp}</b>"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, "#e0f2fe"], [0.5, "#38bdf8"], [1, "#0284c7"]],
        showscale=False
    ))

    # Add text annotations
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=x[j],
                y=y[i],
                text=annotations[i][j],
                showarrow=False,
                font=dict(color="#0f172a", size=14)
            )

    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Ground Truth Label",
        template="plotly_white",
        width=520,
        height=450,
        font=dict(color=TEXT_COLOR)
    )
    return fig


def plot_feature_importance_bar(importance_data: Dict[str, Any], title: str = "Feature Importance") -> go.Figure:
    """Generates horizontal bar chart with error bars for permutation importance."""
    features = list(importance_data.keys())
    means = [importance_data[f]["importance_mean"] for f in features]
    stds = [importance_data[f]["importance_std"] for f in features]

    fig = go.Figure(go.Bar(
        x=means,
        y=features,
        orientation="h",
        error_x=dict(type="data", array=stds, visible=True),
        marker=dict(color=PRIMARY_COLOR)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean ROC-AUC Score Decrease Upon Permutation",
        yaxis_title="Biomarker / Clinical Feature",
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
        height=450,
        font=dict(color=TEXT_COLOR)
    )
    return fig


def plot_logistic_regression_coefficients(coeff_data: Dict[str, Any]) -> go.Figure:
    """Plots Logistic Regression coefficients / odds ratios."""
    features = list(coeff_data.keys())
    odds_ratios = [coeff_data[f]["odds_ratio"] for f in features]
    coefs = [coeff_data[f]["coefficient"] for f in features]

    # Color by whether it elevates odds (OR > 1) or decreases odds (OR < 1)
    bar_colors = [DANGER_COLOR if c > 0 else PRIMARY_COLOR for c in coefs]

    fig = go.Figure(go.Bar(
        x=coefs,
        y=features,
        orientation="h",
        marker=dict(color=bar_colors),
        customdata=odds_ratios,
        hovertemplate="Feature: %{y}<br>Coefficient: %{x:.3f}<br>Odds Ratio: %{customdata:.3f}<extra></extra>"
    ))

    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#64748b")

    fig.update_layout(
        title="Logistic Regression Standardized Feature Weights (Log-Odds Impact)",
        xaxis_title="Standardized Log-Odds Coefficient (Positive = Risk Elevating)",
        yaxis_title="Feature",
        template="plotly_white",
        height=450,
        font=dict(color=TEXT_COLOR)
    )
    return fig


def plot_calibration_curve_chart(calibration_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Plots probability calibration curve vs ideal diagonal."""
    prob_true = calibration_data.get("prob_true", [])
    prob_pred = calibration_data.get("prob_pred", [])
    brier = calibration_data.get("brier_score", 0.0)

    fig = go.Figure()

    # Perfect calibration reference
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(color="#94a3b8", dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode="lines+markers",
        name=f"{model_name} (Brier: {brier})",
        line=dict(color=PRIMARY_COLOR, width=2.5),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f"Probability Calibration Curve: {model_name}",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives (Empirical Probability)",
        template="plotly_white",
        font=dict(color=TEXT_COLOR)
    )
    return fig


def plot_learning_curve_chart(lc_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Plots learning curve showing training vs validation ROC-AUC score across sample sizes."""
    sizes = lc_data.get("train_sizes", [])
    train_mean = lc_data.get("train_scores_mean", [])
    val_mean = lc_data.get("val_scores_mean", [])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sizes,
        y=train_mean,
        mode="lines+markers",
        name="Training Score",
        line=dict(color="#2563eb", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=sizes,
        y=val_mean,
        mode="lines+markers",
        name="Cross-Validation Score",
        line=dict(color=PRIMARY_COLOR, width=2)
    ))

    fig.update_layout(
        title=f"Learning Curve: {model_name} (Bias-Variance Diagnosis)",
        xaxis_title="Number of Training Instances",
        yaxis_title="ROC-AUC Score",
        template="plotly_white",
        font=dict(color=TEXT_COLOR)
    )
    return fig


def plot_kmeans_elbow_silhouette(kmeans_data: Dict[str, Any]) -> go.Figure:
    """Plots Inertia (Elbow method) and Silhouette Score across values of k."""
    k_vals = kmeans_data.get("k_values", [])
    inertias = kmeans_data.get("inertias", [])
    silhouettes = kmeans_data.get("silhouette_scores", [])

    fig = go.Figure()

    # Inertia line
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=inertias,
        mode="lines+markers",
        name="Inertia (Sum of Squared Distances)",
        line=dict(color="#2563eb", width=2.5),
        yaxis="y1"
    ))

    # Silhouette line
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=silhouettes,
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color=PRIMARY_COLOR, width=2.5, dash="dot"),
        yaxis="y2"
    ))

    fig.update_layout(
        title="K-Means Cluster Optimization: Elbow Plot & Silhouette Analysis",
        xaxis_title="Number of Clusters (k)",
        yaxis=dict(title="Inertia", titlefont=dict(color="#2563eb"), tickfont=dict(color="#2563eb")),
        yaxis2=dict(
            title="Silhouette Score",
            titlefont=dict(color=PRIMARY_COLOR),
            tickfont=dict(color=PRIMARY_COLOR),
            overlaying="y",
            side="right"
        ),
        template="plotly_white",
        legend=dict(x=0.5, y=1.1, orientation="h"),
        font=dict(color=TEXT_COLOR)
    )
    return fig


def plot_pca_clusters(pca_records: List[Dict[str, Any]]) -> go.Figure:
    """Plots 2D PCA projection colored by K-Means cluster grouping."""
    df_pca = pd.DataFrame(pca_records)

    fig = px.scatter(
        df_pca,
        x="PCA_1",
        y="PCA_2",
        color="Cluster",
        color_discrete_sequence=["#0f766e", "#f59e0b", "#3b82f6", "#ef4444"],
        title="2D PCA Projection of Patients (Unsupervised Cluster Separation)",
        labels={"PCA_1": "Principal Component 1", "PCA_2": "Principal Component 2"}
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR)
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plots correlation heatmap among numeric features."""
    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Tealgrn",
        title="Feature Pearson Correlation Heatmap"
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(color=TEXT_COLOR)
    )
    return fig
