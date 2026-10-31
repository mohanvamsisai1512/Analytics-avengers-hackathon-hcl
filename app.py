# app.py
# Customer Churn Intelligence Dashboard
# Run: streamlit run app.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve,
    precision_recall_curve, brier_score_loss
)

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- Styling ---------------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e7ebf2;
        padding: 14px 16px; border-radius: 12px;
    }
    div[data-testid="stMetric"] label {font-size: 0.88rem;}
    .section-note {color: #667085; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Intelligence Dashboard")
st.caption("Churn prediction • Risk prioritization • Retention intelligence")

# ----------------------------- Helpers ---------------------------------
def load_csv(uploaded_file, default_path):
    """Prefer an uploaded CSV; otherwise load the local exported file if present."""
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")
            return None
    if os.path.exists(default_path):
        try:
            return pd.read_csv(default_path)
        except Exception as exc:
            st.warning(f"Could not read {default_path}: {exc}")
    return None


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def normalize_probability(series):
    values = pd.to_numeric(series, errors="coerce")
    # If probabilities were exported on a 0-100 scale, convert to 0-1.
    if values.notna().any() and values.max() > 1 and values.max() <= 100:
        values = values / 100
    return values.clip(0, 1)


def metric_card_values(df, probability_col, target_col):
    p = normalize_probability(df[probability_col])
    valid = p.notna()
    d = df.loc[valid].copy()
    d["_prob"] = p.loc[valid]
    result = {
        "customers": len(d),
        "mean_risk": float(d["_prob"].mean()) if len(d) else np.nan,
        "actual_churn_rate": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": np.nan
    }
    if target_col:
        y = pd.to_numeric(d[target_col], errors="coerce")
        mask = y.notna()
        y = y[mask].astype(int)
        probs = d.loc[mask, "_prob"]
        if len(y):
            result["actual_churn_rate"] = float(y.mean())
            result["brier"] = float(brier_score_loss(y, probs))
            if y.nunique() == 2:
                result["roc_auc"] = float(roc_auc_score(y, probs))
                result["pr_auc"] = float(average_precision_score(y, probs))
    return result


def top_k_table(df, probability_col, target_col, customer_col):
    ranked = df.copy()
    ranked["_prob"] = normalize_probability(ranked[probability_col])
    ranked = ranked.dropna(subset=["_prob"]).sort_values("_prob", ascending=False)
    rows = []
    total_churners = None
    if target_col:
        y = pd.to_numeric(ranked[target_col], errors="coerce")
        if y.notna().any():
            total_churners = int(y.fillna(0).sum())

    for pct in [5, 10, 20, 30, 50]:
        n = max(1, int(np.ceil(len(ranked) * pct / 100))) if len(ranked) else 0
        selected = ranked.head(n)
        row = {"Top customer %": f"{pct}%", "Customers flagged": n}
        if target_col and total_churners is not None:
            y_selected = pd.to_numeric(selected[target_col], errors="coerce").fillna(0).astype(int)
            captured = int(y_selected.sum())
            precision = safe_rate(captured, n)
            recall = safe_rate(captured, total_churners)
            prevalence = safe_rate(total_churners, len(ranked))
            row.update({
                "Churners captured": captured,
                "Precision@K": precision,
                "Recall@K": recall,
                "Lift": safe_rate(precision, prevalence)
            })
        rows.append(row)
    return pd.DataFrame(rows), ranked


# ----------------------------- Sidebar ---------------------------------
st.sidebar.header("Dashboard data")
st.sidebar.caption(
    "Upload the CSV exports from your notebook, or keep them beside app.py "
    "using the default filenames below."
)

queue_upload = st.sidebar.file_uploader(
    "Customer prediction / retention queue CSV",
    type=["csv"],
    key="queue"
)
segment_upload = st.sidebar.file_uploader(
    "Segment performance CSV (optional)",
    type=["csv"],
    key="segments"
)
shap_upload = st.sidebar.file_uploader(
    "SHAP feature importance CSV (optional)",
    type=["csv"],
    key="shap"
)
threshold_upload = st.sidebar.file_uploader(
    "Threshold analysis CSV (optional)",
    type=["csv"],
    key="threshold"
)

queue_df = load_csv(queue_upload, "retention_queue.csv")
if queue_df is None:
    queue_df = load_csv(None, "customer_retention_queue.csv")
segment_df = load_csv(segment_upload, "segmented_performance.csv")
shap_df = load_csv(shap_upload, "shap_feature_importance.csv")
threshold_df = load_csv(threshold_upload, "threshold_analysis.csv")

if queue_df is None or queue_df.empty:
    st.info(
        "Upload your customer prediction/retention queue CSV to begin. "
        "It should contain a churn probability column, and ideally the actual churn label."
    )
    st.markdown("**Expected key columns**")
    st.code(
        "predicted_churn_probability\n"
        "actual_churn_30d  # optional, for evaluation metrics\n"
        "customer_id       # optional, for customer lookup"
    )
    st.stop()

# ----------------------------- Detect columns --------------------------
prob_col = first_existing_column(
    queue_df,
    ["predicted_churn_probability", "churn_probability", "churn_prob",
     "probability", "prediction_probability", "risk_score"]
)
target_col = first_existing_column(
    queue_df,
    ["actual_churn_30d", "churn_30d", "churn", "is_churn", "target",
     "actual_churn", "Exited"]
)
customer_col = first_existing_column(
    queue_df,
    ["customer_id", "customerID", "CustomerID", "customerId", "id", "ID"]
)

if prob_col is None:
    st.error(
        "I couldn't identify the churn probability column. Rename it to "
        "'predicted_churn_probability' or update the candidate list in app.py."
    )
    st.write("Columns found:", list(queue_df.columns))
    st.stop()

data = queue_df.copy()
data[prob_col] = normalize_probability(data[prob_col])
data = data.dropna(subset=[prob_col]).copy()
if target_col:
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

# ----------------------------- Navigation ------------------------------
pages = [
    "Overview",
    "Risk & Threshold Analysis",
    "Retention Priority Queue",
    "Segment Performance",
    "Model Explainability",
    "Customer Explorer"
]
page = st.sidebar.radio("Navigate", pages)

# Shared threshold control
st.sidebar.divider()
threshold = st.sidebar.slider(
    "Churn-risk threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help="Customers with predicted churn probability at or above this value are flagged."
)
data["risk_flag"] = data[prob_col] >= threshold
data["risk_band"] = pd.cut(
    data[prob_col],
    bins=[-0.001, 0.20, 0.50, 1.001],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

n_customers = len(data)
n_flagged = int(data["risk_flag"].sum())
flagged_share = safe_rate(n_flagged, n_customers)

# ----------------------------- Overview --------------------------------
if page == "Overview":
    st.subheader("Executive overview")
    stats = metric_card_values(data, prob_col, target_col)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers evaluated", f"{n_customers:,}")
    c2.metric("Flagged customers", f"{n_flagged:,}", f"{flagged_share:.1%} of portfolio")
    c3.metric("Average predicted risk", f"{stats['mean_risk']:.1%}" if pd.notna(stats["mean_risk"]) else "N/A")
    if pd.notna(stats["actual_churn_rate"]):
        c4.metric("Actual churn rate", f"{stats['actual_churn_rate']:.1%}")
    else:
        c4.metric("Actual churn rate", "Not supplied")

    if target_col:
        m1, m2, m3 = st.columns(3)
        m1.metric("PR-AUC / Average precision", f"{stats['pr_auc']:.3f}" if pd.notna(stats["pr_auc"]) else "N/A")
        m2.metric("ROC-AUC", f"{stats['roc_auc']:.3f}" if pd.notna(stats["roc_auc"]) else "N/A")
        m3.metric("Brier score", f"{stats['brier']:.3f}" if pd.notna(stats["brier"]) else "N/A")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            data, x=prob_col, nbins=25,
            title="Distribution of predicted churn probabilities",
            labels={prob_col: "Predicted churn probability", "count": "Customers"}
        )
        fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"Threshold {threshold:.2f}")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        band_counts = data["risk_band"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).reset_index()
        band_counts.columns = ["Risk band", "Customers"]
        fig = px.pie(
            band_counts, names="Risk band", values="Customers",
            title="Customer distribution by risk band", hole=0.48
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Current threshold summary")
    st.write(
        f"At a threshold of **{threshold:.2f}**, **{n_flagged:,} customers "
        f"({flagged_share:.1%})** are included in the retention queue."
    )
    st.caption("Risk bands are descriptive dashboard bands; the intervention threshold is controlled separately.")

# ----------------------- Risk & Threshold Analysis ---------------------
elif page == "Risk & Threshold Analysis":
    st.subheader("Risk and threshold analysis")
    st.write(
        "Adjust the threshold in the sidebar. A lower threshold flags more customers "
        "and generally increases recall; a higher threshold reduces the queue and generally increases precision."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected threshold", f"{threshold:.2f}")
    c2.metric("Customers flagged", f"{n_flagged:,}")
    c3.metric("Flagged portfolio share", f"{flagged_share:.1%}")

    if target_col:
        y_true = data[target_col]
        valid = y_true.notna()
        y_true = y_true[valid].astype(int)
        y_pred = (data.loc[valid, prob_col] >= threshold).astype(int)
        if len(y_true) and y_true.nunique() == 2:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            a, b, c = st.columns(3)
            a.metric("Precision", f"{precision:.3f}")
            b.metric("Recall", f"{recall:.3f}")
            c.metric("F1-score", f"{f1:.3f}")

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual non-churn", "Actual churn"],
                columns=["Predicted non-churn", "Predicted churn"]
            )
            st.markdown("#### Confusion matrix")
            st.dataframe(cm_df, use_container_width=True)

            thresholds = np.arange(0.10, 0.91, 0.05)
            rows = []
            for t in thresholds:
                yp = (data.loc[valid, prob_col] >= t).astype(int)
                rows.append({
                    "Threshold": t,
                    "Flagged customers": int(yp.sum()),
                    "Precision": precision_score(y_true, yp, zero_division=0),
                    "Recall": recall_score(y_true, yp, zero_division=0),
                    "F1": f1_score(y_true, yp, zero_division=0)
                })
            sweep = pd.DataFrame(rows)
            fig = go.Figure()
            for metric in ["Precision", "Recall", "F1"]:
                fig.add_trace(go.Scatter(
                    x=sweep["Threshold"], y=sweep[metric],
                    mode="lines+markers", name=metric
                ))
            fig.update_layout(
                title="Precision, recall and F1 across thresholds",
                xaxis_title="Probability threshold",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Threshold trade-off table")
            st.dataframe(sweep.round(3), use_container_width=True)
            st.download_button(
                "Download threshold analysis",
                sweep.to_csv(index=False).encode("utf-8"),
                "dashboard_threshold_analysis.csv",
                "text/csv"
            )
        else:
            st.warning("The actual churn column must contain both 0 and 1 labels to calculate classification metrics.")
    else:
        st.info("Upload a queue CSV containing the actual churn label to calculate precision, recall, F1 and confusion matrix.")

# ----------------------- Retention Priority Queue ----------------------
elif page == "Retention Priority Queue":
    st.subheader("Retention prioritization")
    st.write("Customers are ranked from highest to lowest predicted churn probability.")

    capacity_table, ranked = top_k_table(data, prob_col, target_col, customer_col)
    if target_col:
        st.markdown("#### Capacity and lift analysis")
        st.dataframe(capacity_table.round(3), use_container_width=True)
        if "Lift" in capacity_table.columns:
            fig = px.bar(
                capacity_table, x="Top customer %", y="Lift",
                title="Lift by retention-team capacity"
            )
            fig.add_hline(y=1, line_dash="dash", annotation_text="Random selection baseline")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Actual churn labels are not present. The ranked queue is available, but precision, recall and lift cannot be calculated.")

    st.markdown("#### Customers flagged at the selected threshold")
    queue = ranked[ranked["_prob"] >= threshold].copy()
    queue.insert(0, "Priority rank", np.arange(1, len(queue) + 1))
    queue["Predicted risk"] = queue["_prob"].map(lambda x: f"{x:.1%}")

    preferred = []
    for col in [customer_col, "customer_segment", "plan_tier", "contract_type", "region",
                "tenure_days", "customer_duration_days", "usage_per_active_day",
                "engagement_decline_score", "payment_failure_rate", "unresolved_ticket_ratio"]:
        if col and col in queue.columns and col not in preferred:
            preferred.append(col)
    display_cols = ["Priority rank"] + preferred + ["Predicted risk"]
    if target_col and target_col in queue.columns:
        display_cols.append(target_col)
    st.dataframe(queue[display_cols], use_container_width=True, height=430)

    st.download_button(
        "Download prioritized retention queue",
        queue.to_csv(index=False).encode("utf-8"),
        "prioritized_retention_queue.csv",
        "text/csv"
    )

# ----------------------- Segment Performance ---------------------------
elif page == "Segment Performance":
    st.subheader("Segment-level performance")
    if segment_df is None or segment_df.empty:
        st.info("Upload segmented_performance.csv to view segment-level model performance.")
    else:
        required = {"segment_type", "segment_value"}
        if not required.issubset(segment_df.columns):
            st.error("Segment CSV should contain 'segment_type' and 'segment_value' columns.")
        else:
            segment_types = sorted(segment_df["segment_type"].dropna().unique().tolist())
            chosen_segment = st.selectbox("Choose segment dimension", segment_types)
            view = segment_df[segment_df["segment_type"] == chosen_segment].copy()
            st.dataframe(view, use_container_width=True)

            metric_options = [
                c for c in ["roc_auc", "pr_auc", "precision", "recall", "f1", "churn_rate",
                            "avg_predicted_probability"]
                if c in view.columns
            ]
            if metric_options:
                selected_metric = st.selectbox("Compare segment metric", metric_options)
                fig = px.bar(
                    view, x="segment_value", y=selected_metric,
                    text_auto=".3f",
                    title=f"{selected_metric.replace('_', ' ').title()} by {chosen_segment}"
                )
                fig.update_layout(xaxis_title=chosen_segment, yaxis_title=selected_metric)
                st.plotly_chart(fig, use_container_width=True)

            if "n" in view.columns:
                st.caption("Always interpret segment metrics alongside sample size (n); small groups can produce unstable estimates.")

# ----------------------- Model Explainability --------------------------
elif page == "Model Explainability":
    st.subheader("Model explainability — SHAP")
    if shap_df is None or shap_df.empty:
        st.info("Upload shap_feature_importance.csv exported from your notebook.")
    else:
        feature_col = first_existing_column(
            shap_df, ["feature", "Feature", "feature_name", "Variable"]
        )
        importance_col = first_existing_column(
            shap_df, ["mean_abs_shap", "mean_abs_shap_value", "importance",
                      "shap_importance", "Mean SHAP", "mean_abs"]
        )
        if feature_col is None or importance_col is None:
            st.error(
                "SHAP CSV needs a feature-name column and an importance column. "
                "Recommended names: feature and mean_abs_shap."
            )
            st.write("Columns found:", list(shap_df.columns))
        else:
            shap_view = shap_df[[feature_col, importance_col]].copy()
            shap_view[importance_col] = pd.to_numeric(shap_view[importance_col], errors="coerce")
            shap_view = shap_view.dropna().sort_values(importance_col, ascending=False).head(20)
            fig = px.bar(
                shap_view.sort_values(importance_col),
                x=importance_col, y=feature_col, orientation="h",
                title="Top global churn drivers (mean absolute SHAP value)"
            )
            fig.update_layout(
                xaxis_title="Mean |SHAP value|",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "SHAP values explain how the fitted model uses features; they show model associations, "
                "not proof that a feature causes churn."
            )
            st.dataframe(shap_view, use_container_width=True)

# ----------------------------- Customer Explorer ------------------------
elif page == "Customer Explorer":
    st.subheader("Customer-level risk explorer")
    if customer_col is None:
        st.warning("No customer identifier column was detected. Add a customer_id column to enable lookup.")
    else:
        customer_ids = data[customer_col].astype(str).dropna().unique().tolist()
        selected_id = st.selectbox("Search/select customer ID", customer_ids)
        customer = data[data[customer_col].astype(str) == selected_id].iloc[0]

        probability = float(customer[prob_col])
        risk_label = "High" if probability >= 0.50 else ("Medium" if probability >= 0.20 else "Low")
        a, b, c = st.columns(3)
        a.metric("Customer ID", str(selected_id))
        b.metric("Predicted churn probability", f"{probability:.1%}")
        c.metric("Risk band", risk_label)

        if target_col and pd.notna(customer[target_col]):
            st.metric("Actual churn outcome", "Churned" if int(customer[target_col]) == 1 else "Did not churn")

        st.markdown("#### Available customer attributes")
        excluded = {prob_col, target_col, "risk_flag", "risk_band"}
        attributes = customer.drop(labels=[x for x in excluded if x and x in customer.index])
        attributes_df = attributes.rename("Value").to_frame()
        st.dataframe(attributes_df, use_container_width=True)

# ----------------------------- Footer -----------------------------------
st.sidebar.divider()
st.sidebar.caption("Dashboard uses exported model predictions and analysis results. It does not retrain the model.")
st.caption(
    "Prototype note: validate the prediction-time availability of every feature and use a genuinely "
    "time-based holdout before presenting results as future-period churn performance."
)
