import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os

st.set_page_config(page_title="Customer Churn & Retention Intelligence", page_icon="📊", layout="wide")
st.title("Customer Churn & Retention Intelligence")
st.caption("Churn-risk scoring • Explainability • Retention prioritization • Model evaluation")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def read_csv(name):
    p=os.path.join(DATA_DIR,name)
    return pd.read_csv(p) if os.path.exists(p) else None

def read_json(name):
    p=os.path.join(DATA_DIR,name)
    if os.path.exists(p):
        with open(p,encoding="utf-8") as f: return json.load(f)
    return {}

queue=read_csv("customer_risk_scores_all.csv")
val=read_csv("validation_predictions.csv")
segments=read_csv("segmented_performance.csv")
shap_df=read_csv("shap_feature_importance.csv")
thresholds=read_csv("threshold_analysis.csv")
capacity=read_csv("capacity_lift.csv")
cust_exp=read_csv("customer_explanations_validation.csv")
metadata=read_json("model_metadata.json")

if queue is None:
    st.error("Missing customer_risk_scores_all.csv. Extract the final artifact ZIP into the same folder as app.py.")
    st.stop()

prob_col="predicted_churn_probability"
target_col="actual_churn_30d"
customer_col="customer_id" if "customer_id" in queue.columns else None
queue[prob_col]=pd.to_numeric(queue[prob_col],errors="coerce").clip(0,1)
queue=queue.dropna(subset=[prob_col]).copy()

st.sidebar.header("Navigation")
page=st.sidebar.radio("Dashboard sections",[
    "Executive Overview","Risk & Intervention Threshold","Retention Priority Queue",
    "Capacity & Lift","Segment Performance","Model Explainability",
    "Customer 360","Methodology & Limitations"
])
st.sidebar.divider()
threshold=st.sidebar.slider("Intervention threshold",0.05,0.95,0.50,0.01)
queue["flagged"]=queue[prob_col]>=threshold
queue["risk_band"]=pd.cut(queue[prob_col],[-.001,.2,.5,1.001],labels=["Low","Medium","High"],include_lowest=True).astype(str)
n=len(queue); flagged=int(queue.flagged.sum())
def pct(x): return f"{x:.1%}"
def fmt(x): return f"{x:.3f}" if pd.notna(x) else "N/A"

if page=="Executive Overview":
    st.subheader("Portfolio snapshot")
    a,b,c,d=st.columns(4)
    a.metric("Customers scored",f"{n:,}")
    b.metric("Customers flagged",f"{flagged:,}",f"{flagged/n:.1%} of scored portfolio")
    c.metric("Average predicted churn risk",pct(queue[prob_col].mean()))
    cval=metadata.get("validation_metrics",{})
    d.metric("Validation PR-AUC",fmt(cval.get("PR_AUC")))
    e,f,g,h=st.columns(4)
    e.metric("Validation ROC-AUC",fmt(cval.get("ROC_AUC")))
    f.metric("Validation precision @ 0.5",fmt(cval.get("precision_at_0_5")))
    g.metric("Validation recall @ 0.5",fmt(cval.get("recall_at_0_5")))
    h.metric("Validation F1 @ 0.5",fmt(cval.get("F1_at_0_5")))
    left,right=st.columns(2)
    with left:
        fig=px.histogram(queue,x=prob_col,nbins=30,title="Predicted churn-risk distribution")
        fig.add_vline(x=threshold,line_dash="dash",annotation_text=f"Threshold {threshold:.2f}")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        counts=queue.risk_band.value_counts().reindex(["Low","Medium","High"],fill_value=0).rename_axis("Risk band").reset_index(name="Customers")
        st.plotly_chart(px.pie(counts,names="Risk band",values="Customers",hole=.45,title="Portfolio risk mix"),use_container_width=True)
    st.info("Portfolio counts and risk scores below are for all eligible rows in the supplied dataset. Model-quality metrics above are calculated separately on the held-out validation cohort.")
    st.markdown("**Current intervention setting**")
    st.write(f"At threshold **{threshold:.2f}**, **{flagged:,}** of **{n:,}** customers are flagged for retention review.")

elif page=="Risk & Intervention Threshold":
    st.subheader("Configurable intervention threshold")
    st.write("Change the threshold in the sidebar to review the flagged customer volume. Classification error metrics are shown only for the held-out validation set.")
    a,b,c=st.columns(3)
    a.metric("Threshold",f"{threshold:.2f}")
    b.metric("Flagged customers (full portfolio)",f"{flagged:,}")
    c.metric("Flagged share",pct(flagged/n))
    if val is not None:
        y=pd.to_numeric(val[target_col],errors="coerce").fillna(0).astype(int)
        p=pd.to_numeric(val[prob_col],errors="coerce").clip(0,1)
        from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
        yp=(p>=threshold).astype(int)
        x,yc,z=st.columns(3)
        x.metric("Validation precision",fmt(precision_score(y,yp,zero_division=0)))
        yc.metric("Validation recall",fmt(recall_score(y,yp,zero_division=0)))
        z.metric("Validation F1",fmt(f1_score(y,yp,zero_division=0)))
        cm=confusion_matrix(y,yp,labels=[0,1])
        st.markdown("**Validation confusion matrix**")
        st.dataframe(pd.DataFrame(cm,index=["Actual non-churn","Actual churn"],columns=["Predicted non-churn","Predicted churn"]),use_container_width=True)
        if thresholds is not None:
            st.markdown("**Validation threshold trade-off**")
            fig=go.Figure()
            for col in ["precision","recall","f1"]:
                fig.add_trace(go.Scatter(x=thresholds.threshold,y=thresholds[col],mode="lines+markers",name=col.title()))
            fig.update_layout(xaxis_title="Threshold",yaxis_title="Score",yaxis_range=[0,1])
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(thresholds.round(3),use_container_width=True)
    else:
        st.warning("validation_predictions.csv is missing; validation errors cannot be calculated.")

elif page=="Retention Priority Queue":
    st.subheader("Prioritized retention review queue")
    st.caption("All eligible customers are ranked by predicted churn probability. This is a review aid, not an automatic retention decision.")
    q=queue[queue.flagged].sort_values(prob_col,ascending=False).copy()
    q.insert(0,"Priority rank",np.arange(1,len(q)+1))
    q["Predicted risk"]=q[prob_col].map(pct)
    cols=["Priority rank"]
    for c in [customer_col,"customer_segment","plan_tier","contract_type","region","tenure_days",
              "usage_change_pct_30d","failed_payments_last_6m","unresolved_tickets","days_to_renewal",
              "support_tickets_last_90d","avg_monthly_usage_hours","login_days_last_30d"]:
        if c and c in q.columns: cols.append(c)
    cols += ["Predicted risk","risk_band"]
    st.metric("Customers in current queue",f"{len(q):,}")
    st.dataframe(q[cols],use_container_width=True,height=520)
    st.download_button("Download current retention queue",q.to_csv(index=False).encode(), "retention_priority_queue.csv","text/csv")

elif page=="Capacity & Lift":
    st.subheader("Retention-team capacity and lift")
    if capacity is None:
        st.warning("capacity_lift.csv missing.")
    else:
        st.write("These figures use validation predictions only, so top-k capture is an evaluation of ranking performance, not a claim about future customers.")
        st.dataframe(capacity.round(3),use_container_width=True)
        fig=px.bar(capacity,x=capacity.top_k_pct.astype(str),y="lift",text="lift",title="Lift at different intervention capacities")
        fig.add_hline(y=1,line_dash="dash",annotation_text="Random selection baseline")
        st.plotly_chart(fig,use_container_width=True)
        fig2=go.Figure()
        fig2.add_trace(go.Bar(x=capacity.top_k_pct.astype(str),y=capacity.precision_at_k,name="Precision@K"))
        fig2.add_trace(go.Scatter(x=capacity.top_k_pct.astype(str),y=capacity.recall_at_k,name="Recall@K",mode="lines+markers"))
        fig2.update_layout(title="Precision and churn capture by capacity",xaxis_title="Top percentage of customers")
        st.plotly_chart(fig2,use_container_width=True)

elif page=="Segment Performance":
    st.subheader("Segmented validation performance")
    if segments is None:
        st.warning("segmented_performance.csv missing.")
    else:
        dimension=st.selectbox("Segment dimension",sorted(segments.segment_type.unique()))
        s=segments[segments.segment_type==dimension].copy()
        st.dataframe(s.round(3),use_container_width=True)
        metric=st.selectbox("Metric", [c for c in ["roc_auc","churn_rate","precision","recall","f1","avg_predicted_probability"] if c in s.columns])
        st.plotly_chart(px.bar(s,x="segment_value",y=metric,text_auto=".3f",title=f"{metric.replace('_',' ').title()} by {dimension}"),use_container_width=True)
        st.caption("Always interpret metrics with sample size. Small segments may have unstable estimates.")

elif page=="Model Explainability":
    st.subheader("Portfolio-level churn risk drivers")
    if shap_df is None:
        st.warning("shap_feature_importance.csv missing.")
    else:
        fcol=next((c for c in ["feature","Feature","feature_name"] if c in shap_df.columns),None)
        icol=next((c for c in ["mean_abs_shap","Mean_Absolute_SHAP","mean_abs_shap_value"] if c in shap_df.columns),None)
        if fcol and icol:
            s=shap_df.copy()
            s[icol]=pd.to_numeric(s[icol],errors="coerce")
            s=s.dropna().sort_values(icol,ascending=False).head(20)
            st.plotly_chart(px.bar(s.sort_values(icol),x=icol,y=fcol,orientation="h",title="Top 20 features by mean absolute SHAP"),use_container_width=True)
            st.dataframe(s,use_container_width=True)
        else: st.error("SHAP export needs feature and mean absolute SHAP columns.")
    st.caption("SHAP describes how the fitted model uses features. It does not establish that a feature causes churn.")

elif page=="Customer 360":
    st.subheader("Individual customer risk and evidence")
    if not customer_col:
        st.warning("No customer ID column available.")
    else:
        ids=queue[customer_col].astype(str).tolist()
        chosen=st.selectbox("Select customer ID",ids)
        row=queue[queue[customer_col].astype(str)==chosen].iloc[0]
        a,b,c=st.columns(3)
        a.metric("Customer ID",chosen)
        b.metric("Predicted churn risk",pct(row[prob_col]))
        c.metric("Risk band",str(row["risk_band"]))
        evidence=[c for c in ["customer_segment","plan_tier","contract_type","region","tenure_days",
                  "usage_change_pct_30d","failed_payments_last_6m","unresolved_tickets","days_to_renewal",
                  "support_tickets_last_90d","avg_monthly_usage_hours","login_days_last_30d"] if c in queue.columns]
        st.markdown("**Available customer evidence**")
        st.dataframe(row[evidence].rename("Value").to_frame(),use_container_width=True)
        if cust_exp is not None and "customer_id" in cust_exp.columns:
            ex=cust_exp[cust_exp.customer_id.astype(str)==chosen]
            if not ex.empty:
                st.markdown("**Top local SHAP signals (validation customers)**")
                st.dataframe(ex,use_container_width=True)
        st.caption("The displayed evidence is for human review; it should not be treated as a confirmed cause of churn.")

elif page=="Methodology & Limitations":
    st.subheader("Problem definition and reproducibility")
    st.markdown("""
**Target definition:** `churn_30d` is the supplied binary target. This dashboard treats it as the 30-day churn label in the provided synthetic dataset. The dataset documentation should specify the precise event and prediction cutoff.

**Scoring:** Logistic Regression with C=0.1, lbfgs solver, class_weight=balanced. Numerical variables use median imputation and StandardScaler; categorical variables use most-frequent imputation and OneHotEncoder with unknown categories ignored.

**Intervention threshold:** User-configurable in the sidebar. Customers at or above the selected probability are flagged for human retention review.

**Retention capacity:** Top-k precision, recall, churn capture and lift are calculated from the held-out validation predictions.

**Explainability:** Global and local SHAP explanations indicate model associations, not causality.
""")
    st.markdown("**Validation results**")
    st.json(metadata.get("validation_metrics",{}))
    st.markdown("**Assumptions and limitations**")
    for note in metadata.get("notes",[]): st.warning(note)
    st.write(f"Dataset rows: {metadata.get('n_total_customers','N/A'):,} | Validation rows: {metadata.get('n_validation','N/A'):,} | Training rows: {metadata.get('n_training','N/A'):,}")
    st.write(f"Unique observation dates: {metadata.get('observation_date_unique_count','N/A')}")
    st.info("The provided dataset has a single common observation date. Therefore, this signup-date cohort split does not satisfy a true later-observation-period chronological holdout. A future-snapshot dataset is required to fully demonstrate that requirement.")

st.divider()
st.caption("Customer Churn Prediction & Retention Prioritization • Decision-support prototype • No automatic customer outreach or retention action")
