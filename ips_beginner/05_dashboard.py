"""
Step 5: the dashboard for your demo.
Run:  streamlit run 05_dashboard.py
Then open the address it prints (normally http://localhost:8501) in your browser.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Intrusion Prevention System", layout="wide")
st.title("Machine-Learning Intrusion Prevention System")
st.caption("Trained on CIC-IDS2017 network flows. Random Forest model that detects attacks and blocks the sources.")

needed = ["results/metrics.json", "results/simulation_summary.json", "models/ips_model.joblib", "prepared/data.joblib"]
missing = [p for p in needed if not os.path.exists(p)]
if missing:
    st.error("Some files are missing: " + ", ".join(missing) + ". Run steps 2, 3 and 4 first.")
    st.stop()


@st.cache_resource
def load_model_and_data():
    saved = joblib.load("models/ips_model.joblib")
    data = joblib.load("prepared/data.joblib")
    return saved["model"], saved["feature_names"], data["X_test"], data["y_test"].reset_index(drop=True)


metrics = json.load(open("results/metrics.json"))
sim = json.load(open("results/simulation_summary.json"))
model, feature_names, X_test, y_test = load_model_and_data()

tab1, tab2, tab3 = st.tabs(["1. Model performance", "2. Prevention simulation", "3. Try the model live"])

# ------------------------------------------------------------------ tab 1
with tab1:
    m = metrics["attack_vs_normal"]
    c = st.columns(5)
    c[0].metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
    c[1].metric("Precision", f"{m['precision'] * 100:.2f}%")
    c[2].metric("Recall", f"{m['recall'] * 100:.2f}%")
    c[3].metric("F1 score", f"{m['f1']:.3f}")
    c[4].metric("False alarms", f"{m['false_positive_rate'] * 100:.2f}%")
    st.write(f"Tested on **{metrics['rows_test']:,}** flows the model never saw during training "
             f"(trained on {metrics['rows_train']:,} flows, {metrics['trees']} decision trees).")

    left, right = st.columns(2)
    with left:
        st.subheader("Detection per attack type")
        rows = [{"Attack type": k, "Caught %": round(v["recall"] * 100, 1), "Test flows": int(v["support"])}
                for k, v in metrics["per_class"].items() if k != "BENIGN"]
        table = pd.DataFrame(rows).sort_values("Caught %", ascending=False)
        st.dataframe(table, hide_index=True, width="stretch")
    with right:
        st.subheader("Most useful features")
        feats = pd.DataFrame(metrics["top_features"], columns=["Feature", "Importance"]).set_index("Feature")
        st.bar_chart(feats)
    if os.path.exists("results/confusion_matrix.png"):
        st.image("results/confusion_matrix.png", caption="Confusion matrix (attack vs normal)", width=380)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write(f"Test flows arrive one by one. An address judged malicious **{sim['strikes_needed']} times in a row** is blocked, "
             "and everything it sends afterwards is dropped. (Addresses are simulated, because the dataset has none.)")
    c = st.columns(4)
    c[0].metric("Attackers blocked", f"{sim['attackers_blocked']} / {sim['attackers_total']}")
    c[1].metric("Normal clients wrongly blocked", f"{sim['normal_clients_wrongly_blocked']} / {sim['normal_clients_total']}")
    c[2].metric("Attack flows stopped", f"{sim['attack_flows_stopped_percent']:.1f}%")
    c[3].metric("Normal flows lost", f"{sim['normal_flows_dropped_percent']:.2f}%")

    if os.path.exists("results/timeline.csv"):
        tl = pd.read_csv("results/timeline.csv").set_index("flow_number")
        st.subheader("Attack flows over time")
        st.line_chart(tl[["attack_dropped", "attack_passed"]].rename(
            columns={"attack_dropped": "Attack flows dropped", "attack_passed": "Attack flows that got through"}))
    if os.path.exists("results/block_events.csv"):
        st.subheader("Block / unblock log")
        events = pd.read_csv("results/block_events.csv")
        st.dataframe(events, hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.write("Pick random flows from the test data and watch the model decide.")
    how_many = st.slider("How many flows", 5, 50, 15)
    if st.button("Classify random flows") or "demo_rows" not in st.session_state:
        st.session_state["demo_rows"] = np.random.choice(len(y_test), size=how_many, replace=False)
    idx = st.session_state["demo_rows"][:how_many]
    probabilities = model.predict_proba(X_test.iloc[idx])
    predicted = model.classes_[probabilities.argmax(axis=1)]
    result = pd.DataFrame({
        "Real type": y_test.iloc[idx].values,
        "Model says": predicted,
        "Confidence %": (probabilities.max(axis=1) * 100).round(1),
        "Decision": np.where(predicted == "BENIGN", "ALLOW", "BLOCK"),
        "Destination port": X_test.iloc[idx]["Destination Port"].astype(int).values,
        "Flow duration (us)": X_test.iloc[idx]["Flow Duration"].astype(int).values,
    })
    result["Correct?"] = np.where((result["Real type"] == "BENIGN") == (result["Model says"] == "BENIGN"), "yes", "NO")
    st.dataframe(result, hide_index=True, width="stretch")

    st.subheader("Or upload your own CSV")
    st.caption("It must have the same columns as the CIC-IDS2017 files (a Label column is optional).")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        raw = pd.read_csv(upload, encoding="latin1", low_memory=False)
        raw.columns = [str(c).strip() for c in raw.columns]
        absent = [c for c in feature_names if c not in raw.columns]
        if absent:
            st.error(f"Missing columns: {absent[:5]} ...")
        else:
            X_new = raw[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            out = pd.DataFrame({"Model says": model.predict(X_new)}, index=X_new.index)
            out["Decision"] = np.where(out["Model says"] == "BENIGN", "ALLOW", "BLOCK")
            st.write(out["Decision"].value_counts().to_frame("flows"))
            st.dataframe(out.head(200), width="stretch")
