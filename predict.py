import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import warnings
from datetime import datetime
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Court Case Priority Predictor",
    page_icon="⚖️",
    layout="wide"
)

# ==============================
# 🧠 LOAD TRAINED ARTIFACTS
# ==============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_priority_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    metadata = joblib.load("model_metadata.pkl")
    return model, label_encoders, feature_names, scaler, metadata

try:
    model, label_encoders, feature_names, scaler, metadata = load_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e}\nPlease ensure model artifacts are in the same directory.")
    st.stop()

# Sidebar Model Info
st.sidebar.header("🧠 Model Information")
st.sidebar.write(f"**Type:** {metadata['model_type']}")
st.sidebar.write(f"**Accuracy:** {metadata['accuracy']:.4f}")
st.sidebar.write(f"**Features:** {metadata['n_features']}")
st.sidebar.write(f"**Simplified Target:** {metadata.get('simplified_target', 'N/A')}")
st.sidebar.markdown("---")

# ==============================
# ⚙️ PREPROCESSING FUNCTION
# ==============================
def preprocess_case(df):
    df = df.copy()
    drop_cols = ["case_id", "cnr_number", "fir_number"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Handle date columns
    date_cols = ["filed_date", "last_hearing_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_days'] = (df[col] - pd.Timestamp("2000-01-01")).dt.days
            df = df.drop(columns=[col])

    # Derived / Interaction features
    if 'case_age_days' in df.columns and 'adjournments_count' in df.columns:
        df['delay_per_adjournment'] = df['case_age_days'] / (df['adjournments_count'] + 1)
    if 'undertrial_duration_months' in df.columns and 'evidence_complexity_score' in df.columns:
        df['complexity_duration_ratio'] = df['evidence_complexity_score'] * df['undertrial_duration_months']
    if 'number_of_petitioners' in df.columns and 'number_of_respondents' in df.columns:
        df['total_parties'] = df['number_of_petitioners'] + df['number_of_respondents']
        df['party_ratio'] = df['number_of_petitioners'] / (df['number_of_respondents'] + 1)

    # Fill numeric and categorical values
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    # Safe encoding
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")
        if col in label_encoders:
            le = label_encoders[col]
            if "Unknown" not in le.classes_:
                le.classes_ = np.append(le.classes_, "Unknown")
            unseen = set(df[col].unique()) - set(le.classes_)
            if unseen:
                st.warning(f"⚠️ Unseen labels in '{col}': {unseen}")
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
            df[col] = le.transform(df[col].astype(str))
        else:
            df[col] = pd.factorize(df[col])[0]

    # Ensure all required features
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0

    return df[feature_names]

# ==============================
# 🎯 PREDICTION FUNCTION
# ==============================
def predict_priority(df):
    X = preprocess_case(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    priority_map = {
        0: 'Very Low Priority',
        1: 'Low Priority',
        2: 'Medium Priority',
        3: 'High Priority',
        4: 'Critical Priority'
    }

    results = pd.DataFrame({
        'case_id': df.get('case_id', pd.Series(range(len(df)))),
        'predicted_priority': [priority_map[p] for p in preds],
        'priority_class': preds,
        'confidence': probs.max(axis=1)
    })

    # Add per-class probabilities (up to 5 classes)
    for i in range(probs.shape[1]):
        results[f'class_{i}_prob'] = probs[:, i]

    return results

# ==============================
# 🎨 Streamlit App Layout
# ==============================
st.title("⚖️ Court Case Priority Prediction Dashboard")
st.markdown("""
This dashboard uses a trained machine learning model to **predict court case priorities**  
based on various case, hearing, and litigant details.
""")

uploaded_file = st.file_uploader("📁 Upload your CSV file (e.g., `synthetic_legal_cases.csv`)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} cases from CSV.")
    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())

    if st.button("🔮 Predict Priorities"):
        with st.spinner("Analyzing and predicting priorities..."):
            results = predict_priority(df)
            st.success("✅ Predictions complete!")
            st.subheader("📊 Prediction Results")
            st.dataframe(results)

            # Charts
            st.subheader("📈 Priority Distribution")
            fig = px.pie(
                results,
                names="predicted_priority",
                title="Predicted Case Priority Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🎯 Confidence Histogram")
            fig2 = px.histogram(
                results,
                x="confidence",
                color="predicted_priority",
                nbins=20,
                title="Model Confidence Levels"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Download CSV
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="court_case_predictions.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Please upload a CSV file to start predictions.")

# ==============================
# 📋 Single Case Form (optional)
# ==============================
st.markdown("---")
st.subheader("🧾 Try Single Case Prediction")
with st.expander("Enter Case Details Manually"):
    district = st.text_input("District", "Jaipur")
    case_type = st.text_input("Case Type", "Family")
    evidence_score = st.number_input("Evidence Complexity Score", 1.0, 10.0, 4.0)
    case_age = st.number_input("Case Age (Days)", 0, 10000, 5000)
    adjournments = st.number_input("Adjournments Count", 0, 100, 10)
    respondents = st.number_input("Number of Respondents", 1, 10, 1)
    petitioners = st.number_input("Number of Petitioners", 1, 10, 3)

    if st.button("Predict Single Case Priority"):
        sample = pd.DataFrame([{
            "district": district,
            "case_type": case_type,
            "evidence_complexity_score": evidence_score,
            "case_age_days": case_age,
            "adjournments_count": adjournments,
            "number_of_petitioners": petitioners,
            "number_of_respondents": respondents
        }])
        result = predict_priority(sample)
        st.success("✅ Prediction Complete!")
        st.write(result)
        st.metric("Predicted Priority", result["predicted_priority"][0])
        st.metric("Confidence", f"{result['confidence'][0]*100:.2f}%")

st.sidebar.caption("Developed by Ctrl + Alt + Defeat")
